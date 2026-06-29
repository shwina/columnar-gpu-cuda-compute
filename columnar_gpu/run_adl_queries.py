import time
import awkward as ak
import cupy as cp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import os

# Reader is selectable via AK_BENCH_READER for A/B of read paths:
#   "shim" (default): pyarrow CPU read + to_backend (read/load NOT comparable, see NOTES.md)
#   "cudf":           GPU-direct cudf.read_parquet (requires cudf in env; read/load comparable)
if os.environ.get("AK_BENCH_READER", "shim").lower() == "cudf":
    import gpu_read_cudf as cudf
    from gpu_read_cudf import cudf_to_awkward
else:
    import cpu_read_shim as cudf  # cudf dropped from default: see cpu_read_shim.py
    from cpu_read_shim import cudf_to_awkward

import pyarrow.parquet as pq

import uproot
from coffea.jitters import hist as gpu_hist
import hist
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector

import pandas as df

print("cudf version", cudf.__version__)


####################################################################################################
### Helper functions ###


# Get the deltas between the times, and print
def get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill,quiet=False):
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_comp = t_after_comp-t_after_load
    dt_after_fill = t_after_fill-t_after_comp
    dt_tot        = t_after_fill-t0
    if not quiet:
        print(f"Time for q1: {dt_tot}")
        print(f"    Time for reading:   {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
        print(f"    Time for loading:   {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
        print(f"    Time for computing: {dt_after_comp} ({np.round(100*(dt_after_comp)/(dt_tot),1)}%)")
        print(f"    Time for histing:   {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")
    return([dt_after_read,dt_after_load,dt_after_comp,dt_after_fill,dt_tot])


# Make a plot comparing the hists from CPU and GPU query
def make_comp_plot(h1,h2=None, h1_tag="CPU",h2_tag="GPU", h1_clr="orange",h2_clr="blue", name="test"):

    fig, ax = plt.subplots(1, 1, figsize=(7,7))

    # Assumes this is the CPU one, so directly call plot1d
    h1.plot1d(linewidth=4,flow="none",color=h1_clr,label=h1_tag);

    # Assumes this is gpu, so call to_hist before plot1d
    if h2 is not None:
        h2.to_hist().plot1d(linewidth=1.5,flow="none",color=h2_clr,label=h2_tag);

    ax.legend(fontsize="21",framealpha=1,frameon=False)
    plt.title(name)

    print(f"Saving as plots/fig_{name}.png")
    fig.savefig(f"plots/fig_{name}.png")
    fig.savefig(f"plots/fig_{name}.pdf")


# Workaround for argmin since not implemented on GPU
# Only tested for axis=1 (i.e., innermost, for an array with 2 axes)
# Use at your own risk
def argmin_workaround_axis1(in_arr,axis,keepdims=False):
    if axis != 1:
        raise Exception("Not tested for axis other than 1")
    min_mask = in_arr == ak.min(in_arr,axis=axis)
    min_idx = ak.firsts(ak.local_index(in_arr)[min_mask])
    if keepdims:
        return(ak.singletons(min_idx))
    else:
        return min_idx

# Check if arrays agree
def arrays_agree(inarr1,inarr2,tag):
    arr1 = ak.to_backend(inarr1,"cpu")
    arr2 = ak.to_backend(inarr2,"cpu")

    # Check for exact agreement
    arr_agree = arr1 == arr2

    # Check for largest difference
    diff_arr = abs(arr1 - arr2)
    print("diff_arr",diff_arr)
    diff_arr = ak.fill_none(diff_arr,0)
    largest_diff = max(diff_arr)
    #threshold = 0
    threshold = 1e-6
    large_differences = diff_arr[diff_arr>threshold]
    frac_large_differences = len(large_differences)/len(arr1)

    idxmax = ak.argmax(diff_arr)
    #print("arr1:",arr1)
    #print("arr2:",arr2)
    print("val in arr1 of the max different:", f"{arr1[idxmax]:.20f}")
    print("val in arr2 of the max different:", f"{arr2[idxmax]:.20f}")
    print("large_differences:",large_differences)
    print("len large_differences:",len(large_differences))
    print("percent large_differences:",f"{np.round(frac_large_differences*100,2)}%")

    # Make plot
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.hist(diff_arr,bins=200,range=(min(diff_arr)-min(diff_arr)*0.2,max(diff_arr)*1.2))
    plt.text(0.19, 0.80, f"Tot entries: {len(diff_arr)}", dict(size=10),transform=fig1.transFigure)
    plt.text(0.19, 0.75, f"Absolute largest diff: {largest_diff}", dict(size=10),transform=fig1.transFigure)
    plt.text(0.19, 0.70, f"Absolute threshold: {threshold}", dict(size=10),transform=fig1.transFigure)
    plt.text(0.19, 0.65, f"Entries with absolute diff larger than threshold: {len(large_differences)}, {np.round(frac_large_differences*100,2)}%", dict(size=10),transform=fig1.transFigure)
    plt.title(f"Differences between CPU and gpu for {tag}")
    plt.yscale('log')
    fig1.savefig(os.path.join(f"plots/gpu_cpu_diff_hist_{tag}_log.png"),format="png")

    return(largest_diff)



####################################################################################################
### ADL queries ###


def _cc_fill(q_hist, axis_name, values):
    """Fill a coffea gpu_hist via cuda.compute.histogram_even (one fused CUB
    DeviceHistogram kernel) instead of coffea's clip/ravel/bincount chain.
    Enabled with AK_BENCH_HIST=cc; otherwise, or if cuda.compute is unavailable
    (e.g. the baseline env), falls back to q_hist.fill(...). Counts go into the
    in-range bins of the dense array; flow bins stay 0 (matches flow="none").
    """
    if os.environ.get("AK_BENCH_HIST") != "cc":
        q_hist.fill(**{axis_name: values}); return
    try:
        import cuda.compute as cc
    except Exception:
        q_hist.fill(**{axis_name: values}); return
    e = q_hist.axis(axis_name).edges()
    e = e.get() if hasattr(e, "get") else np.asarray(e)
    nbins = len(e) - 1; lo = float(e[0]); hi = float(e[-1])
    arr = ak.drop_none(ak.ravel(values))
    v = cp.ascontiguousarray(ak.to_cupy(arr).astype(cp.float32))
    counts = cp.zeros(nbins, dtype=cp.int32)
    cc.histogram_even(d_samples=v, d_histogram=counts, num_output_levels=nbins + 1,
                      lower_level=np.float32(lo), upper_level=np.float32(hi),
                      num_samples=int(v.size))
    dense = cp.zeros(q_hist._dense_shape, dtype=cp.float64)
    dense[1:nbins + 1] = counts.astype(cp.float64)
    q_hist._sumw[()] = dense


# Q1 query GPU
# Fill hist with met for all events
def query1_gpu(filepath,makeplot=False):

    print("\nStarting Q1 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["MET_pt"])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    MET_pt = cudf_to_awkward(table["MET_pt"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    # Time after compute (no actual compute in this query)
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    _cc_fill(q_hist, "met", MET_pt)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q1_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,MET_pt,dt_lst)



# Q1 query GPU, variant b: cuda.compute histogram_even instead of coffea bincount
#
# Same result as query1_gpu (per-bin counts), but the binning + counting are
# fused into a single CUB DeviceHistogram kernel instead of coffea's
# clip/floor/ravel_multi_index/bincount cupy chain. The samples are fed through
# a TransformIterator so any per-element transform would fuse into that same
# kernel (identity here, since Q1 histograms the raw column).
#
# Notes from bring-up:
#   * CounterT must be int32 (or uint64): CUB's atomic path has no signed
#     int64 atomicAdd overload, so an int64 d_histogram fails to NVRTC-compile.
#   * histogram_even bins are [lower, upper) with num_bins = num_output_levels-1,
#     out-of-range samples dropped -> matches coffea's flow="none" in-range bins.
def query1b_gpu(filepath,makeplot=False):

    import cuda.compute as cc                       # awkward3-only; imported lazily
    from cuda.compute.iterators import TransformIterator

    print("\nStarting Q1b code on gpu (cuda.compute histogram_even)..")

    NBINS, LO, HI = 100, 0.0, 200.0

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    MET_pt = cudf_to_awkward(table["MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    # No per-event compute in Q1
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    met_cp = ak.to_cupy(MET_pt)
    samples = TransformIterator(met_cp, lambda x: x)   # fusion hook (identity for Q1)
    counts = cp.zeros(NBINS, dtype=cp.int32)
    cc.histogram_even(
        d_samples=samples,
        d_histogram=counts,
        num_output_levels=NBINS + 1,
        lower_level=np.float32(LO),
        upper_level=np.float32(HI),
        num_samples=int(met_cp.size),
    )

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        edges = np.linspace(LO, HI, NBINS + 1)
        ax.stairs(counts.get(), edges, linewidth=1.5)
        ax.set_xlabel("$E_{T}^{miss}$ [GeV]")
        fig.savefig("plots/fig_q1b_gpu.png")

    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(counts,MET_pt,dt_lst)



# Q1 query CPU
# Fill hist with met for all events
def query1_cpu(filepath,makeplot=False):

    # Fill hist with met for all events
    print("\nStarting Q1 code on cpu..")

    # Get met pt and fill hist

    # Time t0
    t0 = time.time()

    table = pq.read_table(filepath, columns = ["MET_pt"])
    #table = ak.from_parquet(filepath, columns = ["MET_pt"])

    # Time after read
    t_after_read = time.time()

    MET_pt = ak.Array(table["MET_pt"])

    # Time after load
    t_after_load = time.time()

    # Time after compute (no actual compute in this query)
    t_after_comp = time.time()

    q_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    q_hist.fill(met=MET_pt)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q1_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,MET_pt,dt_lst)



# Q2 query GPU
# Fill hist with pt for all jets
def query2_gpu(filepath,makeplot=False):

    print("\nStarting Q2 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt"])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Jet_pt = cudf_to_awkward(table["Jet_pt"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    # Time after compute (no actual compute in this query)
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    fillarr = ak.flatten(Jet_pt)
    _cc_fill(q_hist, "ptj", fillarr)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q2_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q2 query CPU
# Fill hist with pt for all jets
def query2_cpu(filepath,makeplot=False):

    print("\nStarting Q2 code on cpu..")

    # Time t0
    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt"])
    #table = ak.from_parquet(filepath, columns = ["Jet_pt"])

    # Time after read
    t_after_read = time.time()

    Jet_pt = ak.Array(table["Jet_pt"])

    # Time after load
    t_after_load = time.time()

    # Time after compute (no actual compute in this query)
    t_after_comp = time.time()

    q_hist = hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]").Double()
    fillarr = ak.flatten(Jet_pt)
    q_hist.fill(ptj=fillarr)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q2_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q3 query GPU
# Fill a hist with pt of jets with eta less than 1
def query3_gpu(filepath,makeplot=False):

    print("\nStarting Q3 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    fillarr = ak.flatten(Jet_pt[abs(Jet_eta) < 1.0])

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    _cc_fill(q_hist, "ptj", fillarr)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q3_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q3 query CPU
# Fill a hist with pt of jets with eta less than 1
def query3_cpu(filepath,makeplot=False):

    print("\nStarting Q3 code on cpu..")

    # Time t0
    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt", "Jet_eta"])
    #table = ak.from_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])

    # Time after read
    t_after_read = time.time()

    Jet_pt = ak.Array(table["Jet_pt"])
    Jet_eta = ak.Array(table["Jet_eta"])

    # Time after load
    t_after_load = time.time()

    fillarr = ak.flatten(Jet_pt[abs(Jet_eta) < 1.0])

    # Time after compute
    t_after_comp = time.time()

    q_hist = hist.new.Reg(
        100, 0, 200, 
        name="ptj", 
        label="Jet $p_{T}$ [GeV]"
    ).Double()
    q_hist.fill(ptj=fillarr)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q3_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q4 query GPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_gpu(filepath,makeplot=False):

    print("\nStarting Q4 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    MET_pt = cudf_to_awkward(table["MET_pt"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    fillarr = MET_pt[has2jets]

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    _cc_fill(q_hist, "met", fillarr)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q4_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)


# Q4 query CPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_cpu(filepath,makeplot=False):

    print("\nStarting Q4 code on cpu..")

    # Time t0
    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt", "MET_pt"])
    #table = ak.from_parquet(filepath, columns = ["Jet_pt", "MET_pt"])

    # Time after read
    t_after_read = time.time()

    Jet_pt = ak.Array(table["Jet_pt"])
    MET_pt = ak.Array(table["MET_pt"])

    # Time after load
    t_after_load = time.time()

    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    fillarr = MET_pt[has2jets]

    # Time after compute
    t_after_comp = time.time()

    q_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    q_hist.fill(met=fillarr)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        qhist.plot1d(flow="none");
        fig.savefig("plots/fig_q4_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q5 query GPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_gpu(filepath,makeplot=False):

    print("\nStarting Q5 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(
        filepath,
        columns = [
            "MET_pt",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    )

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    MET_pt = cudf_to_awkward(table["MET_pt"])
    Muon_pt = cudf_to_awkward(table["Muon_pt"])
    Muon_eta = cudf_to_awkward(table["Muon_eta"])
    Muon_phi = cudf_to_awkward(table["Muon_phi"])
    Muon_mass = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    fillarr = MET_pt[goodevent]

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    _cc_fill(q_hist, "met", fillarr)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q5_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)



# Q5 query CPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_cpu(filepath,makeplot=False):

    print("\nStarting Q5 code on cpu..")

    # Time t0
    t0 = time.time()

    #table = ak.from_parquet(
    table = pq.read_table(
        filepath,
        columns = [
            "MET_pt",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    )

    # Time after read
    t_after_read = time.time()

    MET_pt      = ak.Array(table["MET_pt"])
    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])

    # Time after load
    t_after_load = time.time()

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    fillarr = MET_pt[goodevent]

    # Time after compute
    t_after_comp = time.time()

    q_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    q_hist.fill(met=fillarr)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q5_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,fillarr,dt_lst)


# Q6 query GPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_gpu(filepath,makeplot=False):

    print("\nStarting Q6 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag",])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Jet_pt   = cudf_to_awkward(table["Jet_pt"])
    Jet_eta  = cudf_to_awkward(table["Jet_eta"])
    Jet_phi  = cudf_to_awkward(table["Jet_phi"])
    Jet_mass = cudf_to_awkward(table["Jet_mass"])
    Jet_btag = cudf_to_awkward(table["Jet_btag"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    jets = ak.zip(
        {
            "pt"  : Jet_pt,
            "eta" : Jet_eta,
            "phi" : Jet_phi,
            "mass": Jet_mass,
            "btag": Jet_btag,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    # Get the pt of the trijet system closest to top
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3

    trijet_t = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet_t.j1.btag,
        np.maximum(
            trijet_t.j2.btag,
            trijet_t.j3.btag,
        ),
    )
    fillarr_1 = trijet_t.p4.pt

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    q_hist_1 = gpu_hist.Hist("Counts", gpu_hist.Bin("pt3j", "Trijet $p_{T}$ [GeV]", 100, 0, 200))
    _cc_fill(q_hist_1, "pt3j", fillarr_1)

    q_hist_2 = gpu_hist.Hist("Counts", gpu_hist.Bin("btag", "Max jet b-tag score", 100, -10, 1))
    _cc_fill(q_hist_2, "btag", maxBtag)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist_1.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q6p1_gpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist_2.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q6p2_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist_1, q_hist_2, fillarr_1, maxBtag, dt_lst)


# Q6 (GPU), chunked over events so the trijet combinations don't exceed GPU memory
# at very large N. Combinations are within-event, so batching by events is exact and
# the two histograms simply accumulate across batches. Batch size via env AK_Q6_BATCH.
def query6_gpu_chunked(filepath, makeplot=False):

    print("\nStarting Q6 code on gpu (chunked)..")
    batch = int(os.environ.get("AK_Q6_BATCH", "1000000"))

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag",])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    jets = ak.zip(
        {
            "pt"  : cudf_to_awkward(table["Jet_pt"]),
            "eta" : cudf_to_awkward(table["Jet_eta"]),
            "phi" : cudf_to_awkward(table["Jet_phi"]),
            "mass": cudf_to_awkward(table["Jet_mass"]),
            "btag": cudf_to_awkward(table["Jet_btag"]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    q_hist_1 = gpu_hist.Hist("Counts", gpu_hist.Bin("pt3j", "Trijet $p_{T}$ [GeV]", 100, 0, 200))
    q_hist_2 = gpu_hist.Hist("Counts", gpu_hist.Bin("btag", "Max jet b-tag score", 100, -10, 1))

    n = len(jets)
    pt_parts, btag_parts = [], []
    for i in range(0, n, batch):
        jb = jets[i:i+batch]
        trijet = ak.combinations(jb, 3, fields=["j1", "j2", "j3"])
        trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
        trijet_t = ak.flatten(
            trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
        )
        maxBtag_b = np.maximum(trijet_t.j1.btag, np.maximum(trijet_t.j2.btag, trijet_t.j3.btag))
        pt_b = trijet_t.p4.pt
        q_hist_1.fill(pt3j=pt_b)        # histograms accumulate across batches
        q_hist_2.fill(btag=maxBtag_b)
        pt_parts.append(pt_b)
        btag_parts.append(maxBtag_b)

    fillarr_1 = ak.concatenate(pt_parts) if pt_parts else pt_parts
    maxBtag = ak.concatenate(btag_parts) if btag_parts else btag_parts

    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()
    # (histograms already filled inside the loop)
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    dt_lst = get_dt(t0, t_after_read, t_after_load, t_after_comp, t_after_fill)
    return(q_hist_1, q_hist_2, fillarr_1, maxBtag, dt_lst)


# Q6 query CPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_cpu(filepath,makeplot=False):

    print("\nStarting Q6 code on cpu..")

    # Time t0
    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_btag"])
    #table = ak.from_parquet(filepath, columns = ["Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_btag"])

    # Time after read
    t_after_read = time.time()

    Jet_pt   = ak.Array(table["Jet_pt"])
    Jet_eta  = ak.Array(table["Jet_eta"])
    Jet_phi  = ak.Array(table["Jet_phi"])
    Jet_mass = ak.Array(table["Jet_mass"])
    Jet_btag = ak.Array(table["Jet_btag"])

    # Time after load
    t_after_load = time.time()

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
            "btag": Jet_btag,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    # Get the pt of the trijet system closest to top
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3

    trijet_t = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet_t.j1.btag,
        np.maximum(
            trijet_t.j2.btag,
            trijet_t.j3.btag,
        ),
    )
    fillarr_1 = trijet_t.p4.pt

    # Time after compute
    t_after_comp = time.time()

    q_hist_1 = hist.new.Reg(100, 0, 200, name="pt3j", label="Trijet $p_{T}$ [GeV]").Double()
    q_hist_1.fill(pt3j=fillarr_1)

    q_hist_2 = hist.new.Reg(100, -10, 1, name="btag", label="Max jet b-tag score").Double()
    q_hist_2.fill(btag=maxBtag)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist_1.plot1d(flow="none");
        fig.savefig("plots/fig_q6p1_cpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist_2.plot1d(flow="none");
        fig.savefig("plots/fig_q6p2_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist_1, q_hist_2, fillarr_1, maxBtag, dt_lst)


# Q7 query GPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_gpu(filepath,makeplot=False):

    print("\nStarting Q7 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"
    ])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Jet_pt          = cudf_to_awkward(table["Jet_pt"])
    Jet_eta         = cudf_to_awkward(table["Jet_eta"])
    Jet_phi         = cudf_to_awkward(table["Jet_phi"])
    Jet_mass        = cudf_to_awkward(table["Jet_mass"])

    Muon_pt         = cudf_to_awkward(table["Muon_pt"])
    Muon_eta        = cudf_to_awkward(table["Muon_eta"])
    Muon_phi        = cudf_to_awkward(table["Muon_phi"])
    Muon_mass       = cudf_to_awkward(table["Muon_mass"])
    Muon_charge     = cudf_to_awkward(table["Muon_charge"])

    Electron_pt     = cudf_to_awkward(table["Electron_pt"])
    Electron_eta    = cudf_to_awkward(table["Electron_eta"])
    Electron_phi    = cudf_to_awkward(table["Electron_phi"])
    Electron_mass   = cudf_to_awkward(table["Electron_mass"])
    Electron_charge = cudf_to_awkward(table["Electron_charge"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')
    leptons_good = leptons[leptons.pt>10]

    # Get good jets and sum pt to get HT
    jet_nearest_to_any_lep, dr = jets.nearest(leptons,return_metric=True)
    jets_good = jets[(jets.pt>30) & (dr>0.4)]
    ht = ak.sum(jets_good.pt,axis=1)

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    # Fill hist
    q_hist = gpu_hist.Hist("Counts", gpu_hist.Bin("sumjetpt", "Scalar sum of jet $p_{T}$ [GeV]", 100, 0, 200))
    _cc_fill(q_hist, "sumjetpt", ht)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q7_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,ht,dt_lst)


# Q7 query CPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_cpu(filepath,makeplot=False):

    print("\nStarting Q7 code on cpu..")

    # Time t0
    t0 = time.time()

    #table = ak.from_parquet(filepath, columns = [
    table = pq.read_table(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"
    ])

    # Time after read
    t_after_read = time.time()

    Jet_pt   = ak.Array(table["Jet_pt"])
    Jet_eta  = ak.Array(table["Jet_eta"])
    Jet_phi  = ak.Array(table["Jet_phi"])
    Jet_mass = ak.Array(table["Jet_mass"])

    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])

    Electron_pt     = ak.Array(table["Electron_pt"])
    Electron_eta    = ak.Array(table["Electron_eta"])
    Electron_phi    = ak.Array(table["Electron_phi"])
    Electron_mass   = ak.Array(table["Electron_mass"])
    Electron_charge = ak.Array(table["Electron_charge"])

    # Time after load
    t_after_load = time.time()

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')
    leptons_good = leptons[leptons.pt>10]

    # Get good jets and sum pt to get HT
    jet_nearest_to_any_lep, dr = jets.nearest(leptons,return_metric=True)
    jets_good = jets[(jets.pt>30) & (dr>0.4)]
    ht = ak.sum(jets_good.pt,axis=1)

    # Time after compute
    t_after_comp = time.time()

    # Fill hist
    q_hist = hist.new.Reg(100, 0, 200, name="sumjetpt", label="Scalar sum of jet $p_{T}$ [GeV]").Double()
    q_hist.fill(sumjetpt=ht)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q7_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,ht,dt_lst)


# Q8 query GPU
# Select events with at least 3 leptons, that inlude a SFOS pair
# Plot MT of the system of the leading non-Z lepton and MET
def query8_gpu(filepath,makeplot=False):

    print("\nStarting Q8 code on gpu..")

    # Time t0
    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "MET_pt", "MET_phi",
    ])

    # Time after read
    cp.cuda.Device(0).synchronize()
    t_after_read = time.time()

    Muon_pt     = cudf_to_awkward(table["Muon_pt"])
    Muon_eta    = cudf_to_awkward(table["Muon_eta"])
    Muon_phi    = cudf_to_awkward(table["Muon_phi"])
    Muon_mass   = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])

    Electron_pt     = cudf_to_awkward(table["Electron_pt"])
    Electron_eta    = cudf_to_awkward(table["Electron_eta"])
    Electron_phi    = cudf_to_awkward(table["Electron_phi"])
    Electron_mass   = cudf_to_awkward(table["Electron_mass"])
    Electron_charge = cudf_to_awkward(table["Electron_charge"])

    MET_pt  = cudf_to_awkward(table["MET_pt"])
    MET_phi = cudf_to_awkward(table["MET_phi"])

    # Time after load
    cp.cuda.Device(0).synchronize()
    t_after_load = time.time()

    MET = ak.zip(
        {
            "pt": MET_pt,
            "phi": MET_phi,
        },
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
            "pdgId": -11 * Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
            "pdgId": -13 * Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')

    # Attatch index to each lepton
    leptons['idx'] = ak.local_index(leptons, axis=1)

    # Get pairs of leptons
    ll_pairs = ak.combinations(leptons, 2, fields=["l0","l1"])
    ll_pairs_idx = ak.argcombinations(leptons, 2, fields=["l0","l1"])

    # Get distance from Z
    dist_from_z_all_pairs = abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)

    # Mask out the pairs that are not SFOS (so that we don't include them when finding the one that's closest to Z)
    # And then of the SFOS pairs, get the index of the one that's cosest to the Z
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    dist_from_z_sfos_pairs = ak.mask(dist_from_z_all_pairs,sfos_mask)
    sfos_pair_closest_to_z_idx = ak.argmin(dist_from_z_sfos_pairs,axis=-1,keepdims=True)

    # Build a mask (of the shape of the original lep array) corresponding to the leps that are part of the Z candidate
    mask_is_z_lep = (leptons.idx == ak.flatten(ll_pairs_idx.l0[sfos_pair_closest_to_z_idx]))
    mask_is_z_lep = (mask_is_z_lep | (leptons.idx == ak.flatten(ll_pairs_idx.l1[sfos_pair_closest_to_z_idx])))
    mask_is_z_lep = ak.fill_none(mask_is_z_lep, False)

    # Get ahold of the leading non-Z lepton
    leps_not_from_z_candidate = leptons[~mask_is_z_lep]
    lead_lep_not_from_z_candidate = leps_not_from_z_candidate[ak.argmax(leps_not_from_z_candidate.pt, axis=1, keepdims=True)]
    lead_lep_not_from_z_candidate = lead_lep_not_from_z_candidate[:,0] # Go from e.g. [None,[lepton object]] to [None,lepton object]

    # Get the MT
    mt = np.sqrt(2 * lead_lep_not_from_z_candidate.pt * MET.pt * (1 - np.cos(MET.delta_phi(lead_lep_not_from_z_candidate))))

    # Apply 3l SFOS selection
    has_3l = ak.num(leptons) >=3
    has_sfos = ak.any(sfos_mask,axis=1)
    mt = mt[has_3l & has_sfos]

    # Time after compute
    cp.cuda.Device(0).synchronize()
    t_after_comp = time.time()

    # Fill hist
    q_hist = gpu_hist.Hist("Counts", gpu_hist.Bin("mt_lep_met", "lep-MET transverse mass [GeV]", 100, 0, 200))
    _cc_fill(q_hist, "mt_lep_met", mt)

    # Time after fill
    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q8_gpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,mt,dt_lst)


# Q8 query CPU
# Select events with at least 3 leptons, that inlude a SFOS pair
# Plot MT of the system of the leading non-Z lepton and MET
def query8_cpu(filepath,makeplot=False):

    print("\nStarting Q8 code on cpu..")

    # Time t0
    t0 = time.time()

    #table = ak.from_parquet(filepath, columns = [
    table = pq.read_table(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "MET_pt", "MET_phi",
    ])

    # Time after read
    t_after_read = time.time()

    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])

    Electron_pt     = ak.Array(table["Electron_pt"])
    Electron_eta    = ak.Array(table["Electron_eta"])
    Electron_phi    = ak.Array(table["Electron_phi"])
    Electron_mass   = ak.Array(table["Electron_mass"])
    Electron_charge = ak.Array(table["Electron_charge"])

    MET_pt = ak.Array(table["MET_pt"])
    MET_phi = ak.Array(table["MET_phi"])

    # Time after load
    t_after_load = time.time()

    MET = ak.zip(
        {
            "pt": MET_pt,
            "phi": MET_phi,
        },
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
            "pdgId": -11 * Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
            "pdgId": -13 * Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')

    # Attatch index to each lepton
    leptons['idx'] = ak.local_index(leptons, axis=1)

    # Get pairs of leptons
    ll_pairs = ak.combinations(leptons, 2, fields=["l0","l1"])
    ll_pairs_idx = ak.argcombinations(leptons, 2, fields=["l0","l1"])

    # Get distance from Z
    dist_from_z_all_pairs = abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)

    # Mask out the pairs that are not SFOS (so that we don't include them when finding the one that's closest to Z)
    # And then of the SFOS pairs, get the index of the one that's cosest to the Z
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    dist_from_z_sfos_pairs = ak.mask(dist_from_z_all_pairs,sfos_mask)
    sfos_pair_closest_to_z_idx = ak.argmin(dist_from_z_sfos_pairs,axis=-1,keepdims=True)

    # Build a mask (of the shape of the original lep array) corresponding to the leps that are part of the Z candidate
    mask_is_z_lep = (leptons.idx == ak.flatten(ll_pairs_idx.l0[sfos_pair_closest_to_z_idx]))
    mask_is_z_lep = (mask_is_z_lep | (leptons.idx == ak.flatten(ll_pairs_idx.l1[sfos_pair_closest_to_z_idx])))
    mask_is_z_lep = ak.fill_none(mask_is_z_lep, False)

    # Get ahold of the leading non-Z lepton
    leps_not_from_z_candidate = leptons[~mask_is_z_lep]
    lead_lep_not_from_z_candidate = leps_not_from_z_candidate[ak.argmax(leps_not_from_z_candidate.pt, axis=1, keepdims=True)]
    lead_lep_not_from_z_candidate = lead_lep_not_from_z_candidate[:,0] # Go from e.g. [None,[lepton object]] to [None,lepton object]

    # Get the MT
    mt = np.sqrt(2 * lead_lep_not_from_z_candidate.pt * MET.pt * (1 - np.cos(MET.delta_phi(lead_lep_not_from_z_candidate))))

    # Apply 3l SFOS selection
    has_3l = ak.num(leptons) >=3
    has_sfos = ak.any(sfos_mask,axis=1)
    mt = mt[has_3l & has_sfos]

    # Time after compute
    t_after_comp = time.time()

    # Fill hist
    q_hist = hist.new.Reg(100, 0, 200, name="mt_lep_met", label="lep-MET transverse mass [GeV]").Double()
    q_hist.fill(mt_lep_met=mt)

    # Time after fill
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q8_cpu.png")

    # Get and print timing information
    dt_lst = get_dt(t0,t_after_read,t_after_load,t_after_comp,t_after_fill)

    return(q_hist,mt,dt_lst)



####################################################################################################
### cuda.compute-native rewrites of the host-bound queries (Q3, Q7) ###
#
# These express the query directly in cuda.compute primitives instead of awkward's
# generic jagged-getitem / cross-join machinery, collapsing the per-op kernel+sync
# soup. Bit-identical histograms to query3_gpu / query7_gpu; see perf/reports/PERF_REPORT.md.
# Named with a "c" suffix; run via bench_driver.py 3c / 7c.

def _jagged_buffers(arr):
    """(offsets int64, content) cupy buffers of a 1-level jagged cuda ak.Array."""
    lay = arr.layout
    return cp.asarray(lay.offsets.data).astype(cp.int64), cp.ascontiguousarray(cp.asarray(lay.content.data))


# Q3 (cuda.compute): flatten(Jet_pt[|Jet_eta|<1]) is pure stream compaction -> one DeviceSelect.
def query3c_gpu(filepath, makeplot=False):
    import cuda.compute as cc
    from cuda.compute import select
    from cuda.compute.iterators import CountingIterator
    print("\nStarting Q3c code on gpu (cuda.compute select)..")

    cp.cuda.Device(0).synchronize(); t0 = time.time()
    table = cudf.read_parquet(filepath, columns=["Jet_pt", "Jet_eta"])
    cp.cuda.Device(0).synchronize(); t_after_read = time.time()

    Jet_pt = cudf_to_awkward(table["Jet_pt"]); Jet_eta = cudf_to_awkward(table["Jet_eta"])
    pt_c = cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_pt)).astype(cp.float32))
    eta_c = cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_eta)).astype(cp.float32))
    cp.cuda.Device(0).synchronize(); t_after_load = time.time()

    N = pt_c.size
    def keep(i):
        return np.uint8(1) if abs(eta_c[i]) < np.float32(1.0) else np.uint8(0)
    idx = cp.empty(N, dtype=cp.int64); nsel = cp.empty(1, dtype=cp.int64)
    select(d_in=CountingIterator(np.int64(0)), d_out=idx, d_num_selected_out=nsel, cond=keep, num_items=N)
    fillarr = pt_c[idx[:int(nsel[0])]]
    cp.cuda.Device(0).synchronize(); t_after_comp = time.time()

    counts = cp.zeros(100, dtype=cp.int32)
    cc.histogram_even(d_samples=cp.ascontiguousarray(fillarr), d_histogram=counts,
                      num_output_levels=101, lower_level=np.float32(0.0),
                      upper_level=np.float32(200.0), num_samples=int(fillarr.size))
    cp.cuda.Device(0).synchronize(); t_after_fill = time.time()

    dt_lst = get_dt(t0, t_after_read, t_after_load, t_after_comp, t_after_fill)
    return (counts, ak.Array(fillarr), dt_lst)


# Q7 (cuda.compute): jets.nearest(leptons) HT -> per-jet segmented_reduce(MIN) of dR over the
# event's leptons, then per-event segmented_reduce(PLUS) for HT, then histogram_even.
def query7c_gpu(filepath, makeplot=False):
    import cuda.compute as cc
    from cuda.compute import segmented_reduce, OpKind
    from math import sqrt, pi as MPI
    print("\nStarting Q7c code on gpu (cuda.compute segmented_reduce)..")

    cp.cuda.Device(0).synchronize(); t0 = time.time()
    table = cudf.read_parquet(filepath, columns=[
        "Jet_pt", "Jet_eta", "Jet_phi", "Electron_eta", "Electron_phi", "Muon_eta", "Muon_phi"])
    cp.cuda.Device(0).synchronize(); t_after_read = time.time()

    g = cudf_to_awkward
    off_j, pt_j = _jagged_buffers(g(table["Jet_pt"]));  pt_j = pt_j.astype(cp.float32)
    _, eta_j = _jagged_buffers(g(table["Jet_eta"]));    eta_j = eta_j.astype(cp.float32)
    _, phi_j = _jagged_buffers(g(table["Jet_phi"]));    phi_j = phi_j.astype(cp.float32)
    lep_eta = ak.to_packed(ak.concatenate([g(table["Electron_eta"]), g(table["Muon_eta"])], axis=1))
    lep_phi = ak.to_packed(ak.concatenate([g(table["Electron_phi"]), g(table["Muon_phi"])], axis=1))
    off_l, eta_l = _jagged_buffers(lep_eta); eta_l = eta_l.astype(cp.float32)
    _, phi_l = _jagged_buffers(lep_phi);     phi_l = phi_l.astype(cp.float32)
    cp.cuda.Device(0).synchronize(); t_after_load = time.time()

    nev = off_j.size - 1; J = pt_j.size
    n_jet = cp.diff(off_j); n_lep = cp.diff(off_l)
    event_of_jet = cp.repeat(cp.arange(nev, dtype=cp.int64), n_jet)
    l_per_jet = n_lep[event_of_jet]
    pair_off = cp.zeros(J + 1, dtype=cp.int64); cp.cumsum(l_per_jet, out=pair_off[1:])
    P = int(pair_off[-1])
    jet_of_pair = cp.repeat(cp.arange(J, dtype=cp.int64), l_per_jet)
    local = cp.arange(P, dtype=cp.int64) - pair_off[jet_of_pair]
    lep_of_pair = off_l[event_of_jet[jet_of_pair]] + local
    deta = eta_j[jet_of_pair] - eta_l[lep_of_pair]
    dphi = phi_j[jet_of_pair] - phi_l[lep_of_pair]
    dphi = (dphi + np.float32(MPI)) % (np.float32(2 * MPI)) - np.float32(MPI)
    dr = cp.sqrt(deta * deta + dphi * dphi).astype(cp.float32)
    dr_min = cp.empty(J, dtype=cp.float32)
    segmented_reduce(d_in=dr, d_out=dr_min, num_segments=J,
                     start_offsets_in=pair_off[:-1], end_offsets_in=pair_off[1:],
                     op=OpKind.MINIMUM, h_init=np.array([np.inf], dtype=np.float32),
                     max_segment_size=int(n_lep.max()) if nev else 1)
    dr_min[l_per_jet == 0] = np.float32(-1.0)
    contrib = cp.where((pt_j > 30) & (dr_min > 0.4), pt_j, cp.float32(0)).astype(cp.float32)
    ht = cp.empty(nev, dtype=cp.float32)
    segmented_reduce(d_in=contrib, d_out=ht, num_segments=nev,
                     start_offsets_in=off_j[:-1], end_offsets_in=off_j[1:],
                     op=OpKind.PLUS, h_init=np.array([0], dtype=np.float32),
                     max_segment_size=int(n_jet.max()) if nev else 1)
    cp.cuda.Device(0).synchronize(); t_after_comp = time.time()

    counts = cp.zeros(100, dtype=cp.int32)
    cc.histogram_even(d_samples=cp.ascontiguousarray(ht), d_histogram=counts,
                      num_output_levels=101, lower_level=np.float32(0.0),
                      upper_level=np.float32(200.0), num_samples=int(ht.size))
    cp.cuda.Device(0).synchronize(); t_after_fill = time.time()

    dt_lst = get_dt(t0, t_after_read, t_after_load, t_after_comp, t_after_fill)
    return (counts, ak.Array(ht), dt_lst)


####################################################################################################


def main():

    # File paths
    ## https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.ipynb
    ##root_filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    ##filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
    #filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_100k.parquet"
    #filepath = "tmp_pq/test_pq_100k.parquet"
    filepath = os.environ.get("AK_BENCH_PARQUET", "/home/coder/columnar_gpu_bench/data/pq_subset_100k.parquet")
    #filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"

    # Print the number of events we are running over
    nevents = len(df.read_parquet(filepath, columns = ["MET_pt"]))
    print(f"\nNumber of nevents to be processed: {nevents}")
    print(f"\n\n########### Running the ADL queries ###########\n")

    # Run the benchmark queries on GPU
    hist_q1_gpu, arr_q1_gpu, t_q1_gpu = query1_gpu(filepath)
    hist_q2_gpu, arr_q2_gpu, t_q2_gpu = query2_gpu(filepath)
    hist_q3_gpu, arr_q3_gpu, t_q3_gpu = query3_gpu(filepath)
    hist_q4_gpu, arr_q4_gpu, t_q4_gpu = query4_gpu(filepath)
    hist_q5_gpu, arr_q5_gpu, t_q5_gpu = query5_gpu(filepath)
    hist_q6p1_gpu, hist_q6p2_gpu, arr_q6p1_gpu, arr_q6p2_gpu, t_q6_gpu = query6_gpu(filepath)
    hist_q7_gpu, arr_q7_gpu, t_q7_gpu = query7_gpu(filepath)
    hist_q8_gpu, arr_q8_gpu, t_q8_gpu = query8_gpu(filepath)
    #exit()

    # Run the benchmark queries on CPU
    hist_q1_cpu,   arr_q1_cpu, t_q1_cpu = query1_cpu(filepath)
    hist_q2_cpu,   arr_q2_cpu, t_q2_cpu = query2_cpu(filepath)
    hist_q3_cpu,   arr_q3_cpu, t_q3_cpu = query3_cpu(filepath)
    hist_q4_cpu,   arr_q4_cpu, t_q4_cpu = query4_cpu(filepath)
    hist_q5_cpu,   arr_q5_cpu, t_q5_cpu = query5_cpu(filepath)
    hist_q6p1_cpu, hist_q6p2_cpu, arr_q6p1_cpu, arr_q6p2_cpu, t_q6_cpu = query6_cpu(filepath)
    hist_q7_cpu,   arr_q7_cpu, t_q7_cpu = query7_cpu(filepath)
    hist_q8_cpu,   arr_q8_cpu, t_q8_cpu = query8_cpu(filepath)
    #exit()

    # Check for event-by-event agreement of the output arrays
    print(f"\n\n########### Check event-by-event agreement of the output arrays ###########\n")
    print("q1 largest difference:",arrays_agree(arr_q1_gpu,arr_q1_cpu,"q1"),"\n")
    print("q2 largest difference:",arrays_agree(arr_q2_gpu,arr_q2_cpu,"q2"),"\n")
    print("q3 largest difference:",arrays_agree(arr_q3_gpu,arr_q3_cpu,"q3"),"\n")
    print("q4 largest difference:",arrays_agree(arr_q4_gpu,arr_q4_cpu,"q4"),"\n")
    print("q5 largest difference:",arrays_agree(arr_q5_gpu,arr_q5_cpu,"q5"),"\n")
    print("q6 largest difference:",arrays_agree(arr_q6p1_gpu,arr_q6p1_cpu,"q6p1"),"\n")
    print("q6 largest difference:",arrays_agree(arr_q6p2_gpu,arr_q6p2_cpu,"q6p2"),"\n")
    print("q7 largest difference:",arrays_agree(arr_q7_gpu,arr_q7_cpu,"q7"),"\n")
    print("q8 largest difference:",arrays_agree(arr_q8_gpu,arr_q8_cpu,"q8"),"\n")
    #exit()

    # Print the times in a way we can easily paste as the plotting inputs
    print(f"\n\n########### Timing info for this run over {nevents} events (for plotting) ###########\n")
    print(f"gpu:\n{[t_q1_gpu,t_q2_gpu,t_q3_gpu,t_q4_gpu,t_q5_gpu,t_q6_gpu,t_q7_gpu,t_q8_gpu]},")
    print(f"cpu:\n{[t_q1_cpu,t_q2_cpu,t_q3_cpu,t_q4_cpu,t_q5_cpu,t_q6_cpu,t_q7_cpu,t_q8_cpu]},")
    #exit()

    # Plotting the query output histos
    print("\n\n########### Making plots ###########\n")
    make_comp_plot(h1=hist_q1_cpu,   h2=hist_q1_gpu,   name=f"query1_nevents{nevents}")
    make_comp_plot(h1=hist_q2_cpu,   h2=hist_q2_gpu,   name=f"query2_nevents{nevents}")
    make_comp_plot(h1=hist_q3_cpu,   h2=hist_q3_gpu,   name=f"query3_nevents{nevents}")
    make_comp_plot(h1=hist_q4_cpu,   h2=hist_q4_gpu,   name=f"query4_nevents{nevents}")
    make_comp_plot(h1=hist_q5_cpu,   h2=hist_q5_gpu,   name=f"query5_nevents{nevents}")
    make_comp_plot(h1=hist_q6p1_cpu, h2=hist_q6p1_gpu, name=f"query6_part1_nevents{nevents}")
    make_comp_plot(h1=hist_q6p2_cpu, h2=hist_q6p2_gpu, name=f"query6_part2_nevents{nevents}")
    make_comp_plot(h1=hist_q7_cpu,   h2=hist_q7_gpu,   name=f"query7_nevents{nevents}")
    make_comp_plot(h1=hist_q8_cpu,   h2=hist_q8_gpu,   name=f"query8_nevents{nevents}")
    print("Done!")



if __name__ == "__main__":
    main()




