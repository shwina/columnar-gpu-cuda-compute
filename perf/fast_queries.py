"""cuda.compute-native reimplementations of the host-bound ADL queries (Q3, Q7),
measured end-to-end (read + compute + fill) against the awkward3 default and the
RawKernel baseline.

Q3  flatten(Jet_pt[|Jet_eta|<1])  -> one DeviceSelect (stream compaction).
Q7  jets.nearest(leptons) HT      -> per-jet segmented_reduce(Min) of dR over the
                                     event's leptons, then per-event segmented sum.
Fill -> cuda.compute.histogram_even.

Read uses the same cudf GPU-direct path as the benchmark, so the comparison is
apples-to-apples end-to-end.
"""
import sys, os, time, math, statistics
from math import sqrt, pi as MPI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
os.environ.setdefault("AK_BENCH_READER", "cudf")
import numpy as np, cupy as cp, awkward as ak
import run_adl_queries as rq
import cuda.compute as cc
from cuda.compute import select, segmented_reduce, OpKind
from cuda.compute.iterators import CountingIterator, TransformIterator

PARQUET = sys.argv[1] if len(sys.argv) > 1 else "../data/pq_subset_1M.parquet"
PI = np.float32(math.pi)
NB, LO, HI = 100, np.float32(0.0), np.float32(200.0)

def jagged_buffers(arr):
    """offsets (int64 cupy), content (cupy) of a 1-level jagged cuda ak.Array."""
    lay = arr.layout
    off = cp.asarray(lay.offsets.data).astype(cp.int64)
    cont = cp.ascontiguousarray(cp.asarray(lay.content.data))
    return off, cont

def hist_even(values_cp, nb=NB, lo=LO, hi=HI):
    counts = cp.zeros(nb, dtype=cp.int32)
    v = cp.ascontiguousarray(values_cp.astype(cp.float32))
    cc.histogram_even(d_samples=v, d_histogram=counts, num_output_levels=nb + 1,
                      lower_level=lo, upper_level=hi, num_samples=int(v.size))
    return counts

def timeit(fn, n=6):
    ts = []
    for _ in range(n):
        cp.cuda.Device(0).synchronize(); t0 = time.perf_counter()
        fn(); cp.cuda.Device(0).synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts[1:]) * 1e3

# ============================================================ Q3
def q3_fast():
    tbl = rq.cudf.read_parquet(PARQUET, columns=["Jet_pt", "Jet_eta"])
    Jet_pt = rq.cudf_to_awkward(tbl["Jet_pt"]); Jet_eta = rq.cudf_to_awkward(tbl["Jet_eta"])
    _, pt_c = jagged_buffers(Jet_pt); _, eta_c = jagged_buffers(Jet_eta)
    pt_c = pt_c.astype(cp.float32); eta_c = eta_c.astype(cp.float32)
    N = pt_c.size
    def keep(i):
        return np.uint8(1) if abs(eta_c[i]) < np.float32(1.0) else np.uint8(0)
    idx = cp.empty(N, dtype=cp.int64); nsel = cp.empty(1, dtype=cp.int64)
    select(d_in=CountingIterator(np.int64(0)), d_out=idx,
           d_num_selected_out=nsel, cond=keep, num_items=N)
    k = int(nsel[0])
    return hist_even(pt_c[idx[:k]])

def q3_ref():   # awkward3 default path, end-to-end
    return rq.query3_gpu(PARQUET)[0]

# ============================================================ Q7
def _packed_buffers(arr):
    off, cont = jagged_buffers(ak.to_packed(arr))
    return off, cp.ascontiguousarray(cont.astype(cp.float32))

def q7_fast():
    cols = ["Jet_pt","Jet_eta","Jet_phi","Electron_eta","Electron_phi","Muon_eta","Muon_phi"]
    t = rq.cudf.read_parquet(PARQUET, columns=cols)
    g = rq.cudf_to_awkward
    Jet_pt = g(t["Jet_pt"]); Jet_eta = g(t["Jet_eta"]); Jet_phi = g(t["Jet_phi"])
    lep_eta = ak.concatenate([g(t["Electron_eta"]), g(t["Muon_eta"])], axis=1)
    lep_phi = ak.concatenate([g(t["Electron_phi"]), g(t["Muon_phi"])], axis=1)

    off_j, pt_j = jagged_buffers(Jet_pt); pt_j = pt_j.astype(cp.float32)
    _,    eta_j = jagged_buffers(Jet_eta); eta_j = eta_j.astype(cp.float32)
    _,    phi_j = jagged_buffers(Jet_phi); phi_j = phi_j.astype(cp.float32)
    off_l, eta_l = _packed_buffers(lep_eta)
    _,     phi_l = _packed_buffers(lep_phi)

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
    dphi = (dphi + PI) % (np.float32(2) * PI) - PI
    dr = cp.sqrt(deta * deta + dphi * dphi).astype(cp.float32)

    dr_min = cp.empty(J, dtype=cp.float32)
    segmented_reduce(d_in=dr, d_out=dr_min, num_segments=J,
                     start_offsets_in=pair_off[:-1], end_offsets_in=pair_off[1:],
                     op=OpKind.MINIMUM, h_init=np.array([np.inf], dtype=np.float32),
                     max_segment_size=int(n_lep.max()) if nev else 1)
    dr_min[l_per_jet == 0] = np.float32(-1.0)        # jets in lepton-less events: excluded

    contrib = cp.where((pt_j > 30) & (dr_min > 0.4), pt_j, cp.float32(0)).astype(cp.float32)
    ht = cp.empty(nev, dtype=cp.float32)
    segmented_reduce(d_in=contrib, d_out=ht, num_segments=nev,
                     start_offsets_in=off_j[:-1], end_offsets_in=off_j[1:],
                     op=OpKind.PLUS, h_init=np.array([0], dtype=np.float32),
                     max_segment_size=int(n_jet.max()) if nev else 1)
    return hist_even(ht)

def q7_ref():
    return rq.query7_gpu(PARQUET)[0]

# ============================================================ Q3 fully fused (no gather array)
def q3_iter():
    tbl = rq.cudf.read_parquet(PARQUET, columns=["Jet_pt", "Jet_eta"])
    Jet_pt = rq.cudf_to_awkward(tbl["Jet_pt"]); Jet_eta = rq.cudf_to_awkward(tbl["Jet_eta"])
    _, pt_c = jagged_buffers(Jet_pt); _, eta_c = jagged_buffers(Jet_eta)
    pt_c = pt_c.astype(cp.float32); eta_c = eta_c.astype(cp.float32)
    N = pt_c.size
    def keep(i):
        return np.uint8(1) if abs(eta_c[i]) < np.float32(1.0) else np.uint8(0)
    idx = cp.empty(N, dtype=cp.int64); nsel = cp.empty(1, dtype=cp.int64)
    select(d_in=CountingIterator(np.int64(0)), d_out=idx,
           d_num_selected_out=nsel, cond=keep, num_items=N)
    k = int(nsel[0])
    def gather_pt(i) -> np.float32:           # fuse the gather into histogram_even
        return pt_c[i]
    counts = cp.zeros(NB, dtype=cp.int32)
    cc.histogram_even(d_samples=TransformIterator(idx, gather_pt), d_histogram=counts,
                      num_output_levels=NB + 1, lower_level=LO, upper_level=HI, num_samples=k)
    return counts

# ============================================================ Q7 fully fused (no dr / contrib arrays)
def q7_iter():
    cols = ["Jet_pt","Jet_eta","Jet_phi","Electron_eta","Electron_phi","Muon_eta","Muon_phi"]
    t = rq.cudf.read_parquet(PARQUET, columns=cols); g = rq.cudf_to_awkward
    Jet_pt = g(t["Jet_pt"]); Jet_eta = g(t["Jet_eta"]); Jet_phi = g(t["Jet_phi"])
    lep_eta = ak.concatenate([g(t["Electron_eta"]), g(t["Muon_eta"])], axis=1)
    lep_phi = ak.concatenate([g(t["Electron_phi"]), g(t["Muon_phi"])], axis=1)
    off_j, pt_j = jagged_buffers(Jet_pt); pt_j = pt_j.astype(cp.float32)
    _, eta_j = jagged_buffers(Jet_eta); eta_j = eta_j.astype(cp.float32)
    _, phi_j = jagged_buffers(Jet_phi); phi_j = phi_j.astype(cp.float32)
    off_l, eta_l = _packed_buffers(lep_eta); _, phi_l = _packed_buffers(lep_phi)

    nev = off_j.size - 1; J = pt_j.size
    n_jet = cp.diff(off_j); n_lep = cp.diff(off_l)
    event_of_jet = cp.repeat(cp.arange(nev, dtype=cp.int64), n_jet)
    l_per_jet = n_lep[event_of_jet]
    pair_off = cp.zeros(J + 1, dtype=cp.int64); cp.cumsum(l_per_jet, out=pair_off[1:])
    P = int(pair_off[-1])
    jet_of_pair = cp.repeat(cp.arange(J, dtype=cp.int64), l_per_jet)
    lep_base = off_l[event_of_jet]                         # per-jet first-lepton offset
    def dr_of_pair(gp) -> np.float32:                     # fuse dR into the Min reduce
        j = jet_of_pair[gp]
        l = lep_base[j] + (gp - pair_off[j])
        de = eta_j[j] - eta_l[l]
        dp = phi_j[j] - phi_l[l]
        dp = (dp + MPI) % (2.0 * MPI) - MPI
        return sqrt(de * de + dp * dp)
    dr_min = cp.empty(J, dtype=cp.float32)
    segmented_reduce(d_in=TransformIterator(CountingIterator(np.int64(0)), dr_of_pair),
                     d_out=dr_min, num_segments=J,
                     start_offsets_in=pair_off[:-1], end_offsets_in=pair_off[1:],
                     op=OpKind.MINIMUM, h_init=np.array([np.inf], dtype=np.float32),
                     max_segment_size=int(n_lep.max()) if nev else 1)
    dr_min[l_per_jet == 0] = np.float32(-1.0)
    def contrib(j) -> np.float32:                          # fuse cut+mask into the HT Sum
        if pt_j[j] > 30.0 and dr_min[j] > 0.4:
            return pt_j[j]
        return 0.0
    ht = cp.empty(nev, dtype=cp.float32)
    segmented_reduce(d_in=TransformIterator(CountingIterator(np.int64(0)), contrib),
                     d_out=ht, num_segments=nev,
                     start_offsets_in=off_j[:-1], end_offsets_in=off_j[1:],
                     op=OpKind.PLUS, h_init=np.array([0], dtype=np.float32),
                     max_segment_size=int(n_jet.max()) if nev else 1)
    return hist_even(ht)

# ============================================================ run
if __name__ == "__main__":
    import io, contextlib
    which = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "q3"
    profile = "--profile" in sys.argv
    if profile:
        from cupy.cuda import nvtx, profiler
        for q in (q3_fast, q3_iter, q7_fast, q7_iter): q(); q()
        cp.cuda.Device(0).synchronize(); profiler.start()
        for name, f in [("Q3_array", q3_fast), ("Q3_iter", q3_iter),
                        ("Q7_array", q7_fast), ("Q7_iter", q7_iter)]:
            nvtx.RangePush(name); f(); cp.cuda.Device(0).synchronize(); nvtx.RangePop()
        profiler.stop(); print("profiled"); sys.exit(0)
    with contextlib.redirect_stdout(io.StringIO()):
        for q in (q3_fast, q3_iter, q7_fast, q7_iter): q(); q()
        h3r = q3_ref()._sumw[()].get()[1:NB+1].astype(np.int64)
        h7r = q7_ref()._sumw[()].get()[1:NB+1].astype(np.int64)
        h3a = q3_fast().get().astype(np.int64); h3i = q3_iter().get().astype(np.int64)
        h7a = q7_fast().get().astype(np.int64); h7i = q7_iter().get().astype(np.int64)
        t3r = timeit(q3_ref); t3a = timeit(q3_fast); t3i = timeit(q3_iter)
        t7r = timeit(q7_ref); t7a = timeit(q7_fast); t7i = timeit(q7_iter)
    print(f"Q3 match: array={np.array_equal(h3r,h3a)} iter={np.array_equal(h3r,h3i)}")
    print(f"Q3 e2e:  awkward3 {t3r:.2f}   array {t3a:.2f}   iter {t3i:.2f} ms")
    print(f"Q7 match: array={np.array_equal(h7r,h7a)} iter={np.array_equal(h7r,h7i)}")
    print(f"Q7 e2e:  awkward3 {t7r:.2f}   array {t7a:.2f}   iter {t7i:.2f} ms")
