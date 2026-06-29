"""Prototype + measure cuda.compute fusion for the 'lackluster' queries.

Three experiments, each: reference (current cupy/awkward path) vs fused
(cuda.compute iterator/algorithm), with correctness check, warm timing, and
NVTX ranges so nsys can count kernels per variant.

1. Q4 cut+reduce: ak.sum(Jet_pt>40, axis=1)  vs  segmented_reduce over a
   TransformIterator (the >40 fuses into the reduction kernel).
2. Histogram fill: coffea gpu_hist.fill (bincount chain)  vs  histogram_even.
3. Q7 metric: deltaR^2 via cupy ufunc chain  vs  one unary_transform over a
   ZipIterator of (eta_j,eta_l,phi_j,phi_l).

Usage: python fuse_proto.py <parquet> [--profile]
  --profile bounds a single timed pass with cudaProfilerStart/Stop for nsys.
"""
import sys, os, time, math, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
import numpy as np
import cupy as cp
import awkward as ak
from cupy.cuda import nvtx, profiler
import pyarrow.parquet as pq
from coffea.jitters import hist as gpu_hist

import cuda.compute as cc
from cuda.compute import segmented_reduce, OpKind, unary_transform
from cuda.compute.iterators import TransformIterator, ZipIterator

PARQUET = sys.argv[1] if len(sys.argv) > 1 else "../data/pq_subset_1M.parquet"
PROFILE = "--profile" in sys.argv
PI = np.float32(math.pi)

def to_cuda(col):
    return ak.to_backend(ak.Array(col), "cuda")

def timeit(fn, n=7):
    ts = []
    for _ in range(n):
        cp.cuda.Device(0).synchronize(); t0 = time.perf_counter()
        fn(); cp.cuda.Device(0).synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts[1:]) * 1e3  # ms, drop first

# ---------------------------------------------------------------- load
tbl = pq.read_table(PARQUET, columns=["Jet_pt", "Jet_eta", "Jet_phi", "MET_pt"])
Jet_pt = to_cuda(tbl["Jet_pt"]); MET_pt = to_cuda(tbl["MET_pt"])
nev = len(Jet_pt)
# flat content + offsets for the fused segmented path
counts_all = cp.asarray(ak.to_cupy(ak.num(Jet_pt, axis=1)).astype(cp.int64))
offsets = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(counts_all)]).astype(cp.int64)
content = cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_pt)).astype(cp.float32))
print(f"events={nev}  total jets={content.size}")

# =============================================================== Q4
def q4_ref():
    return ak.to_cupy(ak.sum(Jet_pt > 40, axis=1)).astype(cp.int64)

def gt40(x):
    return np.int32(1) if x > np.float32(40.0) else np.int32(0)
def q4_fused():
    out = cp.empty(nev, dtype=cp.int32)
    it = TransformIterator(content, gt40)
    segmented_reduce(d_in=it, d_out=out, num_segments=nev,
                     start_offsets_in=offsets[:-1], end_offsets_in=offsets[1:],
                     op=OpKind.PLUS, h_init=np.zeros(1, dtype=np.int32),
                     max_segment_size=int(counts_all.max()))
    return out

# =============================================================== Histogram fill
fillarr = ak.flatten(Jet_pt)                 # Q2's fill input
fill_cp = cp.ascontiguousarray(ak.to_cupy(fillarr).astype(cp.float32))
NB, LO, HI = 100, 0.0, 200.0
def fill_ref():
    h = gpu_hist.Hist("Counts", gpu_hist.Bin("ptj", "x", NB, LO, HI))
    h.fill(ptj=fillarr)
    return h
def fill_fused():
    counts = cp.zeros(NB, dtype=cp.int32)
    cc.histogram_even(d_samples=fill_cp, d_histogram=counts, num_output_levels=NB + 1,
                      lower_level=np.float32(LO), upper_level=np.float32(HI),
                      num_samples=int(fill_cp.size))
    return counts

# =============================================================== Q7 metric (deltaR^2)
rng = cp.random.default_rng(0)
NP_ = content.size  # ~ representative pair count
eta_j = rng.standard_normal(NP_, dtype=cp.float32) * 2
eta_l = rng.standard_normal(NP_, dtype=cp.float32) * 2
phi_j = (rng.random(NP_, dtype=cp.float32) * 2 - 1) * PI
phi_l = (rng.random(NP_, dtype=cp.float32) * 2 - 1) * PI
def dr2_ref():
    deta = eta_j - eta_l
    dphi = phi_j - phi_l
    dphi = (dphi + PI) % (np.float32(2) * PI) - PI
    return deta * deta + dphi * dphi
def dr2_op(t):
    deta = t[0] - t[1]
    dphi = t[2] - t[3]
    dphi = (dphi + PI) % (np.float32(2) * PI) - PI
    return deta * deta + dphi * dphi
def dr2_fused():
    out = cp.empty(NP_, dtype=cp.float32)
    z = ZipIterator(eta_j, eta_l, phi_j, phi_l)
    unary_transform(d_in=z, d_out=out, op=dr2_op, num_items=NP_)
    return out

# ---------------------------------------------------------------- correctness
def check(name, a, b, tol=1e-4):
    a = cp.asarray(a); b = cp.asarray(b)
    md = float(cp.abs(a.astype(cp.float64) - b.astype(cp.float64)).max()) if a.size else 0.0
    print(f"  [{name}] match={'OK' if md<=tol else 'DIFF'}  max|diff|={md:.3e}  (n={a.size})")

# warm everything (JIT)
for f in (q4_ref, q4_fused, fill_ref, fill_fused, dr2_ref, dr2_fused):
    f(); f()
cp.cuda.Device(0).synchronize()

print("\n=== correctness ===")
check("Q4 cut+reduce", q4_ref(), q4_fused())
hr = fill_ref()._sumw[()].get()[1:NB+1].astype(np.int64); hf = fill_fused().get().astype(np.int64)
print(f"  [hist fill] match={'OK' if np.array_equal(hr,hf) else 'DIFF'}  (in-range sum {hr.sum()} vs {hf.sum()})")
check("Q7 deltaR^2", dr2_ref(), dr2_fused())

# =============================================================== Q4 end-to-end (compute+fill)
def q4_full_ref():
    has2 = ak.sum(Jet_pt > 40, axis=1) >= 2
    fa = MET_pt[has2]
    h = gpu_hist.Hist("Counts", gpu_hist.Bin("met", "x", NB, LO, HI)); h.fill(met=fa)
    return h._sumw[()].get()[1:NB+1].astype(np.int64)
def q4_full_fused():
    out = cp.empty(nev, dtype=cp.int32)
    segmented_reduce(d_in=TransformIterator(content, gt40), d_out=out, num_segments=nev,
                     start_offsets_in=offsets[:-1], end_offsets_in=offsets[1:],
                     op=OpKind.PLUS, h_init=np.zeros(1, dtype=np.int32),
                     max_segment_size=int(counts_all.max()))
    met_cp = ak.to_cupy(MET_pt).astype(cp.float32)[out >= 2]
    counts = cp.zeros(NB, dtype=cp.int32)
    cc.histogram_even(d_samples=cp.ascontiguousarray(met_cp), d_histogram=counts,
                      num_output_levels=NB + 1, lower_level=np.float32(LO),
                      upper_level=np.float32(HI), num_samples=int(met_cp.size))
    return counts.get().astype(np.int64)
for f in (q4_full_ref, q4_full_fused):
    f(); f()
check("Q4 end-to-end", q4_full_ref(), q4_full_fused())

if PROFILE:
    profiler.start()
    for name, f in [("Q4FULL_ref", q4_full_ref), ("Q4FULL_fused", q4_full_fused),
                    ("Q4_ref", q4_ref), ("Q4_fused", q4_fused),
                    ("FILL_ref", fill_ref), ("FILL_fused", fill_fused),
                    ("DR2_ref", dr2_ref), ("DR2_fused", dr2_fused)]:
        nvtx.RangePush(name); f(); cp.cuda.Device(0).synchronize(); nvtx.RangePop()
    profiler.stop()
else:
    print("\n=== warm timing (median ms) ===")
    print(f"  Q4 full ref {timeit(q4_full_ref):7.3f}  fused {timeit(q4_full_fused):7.3f}  (compute+fill)")
    print(f"  Q4   ref {timeit(q4_ref):7.3f}  fused {timeit(q4_fused):7.3f}")
    print(f"  FILL ref {timeit(fill_ref):7.3f}  fused {timeit(fill_fused):7.3f}")
    print(f"  DR2  ref {timeit(dr2_ref):7.3f}  fused {timeit(dr2_fused):7.3f}")
