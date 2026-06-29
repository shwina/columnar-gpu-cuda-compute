"""Q3 the cuda.compute way: flatten(Jet_pt[|Jet_eta|<1]) is just stream
compaction of the flat pt content by the flat predicate -- one DeviceSelect.

Reference (awkward jagged masked-getitem + flatten) vs fused (cuda.compute
select over a CountingIterator with the predicate reading eta, then gather pt).
Correctness + warm timing + (with --profile) kernel count via nsys NVTX.
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
import numpy as np, cupy as cp, awkward as ak
import pyarrow.parquet as pq
from cupy.cuda import nvtx, profiler
from cuda.compute import select
from cuda.compute.iterators import CountingIterator

PARQUET = sys.argv[1] if len(sys.argv) > 1 else "../data/pq_subset_1M.parquet"
PROFILE = "--profile" in sys.argv

def to_cuda(c): return ak.to_backend(ak.Array(c), "cuda")
tbl = pq.read_table(PARQUET, columns=["Jet_pt", "Jet_eta"])
Jet_pt, Jet_eta = to_cuda(tbl["Jet_pt"]), to_cuda(tbl["Jet_eta"])

# flat content (flatten discards event boundaries -> structure irrelevant to result)
pt_c  = cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_pt)).astype(cp.float32))
eta_c = cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_eta)).astype(cp.float32))
N = pt_c.size

def q3_ref():
    return cp.ascontiguousarray(ak.to_cupy(ak.flatten(Jet_pt[abs(Jet_eta) < 1.0])).astype(cp.float32))

def keep(i):                      # predicate reads eta_c[i] (device array captured)
    return np.uint8(1) if abs(eta_c[i]) < np.float32(1.0) else np.uint8(0)
idx_out = cp.empty(N, dtype=cp.int64)
nsel = cp.empty(1, dtype=cp.int64)
def q3_fused():
    select(d_in=CountingIterator(np.int64(0)), d_out=idx_out,
           d_num_selected_out=nsel, cond=keep, num_items=N)
    k = int(nsel[0])              # one sync (output length)
    return pt_c[idx_out[:k]]      # gather pt at selected indices

def timeit(fn, n=7):
    ts=[]
    for _ in range(n):
        cp.cuda.Device(0).synchronize(); t0=time.perf_counter()
        fn(); cp.cuda.Device(0).synchronize(); ts.append(time.perf_counter()-t0)
    return statistics.median(ts[1:])*1e3

for f in (q3_ref, q3_fused): f(); f()
a, b = q3_ref(), q3_fused()
md = float(cp.abs(cp.sort(a)-cp.sort(b)).max()) if a.size==b.size else -1
print(f"N_content={N}  ref_kept={a.size}  fused_kept={b.size}  match={'OK' if (a.size==b.size and md==0) else 'DIFF'} (max|diff|={md})")

if PROFILE:
    profiler.start()
    for name,f in [("Q3_ref",q3_ref),("Q3_fused",q3_fused)]:
        nvtx.RangePush(name); f(); cp.cuda.Device(0).synchronize(); nvtx.RangePop()
    profiler.stop()
else:
    print(f"warm: ref {timeit(q3_ref):.3f} ms   fused {timeit(q3_fused):.3f} ms")
