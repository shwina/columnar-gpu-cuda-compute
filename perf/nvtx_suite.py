"""Generously NVTX-annotated run of all 8 ADL GPU queries, in one process, for
visual inspection in the Nsight Systems GUI.

Annotation layers (all via monkeypatch -- query bodies untouched):
  * one push/pop range per query:           "Q1".."Q8"
  * stage ranges:    "stage:read" (reader.read_parquet),
                     "stage:load" (cudf_to_awkward -> cuda),
                     "stage:fill" (histogram fill)
  * one range per awkward op:               "ak.combinations", "ak.sum", ...
  * cuda.compute emits its own "CCCL:cub::..." ranges for free.

Each query is warmed once (JIT/compile excluded), then the annotated pass is
bounded by cudaProfilerStart/Stop so nsys --capture-range=cudaProfilerApi
records only the clean second pass.

Usage (under nsys, see run command printed by the harness):
  AK_BENCH_READER=cudf python nvtx_suite.py <parquet>
"""
import sys, os, io, contextlib
import awkward as ak
import cupy as cp
from cupy.cuda import nvtx, profiler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
import run_adl_queries as rq
from coffea.jitters import hist as gpu_hist

PARQUET = sys.argv[1] if len(sys.argv) > 1 else "../data/pq_subset_1M.parquet"

# ---- op-level NVTX on a generous set of awkward operations -------------------
AK_OPS = ["combinations", "argcombinations", "sum", "any", "all", "flatten",
          "argmin", "argmax", "min", "max", "num", "mask", "fill_none",
          "concatenate", "local_index", "singletons", "firsts", "zip",
          "with_name", "to_backend", "where", "cartesian", "to_cupy",
          "from_cupy", "ravel", "drop_none", "sort", "argsort", "broadcast_arrays",
          "mask", "is_none", "values_astype"]
def _wrap(mod, name, label):
    orig = getattr(mod, name, None)
    if orig is None or getattr(orig, "_nvtx_wrapped", False):
        return
    def w(*a, **k):
        nvtx.RangePush(label)
        try:
            return orig(*a, **k)
        finally:
            nvtx.RangePop()
    w._nvtx_wrapped = True
    setattr(mod, name, w)

for nm in AK_OPS:
    _wrap(ak, nm, f"ak.{nm}")

# stage markers: reader read + load(to-cuda) + fill
_wrap(rq.cudf, "read_parquet", "stage:read")
# cudf_to_awkward is imported into rq's namespace
_wrap(rq, "cudf_to_awkward", "stage:load")
# fill paths
_wrap(rq, "_cc_fill", "stage:fill")
_orig_fill = gpu_hist.Hist.fill
def _fill_w(self, *a, **k):
    nvtx.RangePush("stage:fill")
    try:
        return _orig_fill(self, *a, **k)
    finally:
        nvtx.RangePop()
gpu_hist.Hist.fill = _fill_w

# ---- run ---------------------------------------------------------------------
QUERIES = list(range(1, 9))
def run_query(q):
    fn = getattr(rq, f"query{q}_gpu")
    return fn(PARQUET)

# warm-up pass (JIT / compile / autotune) -- NOT profiled
with contextlib.redirect_stdout(io.StringIO()):
    for q in QUERIES:
        run_query(q)
cp.cuda.Device(0).synchronize()

# annotated pass -- profiled
profiler.start()
nvtx.RangePush("ADL_suite")
with contextlib.redirect_stdout(io.StringIO()):
    for q in QUERIES:
        nvtx.RangePush(f"Q{q}")
        run_query(q)
        cp.cuda.Device(0).synchronize()
        nvtx.RangePop()
nvtx.RangePop()
cp.cuda.Device(0).synchronize()
profiler.stop()
print("done: annotated pass over Q1-Q8")
