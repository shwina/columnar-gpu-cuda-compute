"""Profile a single ADL GPU query in isolation with NVTX ranges.

Usage: python prof_query.py <qnum 1-8> <parquet_path> [nwarm]

- Warms up the query `nwarm` times (default 2) OUTSIDE the profiler range so
  cuda.compute JIT / CuPy RawKernel compilation is excluded.
- Wraps a curated set of `ak.*` operations (and the vector behavior add/mass)
  with NVTX push/pop so the nsys timeline attributes GPU kernels to the awkward
  operation that launched them.
- Bounds the single timed iteration with cudaProfilerStart/Stop so nsys
  (run with --capture-range=cudaProfilerApi) captures ONLY that iteration.
- Prints the dt_lst (read, load, comp, fill, total) as RESULT_JSON.

Set AK_NVTX=0 to disable op-level NVTX wrapping (raw timeline only).
"""
import sys, os, json, io, contextlib
import awkward as ak
import cupy as cp
from cupy.cuda import nvtx, profiler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
import run_adl_queries as rq

# ---- NVTX op wrapping -------------------------------------------------------
_WRAP = os.environ.get("AK_NVTX", "1") != "0"

def _wrap(modname, fnname):
    mod = ak if modname == "ak" else getattr(ak, modname)
    orig = getattr(mod, fnname)
    def wrapper(*a, **k):
        nvtx.RangePush(f"ak.{fnname}")
        try:
            return orig(*a, **k)
        finally:
            nvtx.RangePop()
    setattr(mod, fnname, wrapper)

if _WRAP:
    for name in ["combinations", "argcombinations", "sum", "any", "flatten",
                 "argmin", "argmax", "num", "mask", "fill_none", "concatenate",
                 "local_index", "singletons", "firsts", "zip", "with_name",
                 "to_backend", "min", "max", "where"]:
        try:
            _wrap("ak", name)
        except Exception as e:
            print("warn: could not wrap", name, e, file=sys.stderr)

# ---- run --------------------------------------------------------------------
qnum = int(sys.argv[1]); filepath = sys.argv[2]
nwarm = int(sys.argv[3]) if len(sys.argv) > 3 else 2
if qnum == 6 and os.environ.get("AK_Q6_CHUNKED"):
    fn = rq.query6_gpu_chunked
else:
    fn = getattr(rq, f"query{qnum}_gpu")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for _ in range(nwarm):
        fn(filepath)                       # warm-up: absorbs JIT/compile (not profiled)
    cp.cuda.Device(0).synchronize()
    profiler.start()
    nvtx.RangePush(f"q{qnum}_timed")
    out = fn(filepath)                      # the one timed+profiled iteration
    nvtx.RangePop()
    cp.cuda.Device(0).synchronize()
    profiler.stop()

t = out[-1]
res = {"q": qnum, "read": t[0], "load": t[1], "comp": t[2], "fill": t[3], "total": t[4]}
print("RESULT_JSON " + json.dumps(res))
