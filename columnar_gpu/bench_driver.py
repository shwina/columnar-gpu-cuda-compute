"""Run a single ADL query (gpu) in isolation, warm, and emit timing as JSON.

Usage: python bench_driver.py <qnum 1-8> <parquet_path> <label>

Runs the query once to warm caches (cuda.compute JIT / CuPy RawKernel compile),
then once timed, and prints a single RESULT_JSON line. Isolating each query in
its own process means a crash (e.g. the baseline's Q8 illegal memory access)
doesn't take down the other queries' measurements.
"""
import sys, json, io, contextlib
import run_adl_queries as rq

import os
qnum = int(sys.argv[1]); filepath = sys.argv[2]; label = sys.argv[3]
if qnum == 6 and os.environ.get("AK_Q6_CHUNKED"):
    fn = rq.query6_gpu_chunked
else:
    fn = getattr(rq, f"query{qnum}_gpu")
res = {"q": qnum, "label": label, "ok": False}
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # silence the query's own chatter
        fn(filepath)                         # warm-up (absorbs JIT/compile)
        out = fn(filepath)                   # timed
    t = out[-1]                              # dt_lst = [read, load, comp, fill, total]
    res.update(ok=True, read=t[0], load=t[1], comp=t[2], fill=t[3], total=t[4])
except Exception as e:
    res.update(error=f"{type(e).__name__}: {str(e)[:160]}")
print("RESULT_JSON " + json.dumps(res))
