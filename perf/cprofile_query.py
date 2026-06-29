"""cProfile the compute of a single GPU query (warm), to find Python hotspots.
Usage: python cprofile_query.py <qnum> <parquet> [nwarm]
The GPU work is ~0.1% of wall for the heavy queries, so cumulative Python time
is the right lens. Prints top functions by cumulative and by tottime.
"""
import sys, os, io, contextlib, cProfile, pstats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
import run_adl_queries as rq
import cupy as cp

qnum = int(sys.argv[1]); filepath = sys.argv[2]
nwarm = int(sys.argv[3]) if len(sys.argv) > 3 else 2
fn = getattr(rq, f"query{qnum}_gpu")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for _ in range(nwarm):
        fn(filepath)
    cp.cuda.Device(0).synchronize()
    pr = cProfile.Profile()
    pr.enable()
    fn(filepath)
    cp.cuda.Device(0).synchronize()
    pr.disable()

st = pstats.Stats(pr)
print(f"\n===== Q{qnum} top 25 by CUMULATIVE time =====")
st.sort_stats("cumulative").print_stats(25)
print(f"\n===== Q{qnum} top 25 by TOTTIME (self) =====")
st.sort_stats("tottime").print_stats(25)
