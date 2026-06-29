"""Verify the cuda.compute histogram_even fill produces bin-identical results to
coffea's fill, for every GPU query. Toggles AK_BENCH_HIST within one process."""
import os, sys, io, contextlib
sys.path.insert(0, "/home/coder/columnar_gpu_bench/columnar_gpu")
import numpy as np
import run_adl_queries as rq

fp = sys.argv[1] if len(sys.argv) > 1 else "/home/coder/columnar_gpu_bench/data/pq_subset_100k.parquet"

def dense_inrange(h):
    d = h._sumw[()].get()
    return d[1:len(d)-2].astype(np.int64)   # drop under/over/nan flow

def run(qnum, cc):
    os.environ["AK_BENCH_HIST"] = "cc" if cc else "off"
    fn = getattr(rq, f"query{qnum}_gpu")
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn(fp)
    return out

for q in [1,2,3,4,5,6,7,8]:
    try:
        a = run(q, False); b = run(q, True)
        if q == 6:
            ha1, ha2 = a[0], a[1]; hb1, hb2 = b[0], b[1]
            ok = np.array_equal(dense_inrange(ha1), dense_inrange(hb1)) and \
                 np.array_equal(dense_inrange(ha2), dense_inrange(hb2))
            extra = f"(pt3j sum {dense_inrange(ha1).sum()} vs {dense_inrange(hb1).sum()}; btag sum {dense_inrange(ha2).sum()} vs {dense_inrange(hb2).sum()})"
        else:
            ha, hb = a[0], b[0]
            ra, rb = dense_inrange(ha), dense_inrange(hb)
            ok = np.array_equal(ra, rb)
            extra = f"(sum {ra.sum()} vs {rb.sum()})"
        print(f"Q{q}: {'OK ' if ok else 'DIFF'}  {extra}")
    except Exception as e:
        print(f"Q{q}: ERROR {type(e).__name__}: {str(e)[:120]}")
