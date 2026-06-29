"""Compare query1_gpu (coffea bincount) vs query1b_gpu (cuda.compute histogram_even).

Confirms the two produce identical in-range bin counts, and reports the fill-stage
timing for each (warm). Usage: python compare_q1.py [parquet_path]
"""
import sys
import numpy as np
import run_adl_queries as rq

fp = sys.argv[1] if len(sys.argv) > 1 else "/home/coder/columnar_gpu_bench/data/pq_subset_100k.parquet"

# coffea reference -> real (in-range) bins are indices 1..100 of the dense array
h_a, _, t_a = rq.query1_gpu(fp)        # warm + (this call) measured below
h_a, _, t_a = rq.query1_gpu(fp)
ref = h_a._sumw[()].get()[1:101].astype(np.int64)

# cuda.compute variant
c_b, _, t_b = rq.query1b_gpu(fp)       # warm
c_b, _, t_b = rq.query1b_gpu(fp)
got = c_b.get().astype(np.int64)

match = np.array_equal(ref, got)
ndiff = int((ref != got).sum())
print("\n================ Q1 vs Q1b ================")
print(f"file              : {fp}")
print(f"coffea in-range sum: {ref.sum()}")
print(f"cuda.compute sum   : {got.sum()}")
print(f"EXACT BIN MATCH    : {match}  (differing bins: {ndiff})")
print(f"fill time  q1  (coffea bincount)    : {t_a[3]*1e3:.3f} ms")
print(f"fill time  q1b (histogram_even)     : {t_b[3]*1e3:.3f} ms")
print(f"total time q1 / q1b                 : {t_a[4]*1e3:.3f} / {t_b[4]*1e3:.3f} ms")
print("===========================================")
sys.exit(0 if match else 1)
