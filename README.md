# columnar_gpu_bench

Benchmarking and performance analysis of the Awkward Array **awkward3 / cuda.compute** GPU
backend on the ADL `q1`–`q8` analysis queries, vs the released **awkward 2.8.11** RawKernel
backend. GPU: NVIDIA RTX 6000 Ada, CUDA 13.3.

## Layout
- `columnar_gpu/` — the ADL benchmark (queries, GPU-direct cudf reader, timing harness).
  `run_adl_queries.py` (queries + `histogram_even` fill), `bench_all.sh`/`bench_driver.py`
  (warm, crash-isolated timing), `gpu_read_cudf.py` (reader).
- `perf/` — profiling harness, microbenchmarks, cuda.compute rewrites, and the report.
  Start with `perf/reports/PERF_REPORT.md` (self-contained); `perf/reports/WORKLOG.md` is the
  chronological investigation log.
- `RESULTS.md` — headline per-query result tables.
- `patches/awkward3-combinations-cache-fix.patch` — the one-line fix to GPU `ak.combinations`
  (coerce captured `length` to `np.int64`) that removes the per-call JIT rebuild; apply to an
  `awkward3` checkout. See PERF_REPORT §5.1 and NVIDIA/cccl#9626.
- `cudf_inject_overrides.txt` — uv overrides to force cudf 26.6 alongside cuda.compute.

## Not tracked (see .gitignore)
`.venv-awkward3/`, `.venv-baseline/` (uv venvs), `data/` (parquet subsets), `awkward_src/`
(built-from-source awkward3 checkout), and generated artifacts (nsys/sqlite/logs/plots).

## Reproduce
See `perf/reports/PERF_REPORT.md` §3 (environment) and §8 (commands).
