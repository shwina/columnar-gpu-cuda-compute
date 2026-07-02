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

Prereqs: NVIDIA GPU + CUDA 13.x, `uv`, and the three parquet subsets in `data/`
(`pq_subset_{100k,1M,10M}.parquet`). All commands run from the repo root unless noted.
Pin the GPU explicitly — CUDA's default `FASTEST_FIRST` order can select the wrong device:
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # 1 == RTX 6000 Ada here; check yours
```

### 1. Build the two environments
Full step-by-step (uv `COMMON` package set + editable `awkward3` / `awkward-cpp` / `cuda-cccl`
builds) is in **`NOTES.md`** ("Env A — awkward3" and "Env B — baseline"). In short:
```bash
# .venv-awkward3 : editable awkward3 (cuda.compute) + local cuda-cccl
# .venv-baseline : released awkward==2.8.11 (RawKernel)
```

### 2. Add the combinations JIT fix (awkward3 only)
The GPU `ak.combinations` per-call JIT-rebuild fix is required for the Q5/Q6/Q8 numbers:
```bash
git -C awkward_src apply ../patches/awkward3-combinations-cache-fix.patch   # coerce length -> np.int64
```

### 3. Force-install cudf 26.6 for GPU-direct reads (both envs)
cudf's `numba-cuda<0.29` pin conflicts with cuda.compute's `cuda-core 1.0` *in metadata only*
(libcudf is C++ and never touches numba-cuda at read time), so override the pin:
```bash
for v in .venv-awkward3 .venv-baseline; do
  VIRTUAL_ENV=$PWD/$v uv pip install --override cudf_inject_overrides.txt \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
    "cudf-cu13==26.6.*"
done
```
This adds only cudf's C++ libs; the cuda/awkward stack is preserved. Reads are selected by
`AK_BENCH_READER` (`cudf`, the default in `bench_all.sh`, or `shim` for the pyarrow CPU path).

### 4. Run the full q1–q8 suite (both envs, warm, crash-isolated)
`bench_all.sh <parquet> <out.txt>` runs each query in its own process (warm-up + timed) in
both envs and writes `RESULT_JSON` lines. Do all three scales:
```bash
cd columnar_gpu
for s in 100k 1M 10M; do
  bash bench_all.sh ../data/pq_subset_$s.parquet logs/cmp_$s.txt      # 10M baseline q5/q6 ~200s each
done
```

### 5. Build the result tables
```bash
for s in 100k 1M 10M; do ../.venv-awkward3/bin/python format_results.py logs/cmp_$s.txt; done
```
Prints the compute-stage speedup table and the full per-stage (read/load/comp/fill/total)
breakdown per scale — these are what `RESULTS.md` contains.

### 6. Q1 histogram: coffea bincount vs cuda.compute `histogram_even`
```bash
cd columnar_gpu
for s in 100k 1M 10M; do
  AK_BENCH_READER=cudf PYTHONPATH=$PWD ../.venv-awkward3/bin/python \
    compare_q1.py ../data/pq_subset_$s.parquet
done
```
Confirms bit-identical bins and reports the fill-stage speedup.

### 7. (Optional) Profiling deep-dive
See `perf/reports/PERF_REPORT.md` "Reproduce" for the nsys sweep, cProfile, and the
cuda.compute cache microbenchmark.
