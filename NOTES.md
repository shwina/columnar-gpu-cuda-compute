# ADL benchmarks on GPU — Awkward cuda.compute vs RawKernel: notes & reproduction

Compare end-to-end ADL physics analysis queries (q1–q8 from `kmohrman/columnar_gpu`)
on two Awkward GPU backends:
- **awkward3** — the `awkward3` dev branch, GPU ops dispatch through **cuda.compute** (CCCL).
- **baseline** — released **awkward 2.8.11**, the last release *before* cuda.compute
  (2.9.x already ships it); GPU ops use hand-written **CuPy RawKernels**.

Everything lives under `/home/coder/columnar_gpu_bench/`. Results tables: `RESULTS.md`.

---

## 1. Findings (TL;DR)

- **Robustness:** awkward3/cuda.compute runs the full suite q1–q8. Baseline (2.8.11)
  **fails q3 and q8 at 1M** and **crashes q8 at 100k** — genuine old-backend bugs:
  - q3 @1M: `ValueError: Negative dimensions are not allowed` (didn't appear at 100k → scale-dependent).
  - q8 @100k: `CUDA_ERROR_ILLEGAL_ADDRESS` in `ak.argmin(axis=-1, keepdims=True)` on
    option-typed jagged data with empty sublists; @1M surfaces as a None→MaskedArray error.
- **Performance at scale (1M, warm compute, RTX 6000 Ada):**
  - q5 (di-muon invariant mass): **8.7×** faster (2.29 s vs 19.95 s).
  - q6 (trijet): **5.6×** faster (3.73 s vs 20.76 s).
  - q4, q7: comparable (~0.8–0.9×). q1/q2: trivial compute.
- **At 100k** everything is comparable/slightly slower on cuda.compute — the GPU is
  under-utilized so per-op launch overhead dominates; the win shows up at scale.
- **cuda.compute *added* GPU `sort`/`argsort`** (absent in 2.8.11), though the ADL
  queries don't use them.

## 2. Methodology

- Hardware: **NVIDIA RTX 6000 Ada (49 GB)** = `nvidia-smi` index 1. CUDA 13.3.
  (A 4 GB T400 is index 0 — do NOT use it; see gotcha #5 below.)
- Two **uv** virtualenvs differing only in Awkward (§4). Same read path, same queries.
- **cudf removed** (see issue #1): both envs read parquet with pyarrow via
  `cpu_read_shim.py` (`ak.Array(pa_col)` → `ak.to_backend("cuda")`), mirroring the
  benchmark's own CPU conversion. ⇒ read/load timings are NOT comparable to cudf's
  GPU-direct read; **only the `compute` stage is the apples-to-apples backend metric.**
- Timing: each query is run in its **own subprocess** (`bench_driver.py`) with a
  **warm-up call then a timed call**, so cuda.compute **JIT is excluded** and a crash in
  one query (e.g. baseline q8) doesn't poison the others. `bench_all.sh` loops q1–q8 ×
  both envs; `format_results.py` prints the markdown table. Metric = warm `compute`
  stage (`dt_lst[2]`).
- Single run per cell, one GPU — treat as indicative, not publication error-barred.

## 3. Issues found & fixes (all real, worth reporting upstream)

1. **cudf ⊥ cuda.compute.** cudf 25.12 pins `numba-cuda 0.19.x` → `cuda-core 0.3.x`
   (`cuda.core.experimental`); cuda.compute needs `cuda-core 1.0.x` (`cuda.core.Device`).
   No common version. → dropped cudf; read via `cpu_read_shim.py`.
2. **Latest released awkward already has cuda.compute** (2.9.x). Baseline must be
   **2.8.11** (released 2025-12-15, just before commit `03905ba2`/#3750 on 2026-01-06).
3. **Baseline fails q3 (1M) and q8** — old RawKernel backend bugs (above).
4. **`run_adl_queries.py` calls `main()` unguarded** at module level → importing it runs
   everything. Added `if __name__ == "__main__":` guard. Also added env var
   `AK_BENCH_PARQUET` to choose the input file.
5. **GPU-selection trap.** CUDA default order = FASTEST_FIRST, so `CUDA_VISIBLE_DEVICES=1`
   selected the **4 GB T400** → phantom q6 OOM at ~3.6 GB. Fix: also set
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` (now in `bench_all.sh`). Verify the device:
   `python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"`.

## 4. Reproduce — environment setup (uv)

Prereqs present here: CUDA 13.3 toolkit, gcc/g++ 13, cmake, ninja; a local CCCL Python
checkout at `/home/coder/cccl/python/cuda_cccl` (provides `cuda.compute`).

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh           # -> ~/.local/bin/uv
export PATH="$HOME/.local/bin:$PATH"

cd /home/coder/columnar_gpu_bench
git clone https://github.com/kmohrman/columnar_gpu        # the benchmark (already patched here)

# Datasets (subsets) into data/
mkdir -p data && cd data
BASE=http://uaf-10.t2.ucsd.edu/~kmohrman/large_files_no_backup/for_ak_gpu/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets
for f in pq_subset_100k.parquet pq_subset_1M.parquet pq_subset_10M.parquet; do wget -q "$BASE/$f"; done
cd ..

# Shared GPU/sci stack pin (works with cuda.compute; numba-cuda 0.30.2 needs cuda-core 1.0.1)
COMMON='numba==0.65.1 numba-cuda[cu13]==0.30.2 cuda-core==1.0.1 cuda-bindings==13.3.1 cuda-pathfinder==1.5.5
        cupy-cuda13x pyarrow pandas matplotlib hist uproot vector pytz
        coffea @ git+https://github.com/scikit-hep/coffea.git@jitters'
```

### Env A — awkward3 (cuda.compute)
```bash
uv venv --python 3.12 .venv-awkward3
export VIRTUAL_ENV=$PWD/.venv-awkward3
uv pip install $COMMON
# build awkward3 from source
git clone --depth 1 -b awkward3 https://github.com/scikit-hep/awkward awkward_src
cd awkward_src && git submodule update --init --recursive            # rapidjson submodule
$VIRTUAL_ENV/bin/python dev/copy-cpp-headers.py                      # generate build artifacts
$VIRTUAL_ENV/bin/python dev/generate-kernel-signatures.py
cd ..
uv pip install -e ./awkward_src/awkward-cpp                          # compiles C++
uv pip install -e ./awkward_src                                      # pure-python awkward (overrides coffea's)
uv pip install -e /home/coder/cccl/python/cuda_cccl                  # cuda.compute
# sanity: backend dispatches through cuda.compute
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $VIRTUAL_ENV/bin/python -c \
 "import awkward as ak; g=ak.to_backend(ak.Array([[3.,1.,2.],[],[5.,4.]]),'cuda'); print(ak.sort(g,axis=1).to_list())"
```

### Env B — baseline (released 2.8.11, RawKernel)
```bash
uv venv --python 3.12 .venv-baseline
export VIRTUAL_ENV=$PWD/.venv-baseline
uv pip install $COMMON
uv pip install awkward==2.8.11      # released, pre-cuda.compute (pulls awkward-cpp 51)
unset VIRTUAL_ENV
```

## 5. Reproduce — running

The benchmark in `columnar_gpu/` is already patched here: `import cpu_read_shim as cudf`,
`__main__` guard, `AK_BENCH_PARQUET` env var. To re-apply from a fresh clone, see §3.4 and
`cpu_read_shim.py`.

```bash
cd /home/coder/columnar_gpu_bench/columnar_gpu

# (a) Full native script: all queries, CPU+GPU, CPU/GPU validation, plots in plots/
PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  AK_BENCH_PARQUET=../data/pq_subset_100k.parquet \
  ../.venv-awkward3/bin/python run_adl_queries.py

# (b) Warm per-query comparison table (both envs, crash-isolated). bench_all.sh sets
#     PYTHONPATH + CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1 internally.
bash bench_all.sh ../data/pq_subset_1M.parquet logs/cmp_1M.txt
../.venv-awkward3/bin/python format_results.py logs/cmp_1M.txt

# 10M (not yet run): swap the parquet path above for ../data/pq_subset_10M.parquet
```

Key files: `run_adl_queries.py` (queries + patched main), `cpu_read_shim.py` (pyarrow read
path), `bench_driver.py` (one warm query, JSON out), `bench_all.sh` (loop+envs),
`format_results.py` (table), `logs/cmp_*.txt` (raw RESULT_JSON), `../RESULTS.md` (tables).
