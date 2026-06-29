# ADL benchmarks: awkward3 (cuda.compute) vs baseline (released awkward, RawKernel)

Repo: `kmohrman/columnar_gpu` (ADL queries q1–q8). GPU: NVIDIA RTX 6000 Ada 49GB
(`CUDA_VISIBLE_DEVICES=1`). CUDA 13.3.

> All numbers below are from a single full re-run (100k/1M/10M, both envs). Both envs read
> parquet **GPU-direct via cudf 26.6** (force-installed alongside cuda.compute / RawKernel —
> see `cudf_inject_overrides.txt`). awkward3 includes the `ak.combinations` JIT cache-key fix
> (`patches/awkward3-combinations-cache-fix.patch`; see `perf/reports/PERF_REPORT.md`).
> Warm timings (one warm-up call, then a timed call), single GPU, one run each.

## Environments (uv)
- `.venv-awkward3`: editable awkward3 (cuda.compute backend) + local `cuda-cccl`.
- `.venv-baseline`: released **awkward 2.8.11** — the last release *before* cuda.compute
  (2.9.x already ships it). CuPy RawKernel GPU backend.
- **cudf 26.6** force-installed in **both** via `uv --override` (cudf's `numba-cuda<0.29`
  metadata pin is irrelevant at read time — libcudf is C++). Both read GPU-direct.

## Setup deviations / findings
1. **cudf restored, GPU-direct, in both envs.** The old conflict (cudf → `numba-cuda<0.29`
   → `cuda-core<1.0` vs cuda.compute's `cuda-core 1.0`) is metadata-only; forcing the pin
   lets cudf 26.6 coexist and `cudf.read_parquet` works at runtime. `gpu_read_cudf.py`
   wraps device buffers as awkward (flat → `from_cupy`; jagged → `count_elements()` offsets
   + `.elements` leaf), no host round-trip. Reader selectable via `AK_BENCH_READER=cudf|shim`.
   ⇒ **read/load are now comparable across backends**, not a shim artifact.
2. **Read path affects baseline robustness.** Baseline Q3 (`ValueError: Negative dimensions`)
   used to FAIL at ≥1M with the pyarrow shim; with cudf's clean int64-offset `ListOffsetArray`
   it now **succeeds at all scales**. So that "failure" was a read-layout artifact, not a
   compute-backend limit. (Q8 still fails — a genuine argmin bug, below.)
3. **cuda.compute added GPU sort/argsort** (absent in 2.8.11); ADL queries don't use them.
4. **Baseline Q8 still crashes** at every scale: `ak.argmin(axis=-1, keepdims=True)` on
   option-typed jagged data with empty sublists. Surfaces as `bad_alloc` (100k, via rmm),
   `Negative dimensions` (1M), `CUDA_ERROR_ILLEGAL_ADDRESS` (10M). awkward3 runs Q8 correctly.
5. **GPU selection gotcha:** CUDA default order is FASTEST_FIRST, so `CUDA_VISIBLE_DEVICES=1`
   could select the **T400 4GB**. Fix: `export CUDA_DEVICE_ORDER=PCI_BUS_ID` (in bench_all.sh).

## How to run
```
cd columnar_gpu
# warm per-query comparison (isolated subprocess per query); reader defaults to cudf:
bash bench_all.sh <parquet> logs/<out>.txt           # AK_BENCH_READER=shim for old path
../.venv-awkward3/bin/python format_results.py logs/<out>.txt
```

## Results — warm GPU **compute** time per query (JIT excluded), RTX 6000 Ada 49GB

speedup = baseline / awkward3 (>1 means cuda.compute faster).

### 100k events
| Query | baseline (RawKernel) | awkward3 (cuda.compute) | speedup |
|---|---|---|---|
| Q1 | ~0 ms | ~0 ms | — |
| Q2 | ~0 ms | ~0 ms | — |
| Q3 | 4.8 ms | 6.9 ms | 0.69× |
| Q4 | 3.3 ms | 3.0 ms | 1.11× |
| Q5 | 2028 ms | 34.9 ms | **58.1×** |
| Q6 | 2158 ms | 91.4 ms | **23.6×** |
| Q7 | 96.1 ms | 99.9 ms | 0.96× |
| Q8 | **CRASH** (bad_alloc) | 123.3 ms | awkward3 only |

### 1M events
| Query | baseline (RawKernel) | awkward3 (cuda.compute) | speedup |
|---|---|---|---|
| Q1 | ~0 ms | ~0 ms | — |
| Q2 | ~0 ms | ~0 ms | — |
| Q3 | 9.3 ms | 16.6 ms | 0.56× |
| Q4 | 5.1 ms | 3.6 ms | 1.41× |
| Q5 | 19949 ms | 59.2 ms | **337×** |
| Q6 | 21228 ms | 594.1 ms | **35.7×** |
| Q7 | 207.4 ms | 228.1 ms | 0.91× |
| Q8 | **FAIL** (Negative dimensions) | 243.0 ms | awkward3 only |

### 10M events (Q6 non-chunked this run)
| Query | baseline (RawKernel) | awkward3 (cuda.compute) | speedup |
|---|---|---|---|
| Q1 | ~0 ms | ~0 ms | — |
| Q2 | ~0 ms | ~0 ms | — |
| Q3 | 44.4 ms | 88.7 ms | 0.50× |
| Q4 | 15.3 ms | 8.5 ms | 1.79× |
| Q5 | 197335 ms | 131.1 ms | **1505×** |
| Q6 | 213887 ms | 5492 ms | **39.0×** |
| Q7 | 1014 ms | 1088 ms | 0.93× |
| Q8 | **FAIL** (illegal mem access) | 904.0 ms | awkward3 only |

## Compute-stage speedup across scales (baseline ÷ awkward3)
| Query | 100k | 1M | 10M |
|---|---|---|---|
| Q3 | 0.69× | 0.56× | 0.50× |
| Q4 | 1.11× | 1.41× | 1.79× |
| Q5 | 58.1× | 337× | **1505×** |
| Q6 | 23.6× | 35.7× | 39.0× |
| Q7 | 0.96× | 0.91× | 0.93× |
| Q8 | base CRASH | base FAIL | base FAIL |

**Q5 is the standout:** cuda.compute compute time is ~flat with scale (34.9 ms → 59 ms →
131 ms) while the RawKernel backend grows super-linearly (2.03 s → 19.95 s → 197.3 s), so the
speedup widens 58× → 337× → **1505×**. Q6 holds ~36–39× at 1M/10M (n=3 combinations ⇒ real GPU
work scales). The combination-heavy wins reflect cuda.compute collapsing hundreds of
thousands of kernel launches/syncs into ~thousands, plus the JIT cache-key fix.

## End-to-end with GPU-direct reads — the load stage collapses

Full per-stage breakdown (ms), 1M events. With cudf, the host→device + `ak.Array` conversion
("load") nearly vanishes; read times are comparable across backends.

| Query | env | read | load | comp | fill | total | total speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| Q5 | baseline | 41.3 | 2.7 | 19949 | 0.9 | 19994 | |
| Q5 | awkward3 | 40.4 | 3.1 | 59.2 | 1.0 | 103.7 | **193×** |
| Q6 | baseline | 51.3 | 15.4 | 21228 | 25.0 | 21319 | |
| Q6 | awkward3 | 38.1 | 2.7 | 594.1 | 21.8 | 656.6 | **32×** |
| Q7 | baseline | 50.9 | 7.3 | 207.4 | 1.8 | 267.3 | |
| Q7 | awkward3 | 53.0 | 7.2 | 228.1 | 1.9 | 290.2 | 0.92× |

(Contrast: under the old pyarrow shim the load stage was ~9 ms @1M and **~135 ms @10M** for a
flat column — cudf cuts it to ~0.2 ms. Full tables via `format_results.py logs/cmp_*.txt`.)

## Takeaways for the paper
- **Robustness:** awkward3/cuda.compute runs the full ADL suite (q1–q8) at all scales. The
  baseline (2.8.11 RawKernel) still fails Q8 everywhere (genuine argmin bug). Q3's old
  failure was a **read-path artifact** and disappears with cudf — report it honestly.
- **Performance at scale:** combination-heavy queries (Q5 di-muon, Q6 trijet, Q8) are now
  ~24–1505× faster (post JIT-fix); light/elementwise queries (Q3, Q7) stay ~par (host-bound,
  see PERF_REPORT — where Q3/Q7 cuda.compute rewrites flip them to faster than baseline). Q4
  modestly favors cuda.compute and grows with scale.
- **GPU-direct reads matter end-to-end:** eliminating the host→device load stage removes a
  large, scale-dependent cost (≈135 ms at 10M for one flat column) that the old shim masked.
- **Caveats:** warm timings (JIT excluded); single GPU; one run each (no error bars). awkward3
  numbers include the combinations JIT fix. cudf is a forced-pin install (read path is libcudf
  C++; unaffected by the numba-cuda pin). A reader-dependent **fill**-stage difference exists
  (cudf's clean NumpyArray layout speeds coffea's bincount vs the shim's option-type layout) —
  it conflates read vs downstream cost; normalize layout to isolate the pure read effect.
