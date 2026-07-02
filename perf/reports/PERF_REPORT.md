# ADL GPU Benchmark — cuda.compute Performance Analysis

What limits the Awkward Array **awkward3 / cuda.compute** GPU backend on the ADL `q1`–`q8`
analysis queries, the changes that close the gaps, and the resulting numbers. Self-contained:
Section 5 is the full recipe to reproduce these numbers from a clean awkward3 checkout.

- **Hardware:** NVIDIA RTX 6000 Ada (49 GB), CUDA 13.3, driver 580.167.08.
- **Backend under test:** `awkward3` branch (GPU ops via `cuda.compute` / CCCL), `cuda-cccl 3.5.0.dev`.
- **Baseline:** released **awkward 2.8.11** (CuPy RawKernel backend — the last release before cuda.compute).
- **Queries:** the 8 ADL benchmark queries from `kmohrman/columnar_gpu` (MET/jet histograms, jet
  selection, di-muon mass, trijet, lepton-cleaned HT, 3ℓ transverse mass).

---

## 1. Summary

The queries are **host-dispatch bound, not GPU bound.** Across the suite the GPU is only
**0.1–23 % busy**; wall time is dominated by Python op-dispatch, CUDA kernel-launch latency, and
**one device→host synchronization per structural op** (a scalar offset/length read-back). The
kernels themselves are fine — the substantive elementwise kernels hit ~80 % of DRAM bandwidth
(`ncu`). So the levers are *fewer launches, fewer syncs, fewer rebuilds* — not faster kernels.

Two consequences, and the work that addresses each:

- **Combinatoric queries (Q5/Q6/Q8)** were crippled by a **per-call JIT rebuild** in GPU
  `ak.combinations` (a cuda.compute cache-key bug). One fix removes it → **~10–36×** on compute.
- **Light queries (Q3/Q7)** do work cuda.compute's reducers don't accelerate, so the stock
  backend is ~par or slightly behind baseline. Rewriting them directly on cuda.compute
  primitives (`select`, `segmented_reduce`) makes them **faster than baseline end-to-end**.

With the combinations fix + GPU-direct reads, awkward3 goes from ~par on the trivial queries to
**hundreds-fold faster** on the combinatoric ones (Q5: ~190× end-to-end at 1M, up to ~950×
end-to-end / ~1500× on the compute stage at 10M); the two queries it had been losing are recovered.

---

## 2. Results

All times are warm (JIT excluded except where noted), single GPU, one run each. Speedup =
baseline ÷ awkward3 (>1 ⇒ cuda.compute faster). Reads use cudf GPU-direct in both environments.

### 2.1 End-to-end (read + load + compute + fill), 1M events

| Q | what it does | baseline | awkward3 | speedup |
|---|---|--:|--:|--:|
| Q1 | MET histogram | 10.7 ms | 10.7 ms | 1.00× |
| Q2 | jet-pT histogram | 21.0 ms | 20.9 ms | 1.00× |
| Q3 | jet pT, \|η\|<1 | 29.3 ms | 36.5 ms | 0.80× → **1.5×** (§5.4) |
| Q4 | MET, ≥2 jets pT>40 | 23.1 ms | 21.6 ms | 1.07× |
| Q5 | di-muon mass (comb n=2) | 19 994 ms | 103.7 ms | **193×** |
| Q6 | trijet (comb n=3) | 21 319 ms | 656.6 ms | **32×** |
| Q7 | lepton-cleaned HT (`nearest`) | 267.3 ms | 290.2 ms | 0.92× → **3.0×** (§5.4) |
| Q8 | 3ℓ SFOS Mᴛ (comb n=2) | **fails** | 298.1 ms | awkward3-only |

For the combinatoric queries the e2e speedup is *smaller* than the compute-stage speedup below,
because read+load are identical across backends (both cudf) and become a fixed floor once
compute collapses. Q3/Q7 with the §5.4 rewrites move from losing to **1.5×/3.0× vs baseline**.

### 2.2 Compute-stage speedup, all scales

| Q | 100k | 1M | 10M |
|---|--:|--:|--:|
| Q3 | 0.69× | 0.56× | 0.50×  (→ faster than baseline with the §5.4 `select` rewrite) |
| Q4 | 1.11× | 1.41× | 1.79×  (→ 4–6× with the §5.4 fused-count + `select` rewrite) |
| Q5 | 58.1× | 337.2× | **1505×** |
| Q6 | 23.6× | 35.7× | 39.0× |
| Q7 | 0.96× | 0.91× | 0.93×  (→ faster with the §5.4 `segmented_reduce` rewrite) |
| Q8 | baseline fails | baseline fails | baseline fails |

Q5 compute is **flat with scale** (34.9 → 59.2 → 131 ms) while the baseline grows super-linearly
(2.03 → 19.95 → 197.3 s), so its speedup widens to 1505× at 10M. Q4 is the one light query that
*wins* in stock form — its hot op (`sum(pt>40, axis=1)`) is a segmented reduce, which cuda.compute
does well; the rest (Q3/Q7) stay host-bound and ~par until rewritten. Q4 still improves further
(1.1–1.8× → 4–6×) once the `pt>40` cut is fused into the count and the `>=2` selection into a single
`select` (§5.4, `query4c_gpu`), removing the boolean-materialize + cupy gather around the reduce.

### 2.3 Robustness

awkward3 runs all of `q1`–`q8` at every scale. The baseline (RawKernel) **fails Q8** at all
scales (genuine `argmin`-on-empty-sublists bug: `bad_alloc`/`Negative dimensions`/illegal access).
Baseline Q3 also failed under the old pyarrow read path, but that was a read-layout artifact and
succeeds with cudf — so it is **not** counted as a backend failure here.

---

## 3. Environment & methodology

- Two uv venvs differing only in Awkward: `.venv-awkward3` (cuda.compute) and `.venv-baseline`
  (awkward 2.8.11, RawKernel). Same read path, same queries.
- **Reads:** cudf 26.6 GPU-direct in both envs (`gpu_read_cudf.py`), selectable via
  `AK_BENCH_READER=cudf|shim`. cudf decodes parquet on the GPU and the device buffers are wrapped
  as awkward content with no host round-trip, so read/load are comparable across backends.
- **Timing:** each query runs in its own process (`bench_driver.py`), warm-up call then a timed
  call; stages bounded by `cp.cuda.Device(0).synchronize()`. Device pinned with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (index 1 = the RTX 6000 Ada; index 0 is a
  4 GB T400 — do not use).
- **Profiling:** `prof_query.py` wraps `ak.*` ops in NVTX and bounds one timed iteration with
  `cudaProfilerStart/Stop`; `nsys --capture-range=cudaProfilerApi` captures only that iteration;
  `parse_nsys.py` extracts kernel count, GPU-busy ns, memcpy, and D2H-sync count from the sqlite.
  `cprofile_query.py` for Python hotspots, `micro_cccl_cache.py` for the cache repro, `ncu` for
  kernel efficiency spot-checks.

---

## 4. Diagnosis: why the queries are host-bound

`busy%` = GPU kernel time ÷ compute-stage wall (low ⇒ host-bound). awkward3 post-fix, 1M, timed
iteration (profiling harness, 2 warmups — so Q5/Q6 compute is slightly below the §2 benchmark
numbers; the kernel/sync counts are harness-independent):

| Q | compute | GPU-busy | busy% | #kernels | #launch | #D2H/sync | dominant work |
|---|--:|--:|--:|--:|--:|--:|---|
| Q1 | ~0 | 0.14 ms | — | 39 | 31 | 12 | histogram fill only |
| Q2 | ~0 | 0.39 ms | — | 39 | 31 | 28 | flatten + fill |
| Q3 | 8.2 ms | 1.83 ms | 23 % | 215 | 177 | 131 | cupy ufuncs + jagged mask-getitem |
| Q4 | 5.6 ms | 0.77 ms | 14 % | 95 | 83 | 92 | segmented sum + cupy ufuncs |
| Q5 | 66.6 ms | 3.9 ms | 6 % | 1129 | 937 | 960 | combinations + di-muon mass |
| Q6 | 372.6 ms | 303 ms | **81 %** | 800 | 686 | 881 | trijet combinations (n=3) |
| Q7 | 97.5 ms | 20.7 ms | 21 % | 1115 | 903 | 2060 | cross-join `nearest` |
| Q8 | 131.6 ms | 11.5 ms | 9 % | 2502 | 2046 | 2041 | combinations + argmin/argmax |

Three takeaways:

1. **Most queries are GPU-idle most of the time** — the bottleneck is the *number* of kernels
   and the host gaps + syncs between them, not kernel speed. The exception is post-fix **Q6**
   (~81 % busy): its n=3 trijet combinatorics are genuine GPU work, so it is now compute-bound.
2. **Light queries do the same work on both backends** (Q4 95 vs 98 kernels, Q7 1115 vs 1022),
   dominated by `cupy_*` ufuncs + structural kernels; cuda.compute's segmented reducers are off
   the critical path → ~par. The exception is Q4, whose hot op *is* a segmented reduce → it wins.
3. **The baseline's combinatoric path explodes:** Q5 baseline issues **285 336 kernels**
   (284 k `cupy_fill`) and **2 000 957 D2H syncs**; cuda.compute collapses this to ~1.1 k
   kernels. That structural collapse — not faster kernels — is the Q5/Q6 win.

---

## 5. The work — from a clean awkward3 checkout to these numbers

### 5.0 Build the environments
Build `awkward3` from source with `cuda.compute` (editable `cuda_cccl`). Force-install **cudf 26.6**
alongside it (`uv pip install --override cudf_inject_overrides.txt …`): cudf's `numba-cuda<0.29`
metadata pin is irrelevant at read time (libcudf is C++), so cudf coexists with cuda.compute and
gives GPU-direct reads in both envs. Baseline env = `awkward==2.8.11`.

### 5.1 Fix the per-call JIT rebuild in `ak.combinations`  → Q5/Q6/Q8: 10–43×
**Root cause.** GPU `ak.combinations` builds a fresh Python closure (`fill_pos`) per combination
position and passes it to cuda.compute's `unary_transform`; the closure captures `length` (the
number of lists, a **Python int**). cuda.compute keys the op via `CachableFunction`
(bytecode + closure cell contents + globals), and its `_make_hashable` sends plain Python ints to
`id(value)`. A large int is a fresh object each call → fresh cache key → the transform is
**JIT-rebuilt on every call** (`call_build` ≈ 0.7–1.0 s × *n* positions). So for the
combinatoric queries the "warm" timings were really measuring JIT, every call.

Evidence (`micro_cccl_cache.py`): a closure capturing a fresh large Python int rebuilds every
call (~700 ms each); capturing the same value as `np.int64` builds once then caches (0 ms). And
directly, `_make_hashable(int(str(10**6))) == _make_hashable(int(str(10**6)))` is `False`.

**Fix (one of two places):**
- *awkward-side* (works on any cuda.compute): in `awkward_ListArray_combinations`
  (`awkward/_connect/cuda/_compute.py`), coerce the captured scalar before the closure:
  ```python
  length = np.int64(length)   # numpy scalars are cache-keyed by value; Python ints by id()
  ```
- *cuda.compute-side* (general, fixes it for everyone): make `_make_hashable` key Python
  `int`/`float`/`bool` by value. **Filed: [NVIDIA/cccl#9626](https://github.com/NVIDIA/cccl/issues/9626).**

**Impact** (1M, compute stage, same profiling harness before/after; validated bit-for-bit against
CPU except a pre-existing 0.011 % argmin tie-break in Q6):

| Q | before | after |
|---|--:|--:|
| Q5 | 2328 ms | **66.6 ms** (~35×) |
| Q6 | 3766 ms | **372.6 ms** (~10×) |
| Q8 | 4798 ms | **131.6 ms** (~36×) |

(The §2 benchmark harness — one warm-up — reports slightly higher post-fix compute, e.g. Q6
≈ 594 ms at 1M, because cuda.compute settles after a second warm-up; §2 is the headline metric.)

### 5.2 GPU-direct reads (cudf)  → honest, comparable read/load
With the pyarrow shim, "load" (host→device copy + conversion) hid a real, scale-dependent cost
(~135 ms for one flat column at 10M). cudf GPU-direct cuts load to ~0.2 ms and makes read/load
comparable across backends, so end-to-end totals are meaningful. (It also gives the baseline a
clean int64-offset layout, which is why baseline Q3 stops failing.)

### 5.3 `histogram_even` fill fusion  → faster fill, opt-in
Coffea's histogram fill is a 10–15-kernel clip/ravel/bincount chain. `_cc_fill` (enabled with
`AK_BENCH_HIST=cc`) replaces it with a single fused CUB `histogram_even` over a
`TransformIterator`, **bit-identical** for all q1–q8 and **flat with scale** (≈0.5 ms vs coffea's
2→7.8 ms at 1M→10M; 38→2 kernels). Falls back to coffea when cuda.compute is unavailable.

### 5.4 cuda.compute-native rewrites of the host-bound losers (Q3, Q7)  → faster than baseline
The two queries awkward3 was *losing* are expressible directly in cuda.compute primitives
(prototyped in `fast_queries.py`, and wired into the benchmark as `query3c_gpu`/`query7c_gpu`
in `run_adl_queries.py` — run via `bench_driver.py 3c|7c`), which collapses their kernel/sync soup:

- **Q3** `flatten(Jet_pt[|Jet_eta|<1])` — flattening discards event structure, so it is pure
  **stream compaction** of the flat `pt` content by the flat `|η|<1` predicate: one
  `DeviceSelect` (cond reads the captured `eta`) + a gather. **193 → 3 kernels**, compute
  16.5 → ~0.1 ms.
- **Q7** `jets.nearest(leptons)` HT — `nearest` is a per-jet **`segmented_reduce(MINIMUM)`** of
  ΔR over the event's leptons (segments = per-jet lepton counts), then a per-event
  `segmented_reduce(PLUS)` for HT, then `histogram_even`. The 1115-kernel / 2060-sync cross-join
  collapses to a handful of primitives. ΔR is fused into the MIN reduce via a `ZipIterator` of
  four **`PermutationIterator`s** (gathered η/φ) + a stateless op — no `deta`/`dphi`/`dr`
  temporaries (see §6); compute ~228 → ~19 ms.

Both are bit-identical to the awkward path and now **beat baseline end-to-end** (1M): Q3
default 36.5 ms → fused **20.8 ms** vs baseline 29.3 ms (**1.4×**); Q7 default 290 ms → fused
**84.6 ms** vs baseline 267 ms (**3.2×**). Both are now read-bound (Q3 ~16/20.8 ms, Q7 read+load
~65/85 ms), so compute is no longer the bottleneck.

**Generalization.** The "structural" ops here map cleanly to cuda.compute: `flatten(x[mask])` →
`select`; structure-preserving `x[mask]` → `select` + `segmented_reduce`; `nearest`/cross-join →
`segmented_reduce` (+ `binary_search`). The remaining engineering is teaching awkward's CUDA
backend to *recognize* these shapes (e.g. route `flatten(x[bool])` to `select`) so users get the
speedup without hand-writing it.

---

## 6. Iterator fusion: reach and limit

`segmented_reduce`/`histogram_even` accept an iterator for their input, so a **stateless**
transform fuses into the kernel with zero temporaries — e.g. Q4's `pt>40` predicate via
`TransformIterator(content, x>40)` into `segmented_reduce` (**48 → 3 kernels**), and the
histogram fill (38 → 2). The *maximal* zero-temporary fusion of Q3/Q7 needs the iterator op to
**index other arrays** (gather `pt` by selected index; read η/φ by pair index) — a **stateful**
op. A stateful op works as a *direct* algorithm op (`select` cond, `unary_transform`) but
**fails inside a `TransformIterator` fed to `segmented_reduce`/`histogram_even`**
(`NotImplementedError` in type inference; `cudaErrorLaunchFailure` at runtime).
**Filed: [NVIDIA/cccl#9627](https://github.com/NVIDIA/cccl/issues/9627).**

**The gather case has a native escape hatch: `PermutationIterator`.** For the common
gather-then-compute pattern, `PermutationIterator(values, indices)` does the indexing inside the
iterator, so the transform op stays *stateless* and sidesteps #9627. This unblocks the
zero-temporary fusions: `histogram_even(PermutationIterator(pt, idx))` (Q3, gather → fill) and a
`ZipIterator` of `PermutationIterator`s + stateless op feeding `segmented_reduce` (Q7 ΔR, used in
`query7c_gpu`). Measured Q7 ΔR+reduce compute: **11.0 → 6.4 ms @1M (1.7×)**, bit-identical,
eliminating the `deta`/`dphi`/`dr` buffers. Since Q3/Q7 are now read-bound the e2e payoff is
small, but it removes DRAM traffic and is the clean general pattern.

---

## 7. Recommendations & upstream issues

**Awkward (high value, low risk)**
1. Land the combinations cache fix (§5.1) and audit every closure passed to cuda.compute in
   `_compute.py` for captured Python scalars — coerce to numpy scalars or pass as device-side
   state. `combinations_len` is already clean; `fill_pos` was the offender.
2. Cut the per-op D2H sync count (one offset/length read-back per structural op; Q7 = 2060). Keep
   next-allocation sizes on device (upper-bound allocation / device-scalar-sized iterators) to
   defer and batch syncs — the main lever for the remaining host-bound queries.
3. Recognize the §5.4 patterns (`flatten(x[mask])` → `select`, `nearest` → `segmented_reduce`) in
   the CUDA backend so the rewrites apply automatically.
4. Use **`PermutationIterator`** for the pervasive `_carry`/gather-then-reduce/scan/fill patterns.
   The stock light queries are dominated by `cupy_take`/`cupy_copy` gather kernels from `_carry`;
   feeding a `PermutationIterator` straight into the consuming algorithm fuses the gather away
   (no materialized take). This is a *stock-backend* lever — it would speed up the queries that
   still lose, not just the hand-tuned rewrites (which already use it; see §6).

**cuda.compute / CCCL**
4. [#9626](https://github.com/NVIDIA/cccl/issues/9626) — key Python `int`/`float`/`bool` by value
   in `_make_hashable` (prevents the silent per-call rebuild class of bug).
5. [#9627](https://github.com/NVIDIA/cccl/issues/9627) — support stateful ops inside a
   `TransformIterator` used as an algorithm input (unlocks zero-temporary gather/segmented fusion).

---

## 8. Reproduce & artifacts

```bash
cd perf
# per-query nsys profiling sweep (host-bound evidence, §4)
bash run_sweep.sh ak3 1M reports/sweep_ak3_1M.jsonl
bash run_sweep.sh base 1M reports/sweep_base_1M.jsonl 1 2 4 5 6 7
../.venv-awkward3/bin/python summarize.py reports/sweep_ak3_1M.jsonl reports/sweep_base_1M.jsonl
# root-cause + repros
../.venv-awkward3/bin/python cprofile_query.py 5 ../data/pq_subset_1M.parquet 2   # call_build hotspot (§5.1)
../.venv-awkward3/bin/python micro_cccl_cache.py                                  # cache-key repro (§5.1, #9626)
# fused rewrites + iterator fusion (§5.4, §6)
../.venv-awkward3/bin/python fast_queries.py ../data/pq_subset_1M.parquet         # Q3/Q7 rewrites, correctness + e2e
../.venv-awkward3/bin/python fuse_proto.py  ../data/pq_subset_1M.parquet          # Q4 transform-reduce, fill, ΔR fusion
```

Key scripts: `prof_query.py`, `parse_nsys.py`, `summarize.py`, `run_sweep.sh` (profiling);
`cprofile_query.py`, `micro_cccl_cache.py` (root cause); `fast_queries.py`, `fuse_proto.py`,
`q3_select_proto.py`, `q7_perm_bench.py` (rewrites/fusion); `patch_fills.py`, `verify_cc_fill.py` (histogram wiring);
`nvtx_suite.py` (generously NVTX-annotated suite for the Nsight Systems GUI). The
chronological investigation log is in `reports/WORKLOG.md`.
