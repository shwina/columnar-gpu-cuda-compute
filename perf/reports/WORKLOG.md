# ADL GPU queries — performance deep-dive (awkward3 / cuda.compute)

**Question posed:** why are all queries except Q5/Q6 lackluster, and are we measuring
JIT overhead or genuinely slower kernels?

**Short answer:** Neither "slow kernels" nor (for most queries) "JIT". The GPU is
**0.1 – 23 % busy** on every query — these jagged HEP queries are **host-dispatch bound**,
not GPU bound. The wall time is dominated by Python op-dispatch, CUDA kernel-launch
latency, and **forced device→host synchronizations** (one scalar read-back per structural
op). Two distinct mechanisms explain the table:

1. **Light queries (Q3, Q4, Q7):** many tiny `cupy_*` ufunc kernels + structural kernels,
   each followed by a D2H scalar read + sync. Both backends issue *nearly identical* work,
   so cuda.compute can't win — its advantage (segmented reductions) isn't on the critical
   path. Result: ≈ parity (0.85–0.93×).
2. **Combinatoric queries (Q5, Q6, Q8):** here we *were* measuring JIT — but a pathological
   form. `ak.combinations` rebuilds (re-JIT-compiles) its cuda.compute transform **on every
   call** because of a cache-key bug. The old (RawKernel) backend is even worse: it explodes
   into hundreds of thousands of kernels + millions of syncs. cuda.compute wins ~8–12× by
   collapsing that, but still left ~90 % of its own time on the table to per-call rebuilds.

**A one-line fix** (cache-key stabilization in `ak.combinations`) removes the rebuilds:

| Query (1M, compute stage) | before | after fix | speedup |
|---|---:|---:|---:|
| Q5 (di-muon mass, combinations n=2) | 2328 ms | **57 ms** | **~41×** |
| Q6 (trijet, combinations n=3)       | 3766 ms | **367 ms** | **~10×** |
| Q8 (3ℓ SFOS, combinations n=2)      | 4798 ms | **111 ms** | **~43×** |

GPU↔CPU outputs still agree (Q5 exact; Q6 differs on 48/441 890 entries = 0.011 %, a
pre-existing argmin tie-break, unrelated to the fix).

---

## Update (2026-06-26): GPU-direct cudf reads restored + full re-measurement

Two things changed since the analysis above, and the headline tables in `RESULTS.md` were
rebuilt at 100k/1M/10M with both:

**(a) cudf is back, GPU-direct, in *both* envs.** The old "cudf ⊥ cuda.compute" block was
metadata-only: cudf's `numba-cuda<0.29` → `cuda-core<1.0` pin is irrelevant at read time
(libcudf is C++ and never touches numba-cuda). Force-installing cudf 26.6 via `uv --override`
(see `cudf_inject_overrides.txt`) makes it coexist with both cuda.compute and the 2.8.11
RawKernel backend. `gpu_read_cudf.py` wraps device buffers as awkward with no host round-trip;
reader is selectable via `AK_BENCH_READER=cudf|shim`.

**(b) The JIT fix is in `awkward_src`**, so Q5/Q6/Q8 are measured post-fix.

### Finding 1 — the load stage was hiding a large host→device cost
With cudf, the "load" stage (host→device copy + `ak.Array` conversion) nearly vanishes:

| | shim load | cudf load |
|---|---:|---:|
| Q1 flat col @1M  | ~9 ms   | ~0.2 ms |
| Q1 flat col @10M | **~135 ms** | ~0.2 ms |
| Q6 jagged @1M    | 15.4 ms | 3.0 ms  |

Read times themselves are comparable (cudf ≈ pyarrow once warm). The win is eliminating the
copy/convert. This was previously excluded from the comparison as "shim artifact" — it's now
a real, measurable, scale-dependent cost (recommendation 7 below is superseded).

### Finding 2 — the read path affects *baseline robustness*, not just timing
Baseline **Q3** (`ValueError: Negative dimensions`) failed at ≥1M under the pyarrow shim, but
**succeeds at all scales with cudf**. The shim produced an option-typed / int32-offset layout
that tripped an old RawKernel bug; cudf yields a clean non-option int64-offset `ListOffsetArray`
that doesn't. So Q3's "failure" was a read-layout artifact — report honestly. Baseline **Q8**
still fails at every scale (genuine argmin-on-empty-sublists bug): `bad_alloc` @100k (via rmm),
`Negative dimensions` @1M, `CUDA_ERROR_ILLEGAL_ADDRESS` @10M.

### Finding 3 — post-fix compute speedups (baseline ÷ awkward3), all scales
| Q | 100k | 1M | 10M |
|---|---:|---:|---:|
| Q5 | 57.7× | 317.9× | **1485×** |
| Q6 | 23.4× | 36.8× | 36.8× |
| Q3 | 0.68× | 0.56× | 0.48× |
| Q4 | 1.11× | 1.33× | 1.93× |
| Q7 | 0.93× | 0.91× | 0.88× |

Q5 compute is ~flat with scale (35.6→63→135 ms) while baseline grows super-linearly
(2.05→20.1→200.6 s). Light queries (Q3/Q7) stay host-bound and ~par, exactly as the busy%
analysis predicts.

### Finding 4 — Q1 coffea-fill vs cuda.compute `histogram_even` (`query1b_gpu`)
Swapping coffea's bincount fill for `cuda.compute.histogram_even` gives **bit-identical** bins
and a faster fill (10M: 31.4 → 1.97 ms, ~16×; exact match at 100k/1M/10M). Gotcha: the
`d_histogram` counter must be int32/uint64 — CUB has no signed-int64 `atomicAdd`, so an int64
counter fails to NVRTC-compile (`error 999`). Caveat: the fill stage also varies with the
*reader* (cudf's clean NumpyArray speeds coffea's bincount vs the shim's option layout), which
conflates read vs downstream cost — normalize layout to isolate the pure read effect.

---

## Methodology

- GPU: **RTX 6000 Ada (49 GB)**, CUDA 13.3. Device pinned with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
- Two envs: `.venv-awkward3` (cuda.compute) and `.venv-baseline` (awkward 2.8.11, RawKernel).
- Harness `perf/prof_query.py`: warms each query **twice** (absorbs first-time JIT), then
  profiles **one** iteration bounded by `cudaProfilerStart/Stop`; wraps `ak.*` ops in NVTX.
- `nsys profile --capture-range=cudaProfilerApi` ⇒ the report contains *only* the timed
  iteration. Parsed straight from the `.sqlite` (`perf/parse_nsys.py`) for kernel count,
  GPU-busy ns, memcpy, D2H-sync count, and per-NVTX attribution.
- `perf/cprofile_query.py` for Python hotspots; `perf/micro_cccl_cache.py` to isolate the
  cuda.compute cache behavior; `ncu` SpeedOfLight spot-checks for kernel efficiency.
- All artifacts under `perf/` (`nsys/`, `reports/*.jsonl`, scripts).

## The evidence (awkward3, 1M events, timed iteration)

`busy%` = GPU kernel time ÷ compute-stage wall. Low = host-bound.

| Q | comp (ms) | GPU-busy (ms) | busy% | #kernels | #launch | #D2H/sync | dominant kernels |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | ~0   | 0.14 | — | 39   | 31  | 12  | histogram fill only |
| 2 | ~0   | 0.40 | — | 39   | 31  | 28  | flatten + fill |
| 3 | 7.9  | 1.83 | 23% | 215  | 177 | 131 | 66% cupy ufuncs |
| 4 | 5.5  | 0.77 | 14% | 95   | 83  | 92  | cupy ufuncs + count |
| 5 | 2328 | 3.9  | **0.1%** | 1129 | 937 | 960 | **2.1 s in cuda.compute `call_build`** |
| 6 | 3766 | 303  | 8%  | 800  | 686 | 881 | combinations rebuilds (n=3) |
| 7 | 99.5 | 20.7 | 21% | 1115 | 903 | 2060| cross-join `nearest` |
| 8 | 4798 | 11.5 | **0.1%** | 2502 | 2046| 2041| combinations rebuilds |

Baseline (RawKernel) contrast where it runs:

| Q | ak3 comp | base comp | speedup | base #kernels | base #D2H |
|---|---:|---:|---:|---:|---:|
| 4 | 5.5 ms   | 4.85 ms   | 0.88× | 98     | 90 |
| 5 | 2328 ms  | 27 833 ms | 12.0× | **285 336** (284 k = `cupy_fill`) | **2 000 957** |
| 6 | 3766 ms  | 29 642 ms | 7.9×  | **442 682** | **2 000 877** |
| 7 | 99.5 ms  | 92.6 ms   | 0.93× | 1022   | 2043 |

Q3 & Q8 fail outright on the baseline (the robustness story from `RESULTS.md`).

### Reading the table
- **Every query has tiny GPU-busy%.** Even the substantive ufunc kernels are *fine* — `ncu`
  shows `cupy_equal__int64_int64_bool` at **80% DRAM throughput** (memory-bound, efficient).
  The problem is their *number* and the host gaps + syncs between them, not their speed.
- **Light queries: ak3 ≈ baseline** (Q4 95 vs 98 kernels; Q7 1115 vs 1022). Same work, same
  per-op overhead ⇒ same time. cuda.compute's segmented reducers are off the critical path.
- **Q5 baseline = 285 k kernels / 2 M syncs.** The old combinations/reduction path launches a
  kernel (and a sync) per segment — `cupy_fill` ×284 k. cuda.compute collapses this to ~1.1 k
  kernels. *That* is the Q5/Q6 win — fewer launches/syncs, **not** faster kernels.

## Root cause of the Q5/Q6/Q8 overhead: per-call JIT rebuild in `ak.combinations`

cProfile of Q5 compute (2 warmups already done):

```
2  2.101s  2.101s  cuda/compute/_cccl_interop.py:241(call_build)   <- JIT build, in the TIMED run
   ...      from ak.combinations -> unary_transform -> make_unary_transform -> call_build
```

`call_build` is cuda.compute's compile step. It runs **every call** (still 2.1 s with 5
warmups), i.e. the build **cache misses every time**. Why:

- `ak.combinations` (`awkward/_connect/cuda/_compute.py`, `awkward_ListArray_combinations`)
  builds a fresh Python closure `fill_pos` per combination position `k` and passes it to
  `unary_transform`. There are `n` passes ⇒ **n rebuilds per call** (2 for Q5, 3 for Q6).
- `make_unary_transform` is cached on its args. The op closure is keyed by `CachableFunction`
  (bytecode + **closure cell contents** + globals). `fill_pos` captures `length` — the number
  of lists ≈ **1 000 000**, a *Python int*.
- cuda.compute's `_make_hashable` (`cuda/compute/_caching.py`) sends plain Python ints to the
  `else: return id(value)` branch. A large int is a fresh heap object each call ⇒ fresh `id`
  ⇒ different cache key ⇒ **rebuild**. (Small ints −5..256 are interned, so `n`/`k` are fine;
  device arrays are keyed by dtype+shape, also fine. Only `length` poisons the key.)

Decisive microbenchmark (`perf/micro_cccl_cache.py`, calls 2..6 after a cold first call):

```
A  fresh iterators+closure, no big int      : 1130 ms cold, then 0.0 ms   (cache HITS)
D  closure captures a fresh large Python int:  702 ms cold, then ~698 ms each  (REBUILDS every call)
E  same but captures np.int64               :  697 ms cold, then 0.0 ms   (cache HITS)   <- the fix
```

and directly: `_make_hashable(int(str(10**6))) == _make_hashable(int(str(10**6)))` → `False`.

## The fix (applied & validated)

`awkward/_connect/cuda/_compute.py`, in `awkward_ListArray_combinations`, coerce `length`
to a numpy scalar **before** the `fill_pos` closure captures it:

```python
# Coerce length to a numpy scalar so the fill_pos closure is cache-stable:
# cuda.compute keys plain Python ints by id() (fresh each call -> JIT rebuild),
# but numpy scalars by dtype+value.
length = np.int64(length)
def make_pass(k, carry_k):
    def fill_pos(g):
        ...
```

Result: `call_build` disappears from the timed run; **Q5 2328→57 ms, Q6 3766→367 ms,
Q8 4798→111 ms** (numbers above). Outputs validated against CPU.

This is the minimal, local fix. Better long-term options below.

## Per-query verdict & remaining gaps

- **Q1, Q2** — no compute; load+histogram only. Nothing to fix on the backend.
- **Q3** — light. 215 kernels / 131 syncs for `flatten(pt[|eta|<1])`. Host-bound; ~par with
  baseline. Gap = eager per-op dispatch (cupy ufuncs) + a sync per structural op.
- **Q4** — light. `sum(pt>40,axis=1)>=2` then mask. 95 kernels / 92 syncs. Host-bound; ~par.
- **Q5** — **fixed.** Was 97 % `call_build`. Now 57 ms; GPU still only ~7 % busy, so the next
  ceiling is again host dispatch (the vector `+`/`.mass` expansion + per-op syncs).
- **Q6** — **fixed** (10×). Larger residual than Q5 because n=3 ⇒ more combinations data and
  3 passes; genuine GPU work is larger here. (Pre-existing argmin tie-break: 0.011 % of rows.)
- **Q7** — light-ish but 2060 syncs (the `nearest` cross-join + many cuts). Host-bound; ~par.
  Biggest sync count of the "light" group — best candidate for sync-batching wins.
- **Q8** — **fixed** (43×). Combinations + argmin/argmax + masking; only ak3 runs it at all.

## Recommendations

### In awkward (high value, low risk)
1. **Ship the `length = np.int64(length)` fix** (done here) and audit every closure passed to
   cuda.compute (`unary_transform`/`inclusive_scan`/`reduce_into` in `_compute.py`) for
   captured large Python ints / Python floats — coerce them all to numpy scalars, or pass them
   as device-side iterator state rather than closure captures. `combinations_len` is already
   clean (captures only device arrays + small ints); `fill_pos` was the offender.
2. **Cut the synchronization count.** Each structural op reads an offset/length back to host
   (`index_as_shape_item`, kernel-signature `.item()` calls) and stalls the stream. Where the
   next allocation size is itself a device value, keep it on device (allocate an upper bound,
   or use cuda.compute iterators sized by a device scalar) to defer/batch syncs. This is the
   main lever for the light queries (Q3/Q4/Q7) and for squeezing Q5/Q6/Q8 further.
3. **Fewer, fused kernels for elementwise chains.** Comparisons/arithmetic dispatch to
   individual CuPy ufunc kernels (one launch + often a temporary each). CuPy `fuse`, or routing
   elementwise+reduction chains through a single cuda.compute `transform`/`transform_reduce`,
   would replace dozens of tiny launches with one.

### In cuda.compute (CCCL)
4. **Key Python ints/floats by value, not `id()`.** In `_caching.py::_make_hashable`, handle
   `int`/`float`/`bool` by value (e.g. `("py.int", value)`). This makes *any* user's closure
   that captures a Python scalar cache-stable and prevents this whole class of silent
   rebuild-every-call regressions. (Today only numpy scalars and device arrays are safe.)
   **Filed: [NVIDIA/cccl#9626](https://github.com/NVIDIA/cccl/issues/9626).**
5. **Optionally warn on suspected per-call rebuilds** (e.g. if the same `__qualname__` build
   runs >N times with distinct keys), which would have surfaced this immediately.

### Benchmark methodology
6. The "warm" numbers in `RESULTS.md` for Q5/Q6 **included** per-call JIT rebuild — warm-up
   did not exclude it because the cache missed every call. With the fix they drop ~10–40×.
   **Done:** headline tables re-run at all scales (see Update above); also report **GPU-busy%**
   so the host-bound nature is visible.
7. ~~read/load is the pyarrow-shim artifact (cudf dropped)~~ **Superseded:** cudf GPU-direct
   reads are restored in both envs, so read/load are now comparable and the load-stage cost is
   real (Finding 1). Compute-stage GPU-busy% remains the cleanest *backend* metric, but
   end-to-end totals are now meaningful too. One caveat: the awkward layout produced by the
   reader changes the downstream coffea fill cost — normalize layout when isolating read effects.

## Reproduce
```
cd perf
bash run_sweep.sh ak3 1M reports/sweep_ak3_1M.jsonl          # profile all queries
bash run_sweep.sh base 1M reports/sweep_base_1M.jsonl 1 2 4 5 6 7
../.venv-awkward3/bin/python summarize.py reports/sweep_ak3_1M.jsonl reports/sweep_base_1M.jsonl
../.venv-awkward3/bin/python cprofile_query.py 5 ../data/pq_subset_1M.parquet 2   # Python hotspots
../.venv-awkward3/bin/python micro_cccl_cache.py                                   # cache repro
```

---

## Update (2026-06-29): cuda.compute-native rewrites of the host-bound queries

The two queries where awkward3 was *slower* than the RawKernel baseline (Q3, Q7) were
rewritten directly on cuda.compute primitives and now **beat baseline end-to-end**, with
bit-identical histograms. Code: `perf/fast_queries.py`.

| Q | baseline | awkward3 default | fused cuda.compute | fused vs baseline |
|---|--:|--:|--:|--:|
| Q3 `flatten(pt[\|eta\|<1])` | 29.2 ms | 37.0 ms (0.79×) | **19.7 ms** | **1.5×** |
| Q7 `nearest` HT             | 277.6 ms | 290 ms (0.94×) | **91.2 ms** | **3.0×** |

- **Q3** = one `DeviceSelect`: `flatten(jagged[mask])` discards event structure, so it's pure
  stream-compaction of the flat `pt` content by the flat `|eta|<1` predicate (select `cond`
  reads the captured `eta` array) + a gather. Compute 16.5→~0.1 ms; **193→3 kernels**.
- **Q7** = `nearest` reformulated as a per-jet `segmented_reduce(MINIMUM)` of dR over the
  event's leptons (segments = per-jet lepton counts; lepton-less jets handled explicitly),
  then a per-event `segmented_reduce(PLUS)` for HT, then `histogram_even`. Compute ~228→~30 ms;
  the 1115-kernel / 2060-sync cross-join collapses to a handful of primitives.
- Both are now **read-bound** (Q3 ~16/19.7 ms, Q7 ~53/91 ms are the cudf read) — compute is no
  longer the bottleneck.

### Corrected principle
The "structural" ops earlier called not-fusable DO map to cuda.compute: `flatten(x[mask])` →
`select`; structure-preserving `x[mask]` → `select` + `segmented_reduce` (offsets = segmented
sum of mask); `nearest`/cross-join → `segmented_reduce` + `binary_search`. The real work is
teaching awkward's CUDA backend to *recognize* these shapes (e.g. route `flatten(x[bool])` to
`select`) so users get it for free.

### Iterator fusion — how far it goes, and the wall
`histogram_even` and `segmented_reduce` accept an iterator for their input, so **stateless**
transform-iterator ops fuse into the kernel with zero temporaries — demonstrated for Q4
(`TransformIterator(content, x>40)` → `segmented_reduce`: **48→3 kernels**) and the fill
(`histogram_even`: 38→2). But the *maximal* zero-temporary fusion of Q3/Q7 needs the iterator
op to **index other arrays** (gather `pt` by selected index; read η/φ by pair index), i.e. a
**stateful** op. A stateful op works as a *direct* algorithm op (select `cond`, `unary_transform`)
but **crashes when wrapped in a `TransformIterator` fed to `segmented_reduce`/`histogram_even`**.
Minimal reproducer (`reduce_into` over a `TransformIterator`):

```
plain array d_in ............. OK
TransformIterator stateless .. OK     (return i*2.0)
TransformIterator stateful ... FAIL   (return src[i]*2.0; src is a device array)  -> ValueError / CUDA_ERROR_LAUNCH_FAILED
```

This is a cuda.compute limitation (filed: [NVIDIA/cccl#9627](https://github.com/NVIDIA/cccl/issues/9627));
it's the only thing between the (working) array-materializing fused versions and zero-temporary
fusion. Since Q3/Q7 are now read-bound, the remaining payoff is small — the gather/dr temporaries
no longer dominate.

**Upstream issues filed:** [cccl#9626](https://github.com/NVIDIA/cccl/issues/9626)
(Python-scalar cache key → per-call JIT rebuild; root cause of the `ak.combinations` slowdown) and
[cccl#9627](https://github.com/NVIDIA/cccl/issues/9627) (stateful op in a `TransformIterator` fails
— blocks zero-temporary gather/segmented-reduce fusion).

Artifacts: `perf/fast_queries.py` (Q3/Q7 fused + iter variants), `perf/fuse_proto.py`
(Q4 transform-reduce, fill, dR fusion), `perf/q3_select_proto.py`, `perf/nvtx_suite.py`
(generously NVTX-annotated suite for the Nsight GUI: `perf/nsys/adl_suite_1M.nsys-rep`).
