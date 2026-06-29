"""Microbenchmark: is cuda.compute's build cache hitting for the call pattern
ak.combinations uses (unary_transform over CountingIterator->DiscardIterator
with a fresh closure each call)?

Times repeated calls and reports per-call wall. If the cache hits, only the
first call should pay the ~1s build; if it misses, every call rebuilds.
"""
import time, statistics
import cupy as cp
from cuda.compute import (CountingIterator, DiscardIterator, unary_transform,
                          reduce_into, OpKind, clear_all_caches)

cp.cuda.Device(0).synchronize()

def timeit(fn, n=5):
    ts = []
    for _ in range(n):
        cp.cuda.Device(0).synchronize(); t0 = time.perf_counter()
        fn(); cp.cuda.Device(0).synchronize()
        ts.append(time.perf_counter() - t0)
    return ts

N = 1_000_000
out = cp.zeros(1, dtype=cp.int64)

# ---- Case A: mirror ak.combinations exactly (fresh iterators + fresh closure) ----
def caseA():
    carry = cp.empty(N, dtype=cp.int64)
    def make_pass(k, carry_k):
        def fill_pos(i):
            carry_k[i] = i + k
            return 0
        return fill_pos
    unary_transform(d_in=CountingIterator(cp.int64(0)),
                    d_out=DiscardIterator(),
                    op=make_pass(0, carry),
                    num_items=N)

# ---- Case B: reuse the SAME iterator objects and SAME op object ----
_ci = CountingIterator(cp.int64(0)); _di = DiscardIterator()
_carry = cp.empty(N, dtype=cp.int64)
def _fill(i):
    _carry[i] = i
    return 0
def caseB():
    unary_transform(d_in=_ci, d_out=_di, op=_fill, num_items=N)

# ---- Case D: closure captures a LARGE fresh Python int (like `length`) ----
import numpy as np
def caseD():
    carry = cp.empty(N, dtype=cp.int64)
    length = int(str(N))       # genuinely fresh large Python int object each call (like int(device_val))
    def fill(g):
        x = length - 1         # capture `length` (Python int)
        carry[g] = x
        return 0
    unary_transform(d_in=CountingIterator(cp.int64(0)), d_out=DiscardIterator(),
                    op=fill, num_items=N)

# ---- Case E: same, but capture a numpy.int64 scalar (the proposed fix) ----
def caseE():
    carry = cp.empty(N, dtype=cp.int64)
    length = np.int64(N)       # numpy scalar -> _make_hashable keys by dtype+bytes
    def fill(g):
        x = length - 1
        carry[g] = x
        return 0
    unary_transform(d_in=CountingIterator(cp.int64(0)), d_out=DiscardIterator(),
                    op=fill, num_items=N)

for name, fn in [("A: combinations-style (fresh iters+closure, no big int)", caseA),
                 ("B: reused iters+op", caseB),
                 ("D: closure captures LARGE Python int `length`", caseD),
                 ("E: closure captures np.int64 `length` (FIX)", caseE)]:
    clear_all_caches()
    ts = timeit(fn, 6)
    print(f"\n{name}")
    print(f"   call 1 (cold): {ts[0]*1000:8.1f} ms")
    print(f"   calls 2..6   : " + ", ".join(f"{t*1000:.1f}" for t in ts[1:]) + " ms")
    print(f"   median 2..6  : {statistics.median(ts[1:])*1000:8.1f} ms")
