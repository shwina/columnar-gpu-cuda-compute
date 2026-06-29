"""Wire cuda.compute histogram_even into all GPU query fills in run_adl_queries.py.
Replaces `q_hist[...].fill(axis=expr)` with `_cc_fill(q_hist[...], "axis", expr)`
ONLY inside functions whose name ends with `_gpu` (leaves CPU fills untouched),
and inserts the _cc_fill helper once. Idempotent.
"""
import re, sys

PATH = "/home/coder/columnar_gpu_bench/columnar_gpu/run_adl_queries.py"
src = open(PATH).read()

HELPER = '''
def _cc_fill(q_hist, axis_name, values):
    """Fill a coffea gpu_hist via cuda.compute.histogram_even (one fused CUB
    DeviceHistogram kernel) instead of coffea's clip/ravel/bincount chain.
    Enabled with AK_BENCH_HIST=cc; otherwise, or if cuda.compute is unavailable
    (e.g. the baseline env), falls back to q_hist.fill(...). Counts go into the
    in-range bins of the dense array; flow bins stay 0 (matches flow="none").
    """
    if os.environ.get("AK_BENCH_HIST") != "cc":
        q_hist.fill(**{axis_name: values}); return
    try:
        import cuda.compute as cc
    except Exception:
        q_hist.fill(**{axis_name: values}); return
    e = q_hist.axis(axis_name).edges()
    e = e.get() if hasattr(e, "get") else np.asarray(e)
    nbins = len(e) - 1; lo = float(e[0]); hi = float(e[-1])
    arr = ak.drop_none(ak.ravel(values))
    v = cp.ascontiguousarray(ak.to_cupy(arr).astype(cp.float32))
    counts = cp.zeros(nbins, dtype=cp.int32)
    cc.histogram_even(d_samples=v, d_histogram=counts, num_output_levels=nbins + 1,
                      lower_level=np.float32(lo), upper_level=np.float32(hi),
                      num_samples=int(v.size))
    dense = cp.zeros(q_hist._dense_shape, dtype=cp.float64)
    dense[1:nbins + 1] = counts.astype(cp.float64)
    q_hist._sumw[()] = dense

'''

if "_cc_fill" not in src:
    # insert helper right before the first query function definition
    m = re.search(r"^# Q1 query GPU", src, re.M)
    src = src[:m.start()] + HELPER + "\n" + src[m.start():]

# Split into top-level functions; patch fills only inside *_gpu bodies.
lines = src.splitlines(keepends=True)
out = []
cur_is_gpu = False
fill_re = re.compile(r'^(\s*)(q_hist(?:_\d)?)\.fill\((\w+)=(.+)\)\s*$')
defn_re = re.compile(r'^def (\w+)\s*\(')
n_patched = 0
for ln in lines:
    d = defn_re.match(ln)
    if d:
        cur_is_gpu = d.group(1).endswith("_gpu") and d.group(1) != "query1b_gpu"
    if cur_is_gpu:
        m = fill_re.match(ln)
        if m and "_cc_fill" not in ln:
            indent, hist, axis, expr = m.groups()
            ln = f'{indent}_cc_fill({hist}, "{axis}", {expr})\n'
            n_patched += 1
    out.append(ln)

open(PATH, "w").write("".join(out))
print(f"patched {n_patched} GPU fill sites; helper present: {'_cc_fill' in open(PATH).read()}")
