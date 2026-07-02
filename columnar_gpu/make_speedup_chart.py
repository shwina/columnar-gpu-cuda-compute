"""Shareable benchmark chart from the cudf-reader re-run logs.

Reads logs/cmp_{100k,1M,10M}.txt (RESULT_JSON lines) and renders a two-panel PNG:
  A) compute-stage speedup (baseline / awkward3) per query x scale, log axis, parity line
  B) absolute compute time at 10M, baseline vs awkward3 (log axis)

Usage: python make_speedup_chart.py            # writes plots/benchmark_cudf_rerun.png
Colors are the dataviz reference palette's categorical slots, used in fixed order;
every bar is direct-labeled (satisfies the light-mode contrast relief rule).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = ["100k", "1M", "10M"]
# dataviz reference palette (light mode), fixed slot order
C_SCALE = {"100k": "#2a78d6", "1M": "#1baf7a", "10M": "#eda100"}   # slots 1,2,3
C_BASE, C_AK3 = "#e34948", "#2a78d6"                                # slots 6,1
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e6e6e2"

def load(scale):
    rows = {}
    with open(os.path.join(HERE, "logs", f"cmp_{scale}.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                rows.setdefault(d["q"], {})[d["label"]] = d
    return rows

data = {s: load(s) for s in SCALES}

# Fused cuda.compute rewrites for the host-bound queries (query3c/7c), which replace
# awkward3's stock Q3/Q7 here. Keyed (scale, qnum) -> comp seconds.
FUSED = {}
with open(os.path.join(HERE, "logs", "fused.txt")) as f:
    for line in f:
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            FUSED[(d["scale"], int(d["q"][:-1]))] = d   # "3c" -> 3
FUSED_Q = {3, 4, 7}

def ak_comp(scale, q):
    """awkward3 compute time (s): fused rewrite for Q3/Q7, else stock query{q}_gpu."""
    if q in FUSED_Q:
        return FUSED[(scale, q)]["comp"]
    return data[scale][q]["awkward3"]["comp"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE})
fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 9), dpi=200)

# ---- Panel A: speedup (baseline / awkward3), queries with real compute + a baseline ----
QA = [3, 4, 5, 6, 7]           # Q1/Q2 ~0ms (noise); Q8 has no baseline
x = np.arange(len(QA)); w = 0.26
for i, s in enumerate(SCALES):
    vals = []
    for q in QA:
        b = data[s][q].get("baseline"); ac = ak_comp(s, q)
        vals.append(b["comp"]/ac if b and b.get("ok") and ac>0 else np.nan)
    bars = axA.bar(x + (i-1)*w, vals, w, label=s, color=C_SCALE[s], zorder=3)
    for rect, v in zip(bars, vals):
        if not np.isnan(v):
            lbl = f"{v:.0f}×" if v >= 10 else f"{v:.2f}×"
            axA.text(rect.get_x()+rect.get_width()/2, v*1.06, lbl, ha="center", va="bottom",
                     fontsize=8, color=INK, rotation=0)
axA.axhline(1.0, color=MUTE, lw=1.5, ls="--", zorder=2)
axA.text(-0.45, 1.18, "parity (1×)", ha="left", va="bottom", fontsize=8, color=MUTE)
axA.set_yscale("log"); axA.set_ylim(0.3, 4000)
axA.set_xticks(x); axA.set_xticklabels([f"Q{q}†" if q in FUSED_Q else f"Q{q}" for q in QA])
axA.set_ylabel("compute speedup  (baseline ÷ awkward3)")
axA.set_title("cuda.compute vs 2.8.11 RawKernel — GPU compute-stage speedup by query & scale",
              fontsize=12, color=INK, loc="left", pad=10)
axA.legend(title="events", frameon=False, ncol=3, loc="upper left")
axA.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)
axA.text(0.0, -0.15,
         "Above the line = cuda.compute faster.  † Q3/Q4/Q7 use cuda.compute-native fused rewrites "
         "(q3c: DeviceSelect;  q4c: segmented_reduce + DeviceSelect;  q7c: segmented_reduce + PermutationIterator ΔR).\n"
         "Q1/Q2 omitted (compute ≈ 0, sub-µs noise).  Q8 omitted — baseline fails at every scale; "
         "awkward3 runs it (124.6 / 243.6 / 904.5 ms).",
         transform=axA.transAxes, fontsize=8, color=MUTE)

# ---- Panel B: absolute compute time @10M, baseline vs awkward3 ----
QB = [3, 4, 5, 6, 7, 8]
xb = np.arange(len(QB)); w2 = 0.38
base_vals = [ (data["10M"][q].get("baseline") or {}).get("comp", np.nan)*1e3
              if (data["10M"][q].get("baseline") or {}).get("ok") else np.nan for q in QB]
ak_vals   = [ ak_comp("10M", q)*1e3 for q in QB]
bbars = axB.bar(xb - w2/2, base_vals, w2, label="baseline (RawKernel)", color=C_BASE, zorder=3)
abars = axB.bar(xb + w2/2, ak_vals,   w2, label="awkward3 (cuda.compute)", color=C_AK3, zorder=3)
for rect, v in list(zip(bbars, base_vals)) + list(zip(abars, ak_vals)):
    if not np.isnan(v):
        lbl = f"{v/1000:.0f}s" if v >= 1000 else f"{v:.0f}ms"
        axB.text(rect.get_x()+rect.get_width()/2, v*1.08, lbl, ha="center", va="bottom", fontsize=7.5, color=INK)
# Q8 baseline fails -> mark it
q8i = QB.index(8)
axB.text(q8i - w2/2, 0.5, "FAIL", ha="center", va="bottom", fontsize=8, color=C_BASE, rotation=90, fontweight="bold")
axB.set_yscale("log"); axB.set_ylim(1, 1e6)
axB.set_xticks(xb); axB.set_xticklabels([f"Q{q}†" if q in FUSED_Q else f"Q{q}" for q in QB])
axB.set_ylabel("compute time @10M events  (ms, log)")
axB.set_title("Absolute GPU compute time at 10M events  (Q3/Q4/Q7 = fused rewrites)",
              fontsize=12, color=INK, loc="left", pad=10)
axB.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.0))
axB.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)

fig.suptitle("ADL HEP queries on RTX 6000 Ada — warm, JIT excluded, cudf GPU-direct reads (both envs)",
             fontsize=10, color=MUTE, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(HERE, "plots", "benchmark_cudf_rerun.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
