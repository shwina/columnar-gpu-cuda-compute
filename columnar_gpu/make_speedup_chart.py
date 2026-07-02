"""Shareable benchmark chart from the cudf-reader re-run logs.

Reads logs/cmp_{100k,1M,10M}.txt + logs/fused.txt (RESULT_JSON lines) and renders a
three-panel PNG:
  A) compute-stage speedup (baseline / our work) per query x scale, log, parity line
  B) end-to-end speedup (read+load+compute+fill) per query x scale, log, parity line
  C) absolute end-to-end wall-clock at 10M, baseline vs our work (log ms)

"our work" = fused cuda.compute rewrites for the host-bound queries (Q3/Q4/Q7 ->
query3c/4c/7c) and the combinations JIT fix for Q5/Q6/Q8 (stock awkward3 otherwise).

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

FUSED = {}
with open(os.path.join(HERE, "logs", "fused.txt")) as f:
    for line in f:
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            FUSED[(d["scale"], int(d["q"][:-1]))] = d   # "3c" -> 3
FUSED_Q = {3, 4, 7}

def ours(scale, q, field):
    """our-work value (s) for `field` ('comp'|'total'): fused rewrite for Q3/Q4/Q7,
    else stock awkward3 (Q5/Q6/Q8)."""
    if q in FUSED_Q:
        return FUSED[(scale, q)][field]
    return data[scale][q]["awkward3"][field]

def qlabel(q):
    return f"Q{q}†" if q in FUSED_Q else f"Q{q}"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE})
fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11, 14), dpi=200)

QA = [3, 4, 5, 6, 7]   # Q1/Q2 ~0 compute (noise); Q8 has no baseline

def speedup_panel(ax, field, ylabel, title, ymax):
    x = np.arange(len(QA)); w = 0.26
    for i, s in enumerate(SCALES):
        vals = []
        for q in QA:
            b = data[s][q].get("baseline"); ov = ours(s, q, field)
            vals.append(b[field]/ov if b and b.get("ok") and ov > 0 else np.nan)
        bars = ax.bar(x + (i-1)*w, vals, w, label=s, color=C_SCALE[s], zorder=3)
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                lbl = f"{v:.0f}×" if v >= 10 else f"{v:.2f}×"
                ax.text(rect.get_x()+rect.get_width()/2, v*1.06, lbl, ha="center",
                        va="bottom", fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTE, lw=1.5, ls="--", zorder=2)
    ax.text(-0.45, 1.15, "parity (1×)", ha="left", va="bottom", fontsize=8, color=MUTE)
    ax.set_yscale("log"); ax.set_ylim(0.3, ymax)
    ax.set_xticks(x); ax.set_xticklabels([qlabel(q) for q in QA])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.legend(title="events", frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ---- Panel A: compute-stage speedup ----
speedup_panel(axA, "comp", "compute speedup  (baseline ÷ ours)",
              "GPU compute-stage speedup by query & scale  (kernels only)", 4000)
axA.text(0.0, -0.16,
         "† Q3/Q4/Q7 use cuda.compute-native fused rewrites (q3c: DeviceSelect;  q4c: segmented_reduce + "
         "DeviceSelect;  q7c: segmented_reduce + PermutationIterator ΔR).  Q1/Q2 omitted (compute ≈ 0).\n"
         "Q8 omitted — baseline fails at every scale; awkward3 runs it (124.6 / 243.6 / 904.5 ms compute).",
         transform=axA.transAxes, fontsize=8, color=MUTE)

# ---- Panel B: end-to-end speedup ----
speedup_panel(axB, "total", "end-to-end speedup  (baseline ÷ ours)",
              "End-to-end speedup by query & scale  (read + load + compute + fill)", 2000)
axB.text(0.0, -0.16,
         "Same GPU-direct cudf read in both, so read/load are shared. The combinatoric queries (Q5/Q6) "
         "stay huge because compute dominates their total;\nthe light queries (Q3/Q4/Q7) shrink to ~1.2–3.9× "
         "— their compute win is real but they are read-bound end-to-end.",
         transform=axB.transAxes, fontsize=8, color=MUTE)

# ---- Panel C: absolute end-to-end wall-clock at 10M ----
QC = [3, 4, 5, 6, 7, 8]
xb = np.arange(len(QC)); w2 = 0.38
base_vals = [ (data["10M"][q].get("baseline") or {}).get("total", np.nan)*1e3
              if (data["10M"][q].get("baseline") or {}).get("ok") else np.nan for q in QC]
our_vals  = [ ours("10M", q, "total")*1e3 for q in QC]
bbars = axC.bar(xb - w2/2, base_vals, w2, label="baseline (RawKernel)", color=C_BASE, zorder=3)
abars = axC.bar(xb + w2/2, our_vals,  w2, label="our work (cuda.compute)", color=C_AK3, zorder=3)
for rect, v in list(zip(bbars, base_vals)) + list(zip(abars, our_vals)):
    if not np.isnan(v):
        lbl = f"{v/1000:.0f}s" if v >= 1000 else f"{v:.0f}ms"
        axC.text(rect.get_x()+rect.get_width()/2, v*1.08, lbl, ha="center", va="bottom", fontsize=7.5, color=INK)
axC.text(QC.index(8) - w2/2, 0.6, "FAIL", ha="center", va="bottom", fontsize=8, color=C_BASE, rotation=90, fontweight="bold")
axC.set_yscale("log"); axC.set_ylim(1, 1e6)
axC.set_xticks(xb); axC.set_xticklabels([qlabel(q) for q in QC])
axC.set_ylabel("end-to-end time @10M events  (ms, log)")
axC.set_title("Absolute end-to-end wall-clock at 10M events  (Q3/Q4/Q7 = fused rewrites)",
              fontsize=12, color=INK, loc="left", pad=10)
axC.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.0))
axC.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"): axC.spines[sp].set_visible(False)

fig.suptitle("ADL HEP queries on RTX 6000 Ada — warm, JIT excluded, cudf GPU-direct reads (both envs)",
             fontsize=10, color=MUTE, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.99])
out = os.path.join(HERE, "plots", "benchmark_cudf_rerun.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
