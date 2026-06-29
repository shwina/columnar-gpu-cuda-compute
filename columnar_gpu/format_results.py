"""Read cmp_*.txt (RESULT_JSON lines) and print markdown comparison tables.

Usage: python format_results.py logs/cmp_1M.txt

Prints two tables per log:
  1. Warm GPU compute-stage time per query (the backend metric): baseline
     (2.8.11 RawKernel) vs awkward3 (cuda.compute), with speedup.
  2. Full per-stage breakdown (read/load/comp/fill/total) for each env, with
     end-to-end total speedup. With cudf GPU-direct reads in both envs, the
     read/load stages and the total are now comparable across backends.
"""
import sys, json

path = sys.argv[1]
rows = {}
for line in open(path):
    line = line.strip()
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    rows.setdefault(d["q"], {})[d["label"]] = d


def ms(d, key):
    if d is None or not d.get("ok"):
        return "—"
    return f"{d[key]*1000:.2f}"


def comp_cell(d):
    if d is None:
        return "—"
    if not d.get("ok"):
        return f"FAIL ({d.get('error','?')[:36]})"
    return f"{d['comp']*1000:.2f} ms"


def speedup(b, a, key):
    if b and a and b.get("ok") and a.get("ok") and a.get(key, 0) > 0:
        return f"{b[key]/a[key]:.2f}×"
    return "—"


# ---- Table 1: compute-stage backend comparison ----
print(f"\n## Warm GPU compute time per query — {path}\n")
print("| Query | baseline (RawKernel) | awkward3 (cuda.compute) | speedup |")
print("|---|---|---|---|")
for q in sorted(rows):
    b = rows[q].get("baseline"); a = rows[q].get("awkward3")
    print(f"| Q{q} | {comp_cell(b)} | {comp_cell(a)} | {speedup(b, a, 'comp')} |")

# ---- Table 2: full per-stage breakdown (ms), end-to-end total speedup ----
print(f"\n## Full per-stage breakdown (ms) — {path}\n")
print("| Query | env | read | load | comp | fill | total | total speedup |")
print("|---|---|---|---|---|---|---|---|")
for q in sorted(rows):
    b = rows[q].get("baseline"); a = rows[q].get("awkward3")
    tsp = speedup(b, a, "total")
    for lab, d in (("baseline", b), ("awkward3", a)):
        if d is None:
            continue
        if not d.get("ok"):
            print(f"| Q{q} | {lab} | — | — | — | — | FAIL | {tsp if lab=='awkward3' else ''} |")
            continue
        sp_here = tsp if lab == "awkward3" else ""
        print(f"| Q{q} | {lab} | {ms(d,'read')} | {ms(d,'load')} | "
              f"{ms(d,'comp')} | {ms(d,'fill')} | {ms(d,'total')} | {sp_here} |")
print()
