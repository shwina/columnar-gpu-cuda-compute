"""Render per-query profiling summary from sweep JSONL files."""
import sys, json

def load(path):
    d = {}
    try:
        for line in open(path):
            line=line.strip()
            if not line: continue
            r = json.loads(line)
            d[r["q"]] = r
    except FileNotFoundError:
        pass
    return d

def us(ns): return ns/1000.0
def ms(ns): return ns/1e6

files = sys.argv[1:]
data = {f: load(f) for f in files}

# Per-query detail for the first (ak3) file
ak3 = load([f for f in files if "ak3" in f][0]) if any("ak3" in f for f in files) else {}
base = load([f for f in files if "base" in f][0]) if any("base" in f for f in files) else {}

print(f"{'Q':<3}{'env':<5}{'comp(ms)':>9}{'GPUbusy(ms)':>12}{'busy%':>7}{'#kern':>7}{'#launch':>8}{'#D2H/sync':>10}{'cupy%':>7}{'cub%':>7}{'awk%':>7}")
print("-"*90)
for q in range(1,9):
    for env,d in [("ak3",ak3),("base",base)]:
        r = d.get(q)
        if not r:
            continue
        if not r.get("ok",False):
            print(f"{q:<3}{env:<5}  FAILED/crashed"); continue
        comp = r.get("comp",0)*1000
        busy = ms(r.get("kern_ns",0))
        kn = r.get("kern_ns",0)
        cl = r.get("kern_by_class",{})
        tot = sum(v["ns"] for v in cl.values()) or 1
        cupy = 100*cl.get("cupy",{}).get("ns",0)/tot
        cub = 100*cl.get("cub",{}).get("ns",0)/tot
        awk = 100*cl.get("awkward",{}).get("ns",0)/tot
        d2h = r.get("memcpy",{}).get("d2h",{}).get("n",0)
        busypct = 100*busy/comp if comp>0 else 0
        print(f"{q:<3}{env:<5}{comp:>9.2f}{busy:>12.3f}{busypct:>6.0f}%{r.get('n_kernels',0):>7}{r.get('n_launch',0):>8}{d2h:>10}{cupy:>6.0f}%{cub:>6.0f}%{awk:>6.0f}%")
    print()

# speedup vs baseline (compute wall)
print("\n=== compute-wall speedup base/ak3 ===")
for q in range(1,9):
    a=ak3.get(q); b=base.get(q)
    if a and a.get("ok") and b and b.get("ok"):
        sp = (b["comp"]/a["comp"]) if a["comp"]>0 else 0
        print(f"  Q{q}: ak3 {a['comp']*1000:.2f}ms  base {b['comp']*1000:.2f}ms  speedup {sp:.2f}x")
    elif a and a.get("ok") and (not b or not b.get("ok")):
        print(f"  Q{q}: ak3 {a['comp']*1000:.2f}ms  base FAIL")
