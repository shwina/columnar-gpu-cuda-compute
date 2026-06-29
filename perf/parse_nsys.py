"""Parse an nsys .sqlite (single captured iteration) into a per-query summary.

Emits one JSON line. Because the report was captured with
--capture-range=cudaProfilerApi, everything in the DB is the timed iteration.
"""
import sys, json, sqlite3, os

db = sys.argv[1]
con = sqlite3.connect(db)
cur = con.cursor()

def has(table):
    return cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None

def sid_map():
    return dict(cur.execute("SELECT id,value FROM StringIds").fetchall())

S = sid_map()

out = {"db": os.path.basename(db)}

# ---- kernels ----
if has("CUPTI_ACTIVITY_KIND_KERNEL"):
    rows = cur.execute("""SELECT shortName, COUNT(*), SUM(end-start)
                          FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY shortName""").fetchall()
    kern = [(S.get(n, str(n)), c, t) for n, c, t in rows]
    n_kern = sum(c for _, c, t in kern)
    kern_ns = sum(t for _, c, t in kern)
    # classify
    def cls(name):
        if name.startswith("cupy") or "cupy_" in name: return "cupy"
        if ("cub::" in name or "CUB_" in name or "Device" in name or "segmented" in name
                or "DeviceReduce" in name or "DeviceScan" in name or "Histogram" in name):
            return "cub"   # CUB / cuda.compute primitives (demangled short names)
        if "awkward" in name: return "awkward"
        return "other"
    bycls = {}
    for name, c, t in kern:
        k = cls(name)
        d = bycls.setdefault(k, [0, 0])
        d[0] += c; d[1] += t
    out["n_kernels"] = n_kern
    out["kern_ns"] = kern_ns
    out["kern_by_class"] = {k: {"n": v[0], "ns": v[1]} for k, v in bycls.items()}
    top = sorted(kern, key=lambda x: -x[2])[:12]
    out["top_kernels"] = [{"name": n[:60], "n": c, "ns": t} for n, c, t in top]
else:
    out["n_kernels"] = 0; out["kern_ns"] = 0

# ---- memcpy ----
if has("CUPTI_ACTIVITY_KIND_MEMCPY"):
    rows = cur.execute("SELECT copyKind, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind").fetchall()
    # copyKind: 1=H2D, 2=D2H, 8=D2D (nsys CUPTI codes)
    cpk = {1: "h2d", 2: "d2h", 8: "d2d"}
    mem = {}
    for k, c, t in rows:
        mem[cpk.get(k, f"k{k}")] = {"n": c, "ns": t}
    out["memcpy"] = mem

# ---- API (runtime) calls ----
for tbl in ["CUPTI_ACTIVITY_KIND_RUNTIME"]:
    if has(tbl):
        rows = cur.execute(f"SELECT nameId, COUNT(*), SUM(end-start) FROM {tbl} GROUP BY nameId").fetchall()
        api = {}
        for nid, c, t in rows:
            api[S.get(nid, str(nid))] = {"n": c, "ns": t}
        # pull the interesting ones
        want = ["cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaLaunchKernel",
                "cuLaunchKernel", "cudaMemcpyAsync", "cudaMallocAsync", "cudaMalloc",
                "cudaFreeAsync", "cudaFree", "cudaStreamSynchronize_v2"]
        out["api"] = {k: api[k] for k in want if k in api}
        out["n_launch"] = api.get("cuLaunchKernel", {}).get("n", 0) + api.get("cudaLaunchKernel", {}).get("n", 0)

# ---- API total + JIT/module-load detection (in the timed region) ----
api_total_ns = 0
jit = {}
for tbl in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
    if has(tbl):
        for nid, c, t in cur.execute(f"SELECT nameId, COUNT(*), SUM(end-start) FROM {tbl} GROUP BY nameId").fetchall():
            nm = S.get(nid, str(nid))
            api_total_ns += (t or 0)
            low = nm.lower()
            if any(s in low for s in ["nvrtc", "modulload", "moduleload", "linkadd", "linkcomplete",
                                       "libraryload", "modulegetfunction", "linkcreate", "jit"]):
                d = jit.setdefault(nm, [0, 0]); d[0]+=c; d[1]+=(t or 0)
out["api_total_ns"] = api_total_ns
out["jit_in_timed"] = {k: {"n": v[0], "ns": v[1]} for k, v in jit.items()}

# ---- timed-range wall + sync detail ----
if has("NVTX_EVENTS"):
    row = cur.execute("SELECT text, MIN(start), MAX(end) FROM NVTX_EVENTS WHERE text LIKE '%_timed'").fetchone()
    if row and row[1] is not None:
        out["timed_wall_ns"] = row[2]-row[1]

# ---- NVTX range projection (kernel busy ns per range name) ----
# join kernels to enclosing NVTX pushpop ranges by time on same... simpler: use NVTX_EVENTS
if has("NVTX_EVENTS") and has("CUPTI_ACTIVITY_KIND_KERNEL"):
    # ranges of interest: our ak.* and CCCL:* and q*_timed
    nvtx = cur.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL").fetchall()
    kerns = cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    agg = {}
    for text, rs, re in nvtx:
        if not (text.startswith("ak.") or text.startswith("CCCL:") or text.endswith("_timed")):
            continue
        busy = 0; n = 0
        for ks, ke in kerns:
            if ks >= rs and ke <= re:
                busy += (ke - ks); n += 1
        a = agg.setdefault(text, [0, 0, 0])
        a[0] += busy; a[1] += n; a[2] += 1
    out["nvtx_kern"] = {k: {"busy_ns": v[0], "n_kern": v[1], "instances": v[2]} for k, v in
                        sorted(agg.items(), key=lambda x: -x[1][0])}

print(json.dumps(out))
