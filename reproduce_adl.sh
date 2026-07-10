#!/usr/bin/env bash
#
# reproduce_adl.sh - build both ADL environments from scratch and reproduce all
# results (the cmp_*/fused logs + the benchmark_cudf_rerun.png chart).
#
# Two uv venvs differing only in Awkward:
#   .venv-awkward3  : editable awkward (main; awkward3/cuda.compute is merged) + cuda.compute
#   .venv-baseline  : released awkward==2.8.11 (RawKernel)
# cudf is force-installed LAST in both with cudf_inject_overrides.txt, because
# cudf-cu13's numba-cuda pin otherwise fights cuda.compute's cuda-core 1.0.x
# (metadata-only conflict; the override keeps the working stack, adds cudf's C++).
#
# Run from the repo root:  ./reproduce_adl.sh
# Skips: SKIP_BUILD=1 (reuse venvs), SKIP_BENCH=1 (reuse logs), SCALES="100k 1M".
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
CG="$REPO/columnar_gpu"
DATA="$REPO/data"
A3="$REPO/.venv-awkward3"
BL="$REPO/.venv-baseline"
CCCL_SRC="${CCCL_SRC:-/home/coder/cccl/python/cuda_cccl}"     # local editable cuda.compute
GPU="${CUDA_VISIBLE_DEVICES:-0}"                               # caller picks the GPU (nvidia-smi index, PCI order)
SCALES="${SCALES:-100k 1M 10M}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
COMMON=(numba==0.65.1 "numba-cuda[cu13]==0.30.2" cuda-core==1.0.1 cuda-bindings==13.3.1
        cuda-pathfinder==1.5.5 cupy-cuda13x pyarrow pandas matplotlib hist uproot vector pytz
        "coffea @ git+https://github.com/scikit-hep/coffea.git@jitters")
CUDF_INSTALL=(--override cudf_inject_overrides.txt --extra-index-url https://pypi.nvidia.com
              --index-strategy unsafe-best-match "cudf-cu13==26.6.*")

log(){ echo -e "\n=== $* ==="; }

# ---- 0. uv ------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# ---- 1. data ----------------------------------------------------------------
log "data subsets"
mkdir -p "$DATA"
BASE=http://uaf-10.t2.ucsd.edu/~kmohrman/large_files_no_backup/for_ak_gpu/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets
for s in 100k 1M 10M; do
  f="$DATA/pq_subset_$s.parquet"
  [ -f "$f" ] && echo "  have pq_subset_$s.parquet" || { echo "  fetching $s"; wget -q "$BASE/pq_subset_$s.parquet" -P "$DATA" || echo "  WARN: download failed ($s)"; }
done

# ---- 2. awkward source (main; awkward3 / cuda.compute is merged) ------------
# Delete awkward_src first if you want to refresh an existing checkout to main.
if [ ! -d "$REPO/awkward_src" ]; then
  log "cloning awkward (main)"
  git clone --depth 1 https://github.com/scikit-hep/awkward "$REPO/awkward_src"
  # the combinations JIT-cache fix is upstream in main; apply the patch only if
  # main doesn't already have it (skip cleanly otherwise)
  if git -C "$REPO/awkward_src" apply --check "$REPO/patches/awkward3-combinations-cache-fix.patch" 2>/dev/null; then
    git -C "$REPO/awkward_src" apply "$REPO/patches/awkward3-combinations-cache-fix.patch"
    echo "  applied combinations JIT-cache patch"
  else
    echo "  combinations patch does not apply (already in main) - skipping"
  fi
fi
log "regenerating awkward-cpp build artifacts"
( cd "$REPO/awkward_src" && git submodule update --init --recursive )

build_env () {   # $1=venv-path  $2=awkward-mode ("src" | "2.8.11")
  local V="$1" MODE="$2"
  log "build ${V##*/}  (awkward=$MODE)"
  uv venv --python 3.12 "$V"
  export VIRTUAL_ENV="$V"
  uv pip install "${COMMON[@]}"
  if [ "$MODE" = src ]; then
    "$V/bin/python" "$REPO/awkward_src/dev/copy-cpp-headers.py"
    "$V/bin/python" "$REPO/awkward_src/dev/generate-kernel-signatures.py"
    uv pip install -e "$REPO/awkward_src/awkward-cpp"      # compiles C++
    uv pip install -e "$REPO/awkward_src"                  # pure-python awkward (overrides coffea's)
    if [ -d "$CCCL_SRC" ]; then uv pip install -e "$CCCL_SRC"
    else echo "  WARN: $CCCL_SRC missing; falling back to pip cuda-cccl"; uv pip install cuda-cccl; fi
  else
    uv pip install "awkward==2.8.11"
  fi
  log "  force-install cudf LAST (override keeps the numba-cuda/cuda-core stack)"
  uv pip install "${CUDF_INSTALL[@]}"
  unset VIRTUAL_ENV
}

if [ "${SKIP_BUILD:-0}" != 1 ]; then
  log "DELETING existing venvs and rebuilding from scratch"
  rm -rf "$A3" "$BL"
  build_env "$A3" src
  build_env "$BL" 2.8.11
  # sanity: awkward3 backend really dispatches through cuda.compute
  CUDA_VISIBLE_DEVICES="$GPU" "$A3/bin/python" -c \
    "import awkward as ak; g=ak.to_backend(ak.Array([[3.,1.,2.],[],[5.,4.]]),'cuda'); print('sort via cuda.compute ->', ak.sort(g,axis=1).to_list())"
fi

# ---- 5. main q1-q8 comparison per scale ------------------------------------
if [ "${SKIP_BENCH:-0}" != 1 ]; then
  for s in $SCALES; do
    log "bench_all.sh @ $s  (10M baseline q5/q6 ~200s each)"
    CUDA_VISIBLE_DEVICES="$GPU" bash "$CG/bench_all.sh" "$DATA/pq_subset_$s.parquet" "$CG/logs/cmp_$s.txt"
  done

  # ---- 6. fused Q3/Q4/Q7 rewrites -> fused.txt (with a scale field) ---------
  log "fused rewrites (q3c/q4c/q7c) -> logs/fused.txt"
  : > "$CG/logs/fused.txt"
  for s in $SCALES; do
    for qc in 3c 4c 7c; do
      line=$(CUDA_VISIBLE_DEVICES="$GPU" AK_BENCH_READER=cudf PYTHONPATH="$CG" \
             "$A3/bin/python" "$CG/bench_driver.py" "$qc" "$DATA/pq_subset_$s.parquet" awkward3 2>/dev/null | grep RESULT_JSON)
      [ -z "$line" ] && { echo "  WARN: $qc @ $s produced nothing"; continue; }
      "$A3/bin/python" -c "import json,sys; d=json.loads(sys.argv[1][len('RESULT_JSON '):]); d['scale']=sys.argv[2]; print(json.dumps(d))" \
        "$line" "$s" >> "$CG/logs/fused.txt"
      echo "  $qc @ $s ok"
    done
  done
fi

# ---- 7. chart ---------------------------------------------------------------
log "make_speedup_chart.py -> plots/benchmark_cudf_rerun.png"
( cd "$CG" && "$A3/bin/python" make_speedup_chart.py )

log "DONE"
echo "Logs:  $CG/logs/{cmp_100k,cmp_1M,cmp_10M,fused}.txt"
echo "Chart: $CG/plots/benchmark_cudf_rerun.png"
echo "Tables: ../.venv-awkward3/bin/python columnar_gpu/format_results.py columnar_gpu/logs/cmp_<scale>.txt"
echo "Deck:  cd <marp> && COLUMNAR_GPU=$CG DECK_MACHINE=<machine> python3 scripts/integrate_adl.py"
