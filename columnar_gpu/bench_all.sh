#!/usr/bin/env bash
# Warm per-query GPU comparison: awkward3 (cuda.compute) vs baseline (2.8.11 RawKernel).
# Each query runs in its own process (warm-up + timed) for crash isolation.
set -u
cd /home/coder/columnar_gpu_bench/columnar_gpu
export PYTHONPATH=/home/coder/columnar_gpu_bench/columnar_gpu
# CUDA default order is FASTEST_FIRST, which makes device 1 the T400(4GB). Force
# PCI order so CUDA_VISIBLE_DEVICES=1 == nvidia-smi index 1 == RTX 6000 Ada (49GB).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Read path: cudf GPU-direct in BOTH envs (cudf force-installed alongside cuda.compute and
# the 2.8.11 RawKernel backend; see cudf_inject_overrides.txt). Set AK_BENCH_READER=shim
# for the old pyarrow CPU path.
export AK_BENCH_READER="${AK_BENCH_READER:-cudf}"
DATA="${1:-/home/coder/columnar_gpu_bench/data/pq_subset_100k.parquet}"
OUT="${2:-logs/cmp_100k.txt}"
: > "$OUT"

run_env () {  # $1=label  $2=venv
  for q in 1 2 3 4 5 6 7 8; do
    line=$(timeout 900 env CUDA_VISIBLE_DEVICES=1 AK_BENCH_READER="$AK_BENCH_READER" "$2/bin/python" bench_driver.py "$q" "$DATA" "$1" 2>/dev/null | grep RESULT_JSON)
    if [ -z "$line" ]; then
      echo "{\"q\": $q, \"label\": \"$1\", \"ok\": false, \"error\": \"process crashed (no result)\"}" | tee -a "$OUT"
    else
      echo "${line#RESULT_JSON }" | tee -a "$OUT"
    fi
  done
}

echo "### dataset: $DATA (reader=$AK_BENCH_READER)"   | tee -a "$OUT"
echo "### awkward3 (cuda.compute)" | tee -a "$OUT"
run_env awkward3 /home/coder/columnar_gpu_bench/.venv-awkward3
echo "### baseline (2.8.11 RawKernel)" | tee -a "$OUT"
run_env baseline /home/coder/columnar_gpu_bench/.venv-baseline
echo "### DONE" | tee -a "$OUT"
