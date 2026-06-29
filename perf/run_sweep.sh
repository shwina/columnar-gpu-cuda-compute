#!/usr/bin/env bash
# Profile every ADL query under nsys for a given env+scale, parse to JSONL.
# Usage: run_sweep.sh <env: ak3|base> <scale: 100k|1M|10M> <out.jsonl> [queries...]
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"

ENV=$1; SCALE=$2; OUT=$3; shift 3
QUERIES=("$@"); [ ${#QUERIES[@]} -eq 0 ] && QUERIES=(1 2 3 4 5 6 7 8)
PARQUET=../data/pq_subset_${SCALE}.parquet
if [ "$ENV" = "ak3" ]; then PY=../.venv-awkward3/bin/python; else PY=../.venv-baseline/bin/python; fi

: > "$OUT"
for q in "${QUERIES[@]}"; do
  tag="q${q}_${ENV}_${SCALE}"
  echo ">>> profiling $tag"
  rm -f nsys/${tag}.nsys-rep nsys/${tag}.sqlite
  # baseline has no NVTX wrappers needed; ak3 uses them. Disable wrap for base to avoid import issues (same harness works for both).
  rj=$(nsys profile --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-memory-usage=true --force-overwrite=true \
        -o nsys/${tag} \
        $PY prof_query.py $q $PARQUET 2 2>/dev/null | grep RESULT_JSON | sed 's/RESULT_JSON //')
  if [ -z "$rj" ]; then
    echo "{\"tag\":\"$tag\",\"q\":$q,\"env\":\"$ENV\",\"scale\":\"$SCALE\",\"ok\":false}" >> "$OUT"
    echo "    FAILED/crashed"; continue
  fi
  # export sqlite + parse
  nsys export --type sqlite --force-overwrite=true -o nsys/${tag}.sqlite nsys/${tag}.nsys-rep >/dev/null 2>&1
  parse=$(../.venv-awkward3/bin/python parse_nsys.py nsys/${tag}.sqlite 2>/dev/null)
  # merge: timing + parse + meta
  ../.venv-awkward3/bin/python -c "
import json,sys
rj=json.loads('''$rj'''); pa=json.loads('''$parse''')
rec={'tag':'$tag','q':$q,'env':'$ENV','scale':'$SCALE','ok':True}
rec.update({'comp':rj['comp'],'load':rj['load'],'fill':rj['fill'],'total':rj['total']})
rec.update(pa)
print(json.dumps(rec))
" >> "$OUT"
  echo "    ok"
done
echo "wrote $OUT"
