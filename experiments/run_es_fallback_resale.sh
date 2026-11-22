#!/usr/bin/env bash
# Face-value fallback (naive_eps, 280 jobs) and resale value (s_salv, 352 jobs), then the tables of Sections 5.4 and 5.5.
# Each variant: harness -> recorded solver rules (p1d_repair) -> analysis; then experiments/es_fallback_resale.py.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
mkdir -p experiments/r2_runs
for v in naive_eps s_salv; do
  "$PY" experiments/harness.py --kind p1d --variant $v --run es_$v --workers 16 --time-limit 1800 \
      > experiments/r2_runs/es_$v.log 2>&1
  echo "$v run exit $?"
  "$PY" experiments/p1d_repair.py --run es_$v > experiments/r2_runs/es_${v}_repair.log 2>&1
  echo "$v repair exit $?"
  "$PY" experiments/analyze_p1d.py --variant $v --run es_$v > experiments/r2_runs/es_${v}_analyze.log 2>&1
  echo "$v analyze exit $?"
done
"$PY" experiments/es_fallback_resale.py > experiments/r2_runs/es_fallback_resale.log 2>&1
echo "tables exit $?"
