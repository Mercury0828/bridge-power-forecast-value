#!/usr/bin/env bash
# Phase 1g run chain (docs/phase1g_plan.md; D-030).
#   bash experiments/run_p1g_chain.sh > experiments/r2_runs/p1g_chain.log 2>&1
# Each variant: harness (10 workers; the host is shared) -> recorded solver rules (p1d_repair: the first call applies
# D-022a, a second call the pure-LP escalation to jobs still unusable) -> analysis; then the Phase-1g report.
# The 2026-09-25 run called p1d_repair once per variant; the escalation was then run by hand on p1g_carry and
# p1g_gasrent_25 (logs p1g_<v>_repair2.log) and both were re-analyzed before the report was regenerated.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
VARIANTS="carry gasrent_25 gasrent_35 gasrent_50"
for v in $VARIANTS; do
  $PY experiments/harness.py --kind p1d --variant $v --run p1g_$v --workers 10 --time-limit 1800 \
      > experiments/r2_runs/p1g_$v.log 2>&1
  echo "$v run exit $?"
  $PY experiments/p1d_repair.py --run p1g_$v > experiments/r2_runs/p1g_${v}_repair.log 2>&1
  echo "$v repair exit $?"
  $PY experiments/p1d_repair.py --run p1g_$v > experiments/r2_runs/p1g_${v}_repair2.log 2>&1
  echo "$v escalation exit $?"
  $PY experiments/analyze_p1d.py --variant $v --run p1g_$v > experiments/r2_runs/p1g_${v}_analyze.log 2>&1
  echo "$v analyze exit $?"
done
$PY experiments/make_p1g_report.py > experiments/r2_runs/p1g_report.log 2>&1
echo "report exit $?"
echo "CHAIN DONE"
