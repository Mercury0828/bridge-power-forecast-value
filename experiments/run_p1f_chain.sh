#!/usr/bin/env bash
# Phase 1f run chain (docs/phase1f_plan.md v1.2; D-026).
#   bash experiments/run_p1f_chain.sh > experiments/r2_runs/p1f_chain.log 2>&1
# Each variant: harness (10 workers; host is shared) -> recorded solver rules (p1d_repair) -> analysis.
# Then the E0-ext TPIT build and evaluation, and the tables.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
VARIANTS="naive tau2 tau6 nocchp s_disc s_gas s_diesel s_grid s_capex s_life s_rent s_eff s_co2 size075 size300 blocks4"
for v in $VARIANTS; do
  $PY experiments/harness.py --kind p1d --variant $v --run p1f_$v --workers 10 --time-limit 1800 \
      > experiments/r2_runs/p1f_$v.log 2>&1
  echo "$v run exit $?"
  $PY experiments/p1d_repair.py --run p1f_$v > experiments/r2_runs/p1f_${v}_repair.log 2>&1
  echo "$v repair exit $?"
  $PY experiments/analyze_p1d.py --variant $v --run p1f_$v > experiments/r2_runs/p1f_${v}_analyze.log 2>&1
  echo "$v analyze exit $?"
done
$PY experiments/p1e_tpit.py --extended > experiments/r2_runs/p1f_tpit_ext.log 2>&1
echo "tpit_ext exit $?"
$PY experiments/p1e_eval.py --ext > experiments/r2_runs/p1f_e0_ext_eval.log 2>&1
echo "e0_ext_eval exit $?"
$PY experiments/make_p1f_report.py > experiments/r2_runs/p1f_report.log 2>&1
echo "report exit $?"
echo "CHAIN DONE"
