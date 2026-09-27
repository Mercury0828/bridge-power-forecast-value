"""Phase 1d (exploratory): the D-022a solver rule applied to jobs whose LP tie-break failed.

    python experiments/p1d_repair.py --run p1d_stage

Phase 1d makes no claims. Its runs adopt the rule verified for Phase 1c in D-022a
(`data/p1d/d022a_verification.json`: the exported second-stage model is byte-identical). A job whose registered LP
tie-break fails (`tiebreak_ok = False`), or whose solve ends with a non-optimal solver status after a failed tie-break
(one refurb job: kUnboundedOrInfeasible from the restore step, optimal with presolve off), is re-solved once, with HiGHS
presolve off in the second stage only.

The replaced attempt is kept in `experiments/r2_runs/<run>.attempt1/`, and the new item records the repair. Jobs that
fail for another reason are left as they are and reported.

Escalation (Phase 1e; exploratory runs only). If a job is still unusable after the D-022a rule, it is re-solved once more
with the same second-stage problem solved as a pure LP with presolve off (tiebreak_mode "lp_pure", presolve off). The
previous attempt is kept in `<run>.attempt2/`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments import p1d_core as p1d  # noqa: E402
from experiments.analyze_p1d import usable  # noqa: E402
from experiments.harness import atomic_write_json, code_fingerprint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d = ROOT / "experiments" / "r2_runs" / a.run
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    for key, spec in sorted(plan["jobs"].items()):
        item = d / "items" / f"J_{key}.json"
        prev = json.loads(item.read_text(encoding="utf-8")) if item.exists() else None
        if usable(spec, prev):
            continue
        solver_status = prev is not None and prev.get("ok") is False and prev.get("status") and not prev.get("error")
        if prev is None or not (prev.get("tiebreak_ok") is False or solver_status) or spec["kind"] not in ("nosig", "sig"):
            print(f"{key}: unusable for another reason; not repaired "
                  f"({None if prev is None else {k: prev.get(k) for k in ('ok', 'error', 'status')}})")
            continue
        escalate = "repair" in prev
        keep_dir = d.parent / f"{a.run}.attempt{2 if escalate else 1}"
        keep_dir.mkdir(exist_ok=True)
        for f in (item, d / "items" / f"J_{key}.timing.json"):
            if f.exists() and not (keep_dir / f.name).exists():
                shutil.copy(f, keep_dir / f.name)
        t0 = time.time()
        P = p1d.params_for(spec["cfg"], spec["over"])
        mode, rule = (("lp_pure", "escalation: the same second-stage LP solved as a pure LP with presolve off")
                      if escalate else ("lp", "D-022a solver rule (presolve off in the second stage)"))
        try:
            res = core._run_job(spec, 1800.0, tiebreak_mode=mode, tiebreak_presolve=False, P=P)
        except Exception as e:                                            # noqa: BLE001
            res = dict(ok=False, error=f"{type(e).__name__}: {e}")
        res["spec_key"] = p1d.job_key(spec)
        res["repair"] = dict(rule=rule, attempt=3 if escalate else 2, fingerprint=code_fingerprint("p1d"))
        secs = round(time.time() - t0, 2)
        atomic_write_json(item, res)
        atomic_write_json(d / "items" / f"J_{key}.timing.json", dict(seconds=secs))
        print(f"{key} {spec['name']} {spec['kind']}: tiebreak_ok={res.get('tiebreak_ok')} ok={res.get('ok')} "
              f"usable={usable(spec, res)} ({secs}s)")


if __name__ == "__main__":
    main()
