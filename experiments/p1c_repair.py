"""Owner decision D-022 (2026-09-24, amended the same day; both taken blind to results): re-solve the Phase-1c job that
stayed unusable after the pre-registered §8 remedy, with the registered LP tie-break and HiGHS presolve off in its second
stage.

    python experiments/p1c_repair.py --run p1c_main

- The job: `39cf540b499e5e769320`, the no-signal policy of A+staged for report outcome r = 4 (mean delay 130 days), used
  by one cell (high dispersion, p = 1.885e-3).
- The failure: after the integers are fixed, HiGHS keeps their integrality and takes the MIP path. The tie-break solve
  returns kSolveError after 0 iterations. It is deterministic, so the 7200 s remedy reproduced it.
- D-022 first tried `tiebreak_mode="lp_pure"`: the same second-stage problem solved as a pure LP (attempt 3). The
  tie-break solved, but its solution was numerically fragile, and the fixed-spine polish was infeasible, so the job
  was still unusable. Presolve is the common cause.
- D-022 as amended: the registered `tiebreak_mode="lp"`, exactly as for the other 375 jobs, with
  `tiebreak_presolve=False` (presolve off in the second stage only; attempt 4). The model, the rule and the inputs are
  unchanged, and no other job is touched.
- Every replaced attempt is kept: `experiments/r2_runs/<run>.attempt1/`, `.attempt2/` and `.attempt3/`. The new item
  records the repair.
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
from experiments.analyze_p1c import usable  # noqa: E402
from experiments.harness import atomic_write_json, code_fingerprint  # noqa: E402

REPAIR = {"p1c_main": ["39cf540b499e5e769320"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1c_main")
    a = ap.parse_args()
    d = ROOT / "experiments" / "r2_runs" / a.run
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    for key in REPAIR[a.run]:
        spec = plan["jobs"][key]
        item = d / "items" / f"J_{key}.json"
        prev = json.loads(item.read_text(encoding="utf-8"))
        if usable(spec, prev):
            print(f"{key}: already usable; nothing to repair")
            continue
        n_prev = 1 + len(list(d.parent.glob(f"{a.run}.attempt*")))    # attempts already kept + the current item
        keep = d.parent / f"{a.run}.attempt{n_prev}"
        keep.mkdir(exist_ok=True)
        for f in (item, d / "items" / f"J_{key}.timing.json"):
            if not (keep / f.name).exists():
                shutil.copy(f, keep / f.name)
        t0 = time.time()
        res = core.run_job(spec, time_limit=1800.0, tiebreak_mode="lp", tiebreak_presolve=False)
        secs = round(time.time() - t0, 2)
        res["repair"] = dict(decision="D-022 (amended)", attempt=n_prev + 1, tiebreak_mode="lp",
                             tiebreak_presolve="off", fingerprint=code_fingerprint("p1c"))
        atomic_write_json(item, res)
        atomic_write_json(d / "items" / f"J_{key}.timing.json", dict(seconds=secs))
        print(f"{key}: ok={res.get('ok')} tiebreak_ok={res.get('tiebreak_ok')} "
              f"bookkeeping={res.get('bookkeeping_maxabs')} shortfall={res.get('backup_shortfall')} "
              f"usable={usable(spec, res)} ({secs}s)")


if __name__ == "__main__":
    main()
