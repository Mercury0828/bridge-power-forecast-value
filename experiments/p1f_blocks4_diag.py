"""Phase 1f E-R diagnostic (not a repair): the three A+staged no-signal jobs of p1f_blocks4 that stay unusable after the
recorded solver rules (LP tie-break -> D-022a -> escalation), re-solved with the tie-break in its original D-008 form
("milp": the second stage re-optimizes every decision, as in R2 and Phase 1a single-copy policies).

    python experiments/p1f_blocks4_diag.py [--sensitivity-only]   -> data/p1f/blocks4_milp_diag.json

The stored items in experiments/r2_runs/p1f_blocks4 are not touched; whether this result may replace them is an owner
decision. The output records, per job, the usability checks of `analyze_p1d.usable` and the expected cost under the
job's own planner marginal, next to the Phase-1c counterpart for reference.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments import p1d_core as p1d  # noqa: E402
from experiments.analyze_p1d import usable  # noqa: E402

RUN = ROOT / "experiments" / "r2_runs" / "p1f_blocks4"
JOBS = ("2fc3b2ec36c8587cf967", "83e478ae47aad86df6f6", "959a7176cdcb3d1e8d47")


def main():
    p = ROOT / "data" / "p1f" / "blocks4_milp_diag.json"
    if "--sensitivity-only" in sys.argv:                       # reuse the stored diagnostic solves
        out = json.loads(p.read_text(encoding="utf-8"))
        out["_sensitivity"] = sensitivity(out)
        p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        print(f"written {p.relative_to(ROOT)}")
        return
    plan = json.loads((RUN / "plan.json").read_text(encoding="utf-8"))
    out = {}
    for k in JOBS:
        spec = plan["jobs"][k]
        P = p1d.params_for(spec["cfg"], spec["over"])
        res = core._run_job(spec, 1800.0, tiebreak_mode="milp", P=P)
        ok = usable(spec, res)
        rec = dict(tiebreak_mode="milp", usable=ok, ok=res.get("ok"), tiebreak_ok=res.get("tiebreak_ok"),
                   status=res.get("status"), primary_gap=res.get("primary_gap"),
                   bookkeeping_maxabs=res.get("bookkeeping_maxabs"), backup_shortfall=res.get("backup_shortfall"))
        if res.get("C") is not None:
            rec["C"] = res["C"]
            rec["J_planner"] = float(np.dot(spec["centre"], res["C"][:len(spec["centre"])]))
        out[k] = rec
        print(k, {x: rec[x] for x in ("usable", "ok", "tiebreak_ok", "status", "bookkeeping_maxabs",
                                       "backup_shortfall")}, flush=True)
    out["_sensitivity"] = sensitivity(out)
    p = ROOT / "data" / "p1f" / "blocks4_milp_diag.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"written {p.relative_to(ROOT)}")


def sensitivity(diag):
    """In memory only: E[Δ_info] of blocks4 on all cells, with the three unusable no-signal items replaced by the
    diagnostic's primary-optimal policies (valid solutions without the canonical tie-break), against Phase 1c on the
    same cells. The stored items and summaries are unchanged."""
    from experiments import analyze_p1d as an
    plan, res = an.load("p1f_blocks4")
    for k, rec in diag.items():
        if k.startswith("_") or rec.get("C") is None or not rec.get("ok"):
            continue
        res[k] = dict(res[k], C=rec["C"], ok=True, tiebreak_ok=None, bookkeeping_maxabs=rec["bookkeeping_maxabs"],
                      backup_shortfall=rec["backup_shortfall"])
    orig_load = an.load
    an.load = lambda run: (plan, res)
    try:
        out = an.analyze("blocks4", "p1f_blocks4")
        S = an.summarize("blocks4", out)
    finally:
        an.load = orig_load
    rows = {}
    for c in ("A+staged", "A", "B", "B+staged"):
        for fam in ("sym1", "sym2"):
            k = f"{c}|blocks4|{fam}|0"
            vs = S["vs_phase1c"][k]
            rows[k] = dict(n=vs["change"]["n"], phase1c=vs["d_info_main"]["mean"],
                           blocks4=vs["d_info_main"]["mean"] + vs["change"]["mean"], change=vs["change"]["mean"])
            print(k, {x: round(y, 3) if isinstance(y, float) else y for x, y in rows[k].items()}, flush=True)
    return dict(n_usable=out["n_usable"], complete=out["complete"], rows=rows,
                note="primary-optimal no-signal policies substituted for the three items; not of record")


if __name__ == "__main__":
    main()
