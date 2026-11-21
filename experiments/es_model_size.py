"""Model size and solve times for the Energy Systems paper's computation paragraph (no optimization is run).

Builds (does not solve) the no-forecast model and the three-copy forecast-aware model for each configuration and reads
their column, row and integer counts from HiGHS; reads the stored solve times of the main run
(experiments/r2_runs/p1c_main/items/*.timing.json). Writes data/es_compute.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import highspy  # noqa: E402

import p1c_core as core  # noqa: E402
from model.bridge import Model  # noqa: E402
from model.signal import SignalModel  # noqa: E402


def size(h):
    lp = h.getLp()
    integ = lp.integrality_
    n_int = sum(1 for v in integ if v != highspy.HighsVarType.kContinuous) if integ else 0
    return dict(columns=int(h.getNumCol()), rows=int(h.getNumRow()), integer=n_int, nonzeros=int(h.getNumNz()))


def main():
    out = dict(model_size={}, solve_seconds={})
    for cfg in core.CONFIGS:
        P = core.params(cfg)
        m = Model(P, time_limit=1.0)
        s = SignalModel(P, core.NCAT, core.TAU, time_limit=1.0)
        out["model_size"][cfg] = dict(no_forecast=size(m.h), forecast_aware=size(s.h))
    run = ROOT / "experiments" / "r2_runs" / "p1c_main"
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    kinds = {jid: spec["kind"] for jid, spec in plan["jobs"].items()}
    secs = {}
    for f in (run / "items").glob("J_*.timing.json"):
        jid = f.name[2:-len(".timing.json")]
        secs.setdefault(kinds.get(jid, "?"), []).append(json.loads(f.read_text())["seconds"])
    for k, v in secs.items():
        v.sort()
        out["solve_seconds"][k] = dict(n=len(v), median=statistics.median(v), max=max(v), total_hours=sum(v) / 3600)
    (ROOT / "data" / "es_compute.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
