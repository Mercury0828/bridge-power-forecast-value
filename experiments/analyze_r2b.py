"""R2b analysis, exactly as pre-registered in data/r2b_preregistration.md (written before any R2b result was seen).

python experiments/analyze_r2b.py --run r2b_loro  ->  data/r2b_results.csv, data/r2b_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.r2_core import load_components, COMPONENTS  # noqa: E402
from experiments.analyze_r2 import test_family_T1  # noqa: E402

SEEDS = {"A": 20260924, "B": 20260925}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r2b_loro")
    a = ap.parse_args()
    comps = load_components()
    run = ROOT / "experiments" / "r2_runs" / a.run / "items"
    rows, flagged = [], []
    for f in sorted(run.glob("L_*.json")):
        if f.name.endswith(".timing.json"):
            continue
        it = json.loads(f.read_text(encoding="utf-8"))
        z = it["ctx"]
        r1, r2 = COMPONENTS[z]
        bad = [(r, k) for r in (r1, r2) for k, v in it["train"][r]["methods"].items()
               if not v.get("ok", False) or v.get("tiebreak_ok") is False]
        if bad:
            flagged.append(dict(item=f.name, bad=bad))
            continue
        per_dir = {}
        for train, test in ((r1, r2), (r2, r1)):
            M = it["train"][train]["methods"]
            C = {k: np.array(v["C"]) for k, v in M.items()}
            ps = comps[test]
            E = {k: float(ps @ C[k]) for k in C}                                   # exact held-out test (primary)
            fam = test_family_T1(ps, SEEDS[z])                                       # reported only
            Ef = {k: fam @ C[k] for k in C}
            per_dir[(train, test)] = dict(
                ROB=(E["CSAA"] - E["CDRO"]) / E["CDRO"], VSSL=(E["B1"] - E["CDRO"]) / E["CDRO"],
                BOX=(E["B5"] - E["CDRO"]) / E["CDRO"],
                ROB_fam=float((Ef["CSAA"].mean() - Ef["CDRO"].mean()) / Ef["CDRO"].mean()),
                ROB_fam_sharepos=float(np.mean(Ef["CSAA"] > Ef["CDRO"])))
        d1, d2 = per_dir[(r1, r2)], per_dir[(r2, r1)]
        rows.append(dict(item=f.stem, ctx=z, p_R=it["p_R"], cap_R=it["cap_R"], delay_mult=it["delay_mult"],
                         ROB=(d1["ROB"] + d2["ROB"]) / 2, VSSL=(d1["VSSL"] + d2["VSSL"]) / 2,
                         BOX=(d1["BOX"] + d2["BOX"]) / 2,
                         ROB_dir1=d1["ROB"], ROB_dir2=d2["ROB"], VSSL_dir1=d1["VSSL"], VSSL_dir2=d2["VSSL"],
                         BOX_dir1=d1["BOX"], BOX_dir2=d2["BOX"],
                         ROB_fam=(d1["ROB_fam"] + d2["ROB_fam"]) / 2,
                         dir1=f"{r1}->{r2}", dir2=f"{r2}->{r1}"))
    with open(ROOT / "data" / "r2b_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for z in ("A", "B"):
        rz = [r for r in rows if r["ctx"] == z]
        by[z] = {m: dict(max=max(r[m] for r in rz), median=float(np.median([r[m] for r in rz])),
                         min=min(r[m] for r in rz), n_ge_1pct=sum(r[m] >= 0.01 for r in rz), n=len(rz))
                 for m in ("ROB", "VSSL", "BOX", "ROB_dir1", "ROB_dir2")}
    passes = any(r["ROB"] >= 0.01 for r in rows)
    summary = dict(run=a.run, n_items=len(rows), flagged=flagged, repair_rule="ROB >= +1% at >= 1 realistic point",
                   repair_succeeds=passes, by_ctx=by)
    (ROOT / "data" / "r2b_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
