"""R2 analysis, exactly as pre-registered in data/r2_preregistration.md sections 4-6. Written before any sweep result
was seen.

python experiments/analyze_r2.py --run r2_main
Outputs: data/r2_results.csv (one row per item x method comparison), data/r2_summary.json, stdout summary.
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
from experiments.r2_core import load_centres  # noqa: E402

ALPHA, K, MIX = 30.0, 50, 0.05
SEEDS = {"A": 20260922, "B": 20260923}
SHIFTS = (-2, 2, 4)


def test_family_T1(phat, seed):
    n = len(phat)
    base = (1 - MIX) * phat + MIX * np.ones(n) / n
    rng = np.random.default_rng(seed)
    return rng.dirichlet(ALPHA * base, size=K)


def shifted(phat, s):
    n = len(phat)
    out = np.zeros(n)
    for i, p in enumerate(phat):
        out[min(max(i + s, 0), n - 1)] += p
    return out


def rel_stats(Eb, Ec):
    """Pre-registered ratio of means, plus per-draw relative-difference percentiles and the share of positive draws."""
    d = (Eb - Ec) / Ec
    return dict(ratio_of_means=float((Eb.mean() - Ec.mean()) / Ec.mean()),
                p025=float(np.percentile(d, 2.5)), p975=float(np.percentile(d, 97.5)),
                share_pos=float(np.mean(Eb > Ec)), E_base=float(Eb.mean()), E_cdro=float(Ec.mean()))


def item_admissible(it):
    """An item enters the decision rules only if every method solved (ok) AND its lexicographic
    tie-break completed (tiebreak_ok is not False; for rolling B1 this covers every node solve). Otherwise the item is
    excluded and reported as flagged, never silently compared."""
    M = it["methods"]
    failed = [k for k in M if not M[k].get("ok", False)]
    flagged = [k for k in M if M[k].get("ok", False) and M[k].get("tiebreak_ok") is False]
    return (not failed and not flagged), failed, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r2_main")
    a = ap.parse_args()
    run = ROOT / "experiments" / "r2_runs" / a.run / "items"
    cen = load_centres()
    tests = {z: test_family_T1(cen[z], SEEDS[z]) for z in ("A", "B")}
    rows, bad = [], []
    for f in sorted(run.glob("*.json")):
        if f.name.endswith(".timing.json"):
            continue
        it = json.loads(f.read_text(encoding="utf-8"))
        z = it["ctx"]
        M = it["methods"]
        admissible, failed, flagged = item_admissible(it)
        if not admissible:
            bad.append(dict(item=f.name, failed=failed, tiebreak_flagged=flagged))
            continue
        C = {k: np.array(M[k]["C"]) for k in M}
        E = {k: tests[z] @ C[k] for k in C}
        realistic = bool(it["residence_on"]) and it["delay_mult"] in (1.0, 3.0)
        base = dict(item=f.stem, ctx=z, p_R=it["p_R"], cap_R=it["cap_R"], delay_mult=it["delay_mult"],
                    residence_on=it["residence_on"], eps_mult=it["eps_mult"], realistic=realistic)
        for bname, metric in (("B1", "VSS"), ("B4", "VCI"), ("CSAA", "vs_CSAA"), ("B3", "vs_B3"), ("B7", "vs_B7")):
            if bname in E and "CDRO" in E:
                r = dict(base, metric=metric, baseline=bname, **rel_stats(E[bname], E["CDRO"]))
                for s in SHIFTS:                    # T2: systematic misspecification (reported, not a rule)
                    ps = shifted(cen[z], s)
                    r[f"T2_shift{s:+d}"] = float((ps @ C[bname] - ps @ C["CDRO"]) / (ps @ C["CDRO"]))
                rows.append(r)
    out_csv = ROOT / "data" / "r2_results.csv"
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def rule(metric, thr):
        cand = [r for r in rows if r["metric"] == metric and r["realistic"] and r["eps_mult"] == 1.0]
        passing = [r for r in cand if r["ratio_of_means"] >= thr]
        best = max(cand, key=lambda r: r["ratio_of_means"]) if cand else None
        return dict(threshold=thr, n_realistic_points=len(cand), n_passing=len(passing),
                    passes=len(passing) > 0,
                    best=None if best is None else {k: best[k] for k in ("item", "ratio_of_means", "p025", "p975",
                                                                          "share_pos")},
                    by_ctx={z: dict(n=len([r for r in cand if r["ctx"] == z]),
                                    n_pass=len([r for r in passing if r["ctx"] == z]),
                                    max=max([r["ratio_of_means"] for r in cand if r["ctx"] == z], default=None),
                                    median=float(np.median([r["ratio_of_means"] for r in cand if r["ctx"] == z]))
                                    if any(r["ctx"] == z for r in cand) else None)
                            for z in ("A", "B")})

    summary = dict(run=a.run, n_rows=len(rows), failed_items=bad,
                   VSS=rule("VSS", 0.02), VCI=rule("VCI", 0.01),
                   secondary={m: rule(m, 0.0) for m in ("vs_CSAA", "vs_B3", "vs_B7")},
                   radius_sensitivity=[{k: r[k] for k in ("item", "metric", "ratio_of_means")}
                                       for r in rows if r["eps_mult"] != 1.0])
    (ROOT / "data" / "r2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_rows", "failed_items", "VSS", "VCI")}, indent=1))
    print("secondary (share of realistic points where CDRO beats the baseline is in r2_results.csv):")
    for m, s in summary["secondary"].items():
        print(f"  {m}: points with CDRO better = {s['n_passing']}/{s['n_realistic_points']}; by ctx {s['by_ctx']}")


if __name__ == "__main__":
    main()
