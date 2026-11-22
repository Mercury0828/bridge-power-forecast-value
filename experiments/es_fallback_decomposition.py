"""Decomposition of the gain from an error allowance in the face-value planner (Section 5.4 of the paper; Online
Resource 1, Section S7). It was made after the main results, reads stored plans and solves nothing.

    python experiments/es_fallback_decomposition.py    -> data/es_fallback_decomposition.json and .md

Needs the runs p1f_naive (runs.zip) and es_naive_eps (runs_es.zip) unzipped into experiments/.

For each report and announced category, the connection histories are split by the face-value planner's law:
- early miss: the connection comes before the first quarter of positive probability;
- in category: within the quarters of positive probability;
- late miss: after the last of them, when the category has expired.
"Same earlier decisions": every stored pre-connection decision taken at a node before expiry agrees between the two
plans. These decisions are the orders, the two rental series (the rental for quarter q is decided at node q - 1), the
filings and the non-road regime, compared with a tolerance in MW. The share is the gain on late misses with the same
earlier decisions over the whole gain. The parts add up to the difference of the two reported values, which is asserted.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import analyze_p1d as ap  # noqa: E402

BASES = ("A", "A+staged")
WEIGHTS = {"01": 0.01, "05": 0.05}
FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")
LEVELS = ("low", "high")
TOLS = (0.01, 0.1, 1.0)
TOL = 0.1


def table(variant):
    p = ROOT / "data" / "p1d" / f"{variant}_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))["table"]


def same_before(a, b, stop, tol):
    """True if the decisions at nodes k < stop of two stored plan copies agree."""
    if a["z"] != b["z"]:
        return False
    if {k for k in a["filings"] if k < stop} != {k for k in b["filings"] if k < stop}:
        return False
    series = [(a["orders"][j], b["orders"][j]) for j in a["orders"]] + [(a[k], b[k]) for k in ("RST", "RNR")]
    return all(np.abs(np.subtract(x[:stop], y[:stop])).max(initial=0.0) <= tol for x, y in series)


def decompose():
    planF, resF = ap.load("p1f_naive")
    planE, resE = ap.load("es_naive_eps")
    cellE = {(c["cfg"], c["level"], c["r"]): c for c in planE["cells"]}
    gain, pre_same, weight = defaultdict(float), defaultdict(float), defaultdict(float)
    for c in planF["cells"]:
        if c["base"] not in BASES:
            continue
        p, tau = c["p"] / len(LEVELS), ap._tau(c)
        jobF = c["jobs"]["sig"]
        assert ap.usable(planF["jobs"][jobF], resF.get(jobF))
        f = resF[jobF]
        support = np.asarray(planF["jobs"][jobF]["law"]["post"], dtype=float)
        laws = {fam: ap._law(c, fam, 0) for fam in FAMS}
        for tag in WEIGHTS:
            jobE = cellE[f"{c['base']}|naive_eps{tag}", c["level"], c["r"]]["jobs"]["sig"]
            assert ap.usable(planE["jobs"][jobE], resE.get(jobE))
            g = resE[jobE]
            d = np.subtract(f["costs"], g["costs"])
            key = (c["base"], tag)
            weight[key] += p
            pre_same[key] += p * same_before(f["detail"][0], g["detail"][0], tau, TOL)
            for fam in FAMS:
                gain[key, fam, "before_estimate"] += p * float(np.dot(laws[fam]["pre"][:tau], d[0][:tau]))
            for y in range(len(support)):
                live = np.nonzero(support[y] > 0)[0]
                live = live[live >= tau]
                if len(live):
                    first, last = int(live[0]), int(live[-1])           # index i is connection quarter i + 1
                else:
                    first, last = len(support[y]), len(support[y])      # never announced under this law
                i = np.arange(len(support[y]))
                parts = dict(early_miss=(i >= tau) & (i < first), in_category=(i >= first) & (i <= last),
                             late_miss=i > last)
                same = {t: same_before(f["detail"][y], g["detail"][y], last + 1, t) for t in TOLS}
                for fam in FAMS:
                    v = np.asarray(laws[fam]["post"][y]) * d[y]
                    for part, mask in parts.items():
                        x = p * float(v[mask].sum())
                        gain[key, fam, part] += x
                        if part == "late_miss":
                            for t in TOLS:
                                if same[t]:
                                    gain[key, fam, t] += x
    sumF, sumE = table("naive"), table("naive_eps")
    rows = []
    for base in BASES:
        for tag, w in WEIGHTS.items():
            key = (base, tag)
            for fam in FAMS:
                part = {k: gain[key, fam, k] for k in ("before_estimate", "early_miss", "in_category", "late_miss")}
                total = sum(part.values())
                vf = sumF[f"{base}|naive|{fam}|0"]["mean"]
                ve = sumE[f"{base}|naive_eps{tag}|{fam}|0"]["mean"]
                # the no-forecast plans of the two runs are the same solves up to the solver tolerance
                assert abs(total - (ve - vf)) < 1e-3, (base, tag, fam, total, ve - vf)
                rows.append(dict(config=base, weight=w, family=fam, face_value=vf, gain=total, **part,
                                 late_same_earlier=gain[key, fam, TOL],
                                 share=gain[key, fam, TOL] / total,
                                 share_by_tolerance={str(t): gain[key, fam, t] / total for t in TOLS},
                                 same_before_estimate=pre_same[key] / weight[key]))
    return dict(tolerance_mw=TOL, rows=rows)


def main():
    res = decompose()
    lines = ["# Gain from an error allowance in the face-value planner, by connection history", "",
             "$M, report-weighted, mean of the two dispersion levels. Made after the main results from stored plans.", "",
             "| configuration | weight | family | face value | gain | before estimate | early miss | in category "
             "| late miss | late miss, same earlier decisions | share | share at 0.01 / 1 MW "
             "| same decisions before the estimate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        s = r["share_by_tolerance"]
        lines.append(
            f"| {r['config']} | {r['weight']} | {r['family']} | {r['face_value']:.2f} | {r['gain']:.3f} "
            f"| {r['before_estimate']:.3f} | {r['early_miss']:.3f} | {r['in_category']:.3f} | {r['late_miss']:.3f} "
            f"| {r['late_same_earlier']:.3f} | {r['share']:.4f} | {s['0.01']:.4f} / {s['1.0']:.4f} "
            f"| {r['same_before_estimate']:.5f} |")
    out = ROOT / "data" / "es_fallback_decomposition"
    out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
