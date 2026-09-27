"""Phase-1c analysis (data/p1c_preregistration.md v2). Written and committed before any Phase-1c result existed.

    python experiments/analyze_p1c.py --run p1c_main  ->  data/p1c_summary.json, data/p1c_rows.json

1. Usability. no-signal and signal jobs must be ok, with no failed tie-break; bookkeeping ≤ 1e-6; backup shortfall
   ≤ 1e-6. If any no-signal or signal job is unusable after the §8 remedy, the verdict is INCOMPLETE. A failed oracle
   job only removes that diagnostic.
2. Exact evaluation, for every cell (configuration, dispersion level, report outcome with probability p), evaluation
   family f and truth shift s:
   - the true joint law is the truth marginal × the family's category kernel under the cell's (training) thresholds,
     cleaned;
   - both frozen policies are evaluated under it;
   - Δ_info = J_nosig − J_sig.
3. Repeated estimation is handled by exact integration over report outcomes:
   - per (configuration, family, shift, level), E = Σ_outcomes p Δ_info;
   - the integration term err is:
     - A: (1 − covered) × J_BOUND, a deterministic bound for the uncovered report tail;
     - B: 4 × the estimated Monte Carlo SE of Σ p̂ Δ (multinomial p̂ from REPORT_MC cohorts). This is STATISTICAL MC
       inference: a one-sided normal-approximation lower bound with nominal coverage ≈ 1 − 3.2e-5.
   The grid statistic is the equal-weight mean over the two dispersion levels, and err is averaged the same way.
4. Claims (Δ = 0; each separately):
   - H1a, H1b: B, sym1 and sym2;
   - H2a, H2b: A+staged, sym1 and sym2;
   - H3a, H3b: the A staging interaction, Δ_info(A+staged) − Δ_info(A), paired by (level, report outcome).

   A claim PASSES iff  E − err ≥ max($3M, 0.5 % of the grid mean no-signal cost of its configuration), where A+staged
   is used for H3. All thresholds apply to the conservative lower end. A is exact up to the tail bound. B's lower end
   is a Monte Carlo confidence bound with the stated coverage. No inference over training replicates is needed,
   because report outcomes are integrated.
5. Reported without claims: A and B+staged; sym4 and opt2; the uninf control; shifts ±2; the B staging interaction; the
   oracle information value and certified room (Δ = 0), probability-weighted.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from model.signal import joint_law, evaluate, clean_law  # noqa: E402

MIN_ABS, MIN_REL = 3.0, 0.005
CLAIMS = {"H1a": ("B", "sym1"), "H1b": ("B", "sym2"), "H2a": ("A+staged", "sym1"), "H2b": ("A+staged", "sym2"),
          "H3a": ("A:staging", "sym1"), "H3b": ("A:staging", "sym2")}


def load(run):
    d = ROOT / "experiments" / "r2_runs" / run
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    res = {}
    for k in plan["jobs"]:
        f = d / "items" / f"J_{k}.json"
        if f.exists():
            res[k] = json.loads(f.read_text(encoding="utf-8"))
    return plan, res


def usable(spec, r):
    if r is None or not r.get("ok"):
        return False
    if spec["kind"] == "oracle":
        return True
    if r.get("tiebreak_ok") is False:
        return False
    return r.get("bookkeeping_maxabs", 1.0) <= 1e-6 and r.get("backup_shortfall", 1.0) <= 1e-6


def analyze(run):
    plan, res = load(run)
    jobs = plan["jobs"]
    ok = {k: usable(jobs[k], res.get(k)) for k in jobs}
    complete = all(ok[k] for k, s in jobs.items() if s["kind"] in ("nosig", "sig"))
    rows = []
    for cell in plan["cells"]:
        J = cell["jobs"]
        if not (ok[J["nosig"]] and ok[J["sig"]]):
            continue
        C_ns = np.array(res[J["nosig"]]["C"])
        C_sig = np.array(res[J["sig"]]["costs"])
        th = tuple(cell["thresholds"])
        for fam in core.FAMILIES:
            for s in core.SHIFTS:
                pt = core.truth(cell["cfg"], cell["level"], s)
                law = clean_law(joint_law(pt, core.true_kernel(th, fam, pt), core.TAU))
                J_ns = evaluate(np.array([C_ns] * core.NCAT), law, core.TAU)
                J_sg = evaluate(C_sig, law, core.TAU)
                row = dict(cfg=cell["cfg"], level=cell["level"], r=cell["r"], p=cell["p"], covered=cell["covered"],
                           family=fam, shift=s, J_nosig=J_ns, J_sig=J_sg, d_info=J_ns - J_sg)
                if s == 0:
                    ko = J["oracle"][fam]
                    if ok[ko]:
                        J_or = evaluate(np.array(res[ko]["costs"]), law, core.TAU)
                        row.update(J_oracle=J_or, L_oracle=res[ko]["dual_bound"], V_oracle=J_ns - J_or,
                                   room_cert=J_sg - res[ko]["dual_bound"])
                rows.append(row)
    return dict(rows=rows, complete=complete, n_jobs=len(jobs), n_usable=int(sum(ok.values())),
                unusable=[k for k in jobs if not ok[k]])


def _level_stat(items, ctx, J_bound):
    """items: [(p, effect, J_nosig)] for one level. -> E, E[J_nosig], err."""
    p = np.array([i[0] for i in items])
    d = np.array([i[1] for i in items])
    Jn = np.array([i[2] for i in items])
    E = float(p @ d)
    EJ = float(p @ Jn / p.sum())
    if ctx == "A":
        err = (1.0 - float(p.sum())) * J_bound
    else:
        var = (float(p @ d ** 2) - E ** 2) / core.REPORT_MC
        err = 4.0 * float(np.sqrt(max(var, 0.0)))
    return E, EJ, err


def _stat(series, ctx, J_bound=core.J_BOUND):
    """series: {(level, r): (p, effect, J_nosig)} -> grid statistic (equal-weight mean over levels)."""
    by = {}
    for (lv, r), v in series.items():
        by.setdefault(lv, []).append(v)
    per = {lv: _level_stat(v, ctx, J_bound) for lv, v in by.items()}
    E = float(np.mean([x[0] for x in per.values()]))
    EJ = float(np.mean([x[1] for x in per.values()]))
    err = float(np.mean([x[2] for x in per.values()]))
    return dict(mean=E, mean_J_nosig=EJ, rel=E / EJ, err=err, lower=E - err,
                by_level={lv: x[0] for lv, x in per.items()}, n=len(series))


def _series(rows, cfg, fam, shift=0, key="d_info"):
    return {(r["level"], r["r"]): (r["p"], r[key], r["J_nosig"]) for r in rows
            if r["cfg"] == cfg and r["family"] == fam and r["shift"] == shift}


def summarize(out):
    rows = out["rows"]
    S = dict(complete=out["complete"], n_jobs=out["n_jobs"], n_usable=out["n_usable"], unusable=out["unusable"],
             claims={}, table={})
    for cfg in core.CONFIGS:
        for fam in core.FAMILIES:
            for s in core.SHIFTS:
                ser = _series(rows, cfg, fam, s)
                if ser:
                    S["table"][f"{cfg}|{fam}|{s}"] = _stat(ser, core.base_ctx(cfg))
    for base in ("A", "B"):
        for fam in core.FAMILIES:
            a, b = _series(rows, base, fam), _series(rows, f"{base}+staged", fam)
            common = sorted(set(a) & set(b))
            if common:
                ser = {k: (b[k][0], b[k][1] - a[k][1], b[k][2]) for k in common}
                S["table"][f"{base}:staging|{fam}|0"] = _stat(ser, base, J_bound=2 * core.J_BOUND)
    for h, (cfg, fam) in CLAIMS.items():
        st = S["table"].get(f"{cfg}|{fam}|0")
        if st is None or not out["complete"]:
            S["claims"][h] = dict(cfg=cfg, family=fam, verdict="INCOMPLETE", **(st or {}))
            continue
        need = max(MIN_ABS, MIN_REL * st["mean_J_nosig"])
        S["claims"][h] = dict(cfg=cfg, family=fam, threshold=need, **st,
                              verdict=("PASS" if st["lower"] >= need else "FAIL"))
    diag = {}
    for cfg in core.CONFIGS:
        for fam in core.FAMILIES:
            R0 = [r for r in rows if r["cfg"] == cfg and r["family"] == fam and r["shift"] == 0 and "V_oracle" in r]
            if R0:
                by = {}
                for r in R0:
                    by.setdefault(r["level"], []).append(r)
                w = lambda k: float(np.mean([sum(r["p"] * r[k] for r in v) / sum(r["p"] for r in v)   # noqa: E731
                                             for v in by.values()]))
                diag[f"{cfg}|{fam}"] = dict(V_oracle=w("V_oracle"), room_cert_mean=w("room_cert"),
                                            room_cert_max=float(max(r["room_cert"] for r in R0)), n=len(R0))
    S["oracle_diagnostics"] = diag
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1c_main")
    a = ap.parse_args()
    out = analyze(a.run)
    S = summarize(out)
    (ROOT / "data" / "p1c_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (ROOT / "data" / "p1c_rows.json").write_text(json.dumps(out["rows"], default=float), encoding="utf-8")
    print(f"complete {S['complete']} | jobs {S['n_jobs']} usable {S['n_usable']}")
    for h, c in S["claims"].items():
        if "mean" not in c:
            print(f"  {h}: {c['verdict']}")
            continue
        print(f"  {h} {c['cfg']:10s} {c['family']}: E[Δ_info] {c['mean']:+8.3f} $M ({100 * c['rel']:+.3f} %), "
              f"lower {c['lower']:+8.3f} vs threshold {c.get('threshold', float('nan')):.3f} | by level "
              f"{ {k: round(v, 3) for k, v in c['by_level'].items()} } -> {c['verdict']}")


if __name__ == "__main__":
    main()
