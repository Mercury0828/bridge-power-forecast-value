"""Phase-1a analysis (data/p1a_preregistration.md §7-§9). Written and committed before any Phase-1a result existed.

    python experiments/analyze_p1a.py --run p1a_main [--out data/p1a_summary_main.json]

Steps:
  1. Admissibility: a job is usable only if the solve and the polish are ok, its tie-break did not fail, bookkeeping
     matches the independent recomputation (<= 1e-6), the backup requirement holds (<= 1e-6), and (DRO) the dual
     equals the primal worst case (<= 1e-6 relative). A cell whose primary methods (CSP, CDRO50, CDRO80) are not all
     usable is excluded and listed. Other methods are dropped only from their own statistics.
  2. Radius selection (training information only): eps* in {0, eps50, eps80} minimizes the mean expected cost over
     the validation family. Ties within 1e-6 relative go to the smaller radius. Pooled DRO uses the context-blind
     pooled family.
  3. Evaluation: J_m = p_true' C_m for every cell (context, dispersion level, number of reports, replicate) and every
     campus shift. p_true is the latent law. Equal weights over the grid.
  4. Primary endpoint and success rule (§8), on the complete registered grid only (otherwise INCOMPLETE):
     (i) in at least one context, a gain of at least 0.5 % and $3M with the replicate-bootstrap 95 % CI above zero;
     (ii) in both contexts, at zero shift and at each dispersion level, the CI upper bound of the harm ratio is at most
     0.5 %; (iii) in both contexts, the CI lower bound of the gain ratio is at least -0.5 %. The Monte Carlo CI is
     reported separately from the spread across the grid (model uncertainty). Every secondary comparison is computed
     on matched rows and reported with its denominator.
  5. Diagnostics: factorial, secondary comparators, ablations, the selected-radius distribution, oracle gap, per-shift
     and per-level breakdowns, the nominal premium, the worst grid truth and the W1-ball certificate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import highspy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import evidence as ev  # noqa: E402
from experiments import p1a_core as core  # noqa: E402

PRIMARY = ("CSP", "CDRO50", "CDRO80")
MIN_REL, MIN_ABS = 0.005, 3.0            # $M
NO_HARM = 0.005
N_BOOT, BOOT_SEED = 2000, 7


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
    if r is None or not r.get("ok") or r.get("tiebreak_ok") is False:
        return False
    if r.get("bookkeeping_maxabs", 1.0) > 1e-6 or r.get("backup_shortfall", 1.0) > 1e-6:
        return False
    if spec["method"] == "dro":
        obj = abs(r["status"].get("primary_obj") or 1.0)
        if r.get("dual_primal_gap", 1.0) > 1e-6 * max(1.0, obj):
            return False
    return True


def select(val, cands):
    """cands: list of (eps, C) in increasing eps. Returns (eps*, C*, scores)."""
    scores = [float((val @ np.asarray(C)).mean()) for _, C in cands]
    best = min(scores)
    for (e, C), s in zip(cands, scores):
        if s <= best + 1e-6 * abs(best):
            return e, C, scores
    raise RuntimeError


def ball_certificate(d, centre, eps, allowed):
    """min p'd over {p: W1(p, centre) <= eps, supp(p) in allowed} (LP over transport plans)."""
    n = len(d)
    sup = [i for i in range(n) if centre[i] > 1e-12]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)      # HiGHS's scheduler is process-global: every instance must agree
    g = {(i, j): h.addVariable(lb=0.0) for i in sup for j in allowed}
    for i in sup:
        h.addConstr(sum(g[i, j] for j in allowed) == centre[i])
    h.addConstr(sum(abs(i - j) * v for (i, j), v in g.items()) <= eps)
    h.minimize(sum(d[j] * v for (i, j), v in g.items()))
    return h.getInfo().objective_function_value


def analyze(run):
    plan, res = load(run)
    econ = plan["econ"]
    jobs = plan["jobs"]
    ok = {k: usable(jobs[k], res.get(k)) for k in jobs}
    C = {k: np.array(res[k]["C"]) for k in jobs if ok[k]}
    missing = [k for k in jobs if k not in res]
    rows, excluded, sel_ctx, sel_pool = [], [], [], []
    # ---- per cell -------------------------------------------------------------------------------------------
    trainings = {}
    for cell in plan["cells"]:
        z, lev, n_rep, r = cell["ctx"], cell["level"], cell["n_rep"], cell["r"]
        J = cell["jobs"]
        if cell["reports"] != json.loads(json.dumps(core.reports_of(z, lev, n_rep, r))):
            raise RuntimeError(f"report mismatch in cell {z} {lev} {n_rep} {r}")
        if not all(ok[J[m]] for m in PRIMARY):
            excluded.append(dict(ctx=z, level=lev, n_rep=n_rep, r=r,
                                 failed=[m for m in PRIMARY if not ok[J[m]]]))
            continue
        key = (lev, n_rep, r)
        if key not in trainings:
            tr = {y: core.training(y, core.reports_of(y, lev, n_rep, r)) for y in ("A", "B")}
            tr["pooled"] = core.pooled_training(tr["A"], tr["B"])
            trainings[key] = tr
        tr = trainings[key]
        e50, e80 = cell["radii"]["0.5"], cell["radii"]["0.8"]
        eps_star, C_cdro, scores = select(tr[z]["val"], [(0.0, C[J["CSP"]]), (e50, C[J["CDRO50"]]),
                                                         (e80, C[J["CDRO80"]])])
        tag = "" if eps_star == 0.0 else ("50" if eps_star == e50 else "80")
        sel_ctx.append(dict(ctx=z, level=lev, n_rep=n_rep, r=r, eps_star=eps_star, tag=tag or "0", scores=scores))
        meth = dict(CSP=C[J["CSP"]], CDRO=C_cdro, CDRO50=C[J["CDRO50"]], CDRO80=C[J["CDRO80"]])
        for abl in ("nobounds", "later"):
            kk = J[f"CDRO{tag}_{abl}"] if tag else J["CSP"]
            if ok[kk]:
                meth[f"CDRO_{abl}"] = C[kk]
        pe50, pe80 = cell["radii_pooled"]["0.5"], cell["radii_pooled"]["0.8"]
        pc = [(0.0, J["PSP"]), (pe50, J["PDRO50"]), (pe80, J["PDRO80"])]
        if all(ok[k] for _, k in pc):
            pe, C_pdro, _ = select(tr["pooled"]["val"], [(e, C[k]) for e, k in pc])
            sel_pool.append(dict(ctx=z, level=lev, n_rep=n_rep, r=r, eps_star=pe))
            meth["PSP"], meth["PDRO"] = C[J["PSP"]], C_pdro
        for m in ("BMIX", "BOX", "B1", "B7"):
            if ok[J[m]]:
                meth[m] = C[J[m]]
        centre = tr[z]["centre"]
        premium = float(centre @ (meth["CDRO"] - meth["CSP"]))
        cert = ball_certificate(meth["CSP"] - meth["CDRO"], centre, e80, ev.allowed_states(z)) if tag else 0.0
        for s in core.SHIFTS:
            p = ev.truth(z, cell["cv"], s)
            okey = core.oracle_key(econ, z, lev, s)
            v = float(p @ C[okey]) if ok.get(okey) else float("nan")
            row = dict(ctx=z, level=lev, n_rep=n_rep, r=r, shift=s, eps_star=eps_star, premium=premium,
                       certificate=cert, oracle=v)
            row.update({f"J_{m}": float(p @ c) for m, c in meth.items()})
            rows.append(row)
    return dict(econ=econ, rows=rows, excluded=excluded, missing=missing, sel_ctx=sel_ctx, sel_pool=sel_pool,
                n_jobs=len(jobs), n_usable=int(sum(ok.values())))


def _mean(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
    return float(v.mean()) if v.size else float("nan")


def matched(rows, a, b):
    """mean(J_a - J_b) over the rows where both are present and finite, with the denominator."""
    ka, kb = (a if a == "oracle" else f"J_{a}"), (b if b == "oracle" else f"J_{b}")
    v = np.array([r[ka] - r[kb] for r in rows if ka in r and kb in r and np.isfinite(r[ka]) and np.isfinite(r[kb])])
    return dict(diff=float(v.mean()) if v.size else float("nan"), n=int(v.size))


def boot(rows, z, rng):
    """Replicate bootstrap (replicates resampled within each (level, n_rep) stratum; a replicate keeps its five
    shifts). Returns 95 % intervals for: the primary difference dJ = mean(J_CSP - J_CDRO); its ratio to mean J_CSP;
    and, per dispersion level, the nominal harm ratio mean(J_CDRO - J_CSP) / mean(J_CSP) over zero-shift rows."""
    agg = {}
    for r in rows:
        if r["ctx"] != z:
            continue
        a = agg.setdefault((r["level"], r["n_rep"]), {}).setdefault(r["r"], np.zeros(5))
        a += (r["J_CSP"] - r["J_CDRO"], r["J_CSP"], 1.0, 0.0, 0.0)
        if r["shift"] == 0:
            a += (0.0, 0.0, 0.0, r["J_CDRO"] - r["J_CSP"], r["J_CSP"])
    draws = {"dJ": [], "rel": [], **{f"harm_{lv}": [] for lv in core.CV_LEVELS}}
    for _ in range(N_BOOT):
        tot = np.zeros(5)
        lv_tot = {lv: np.zeros(2) for lv in core.CV_LEVELS}
        for (lv, _n), reps in agg.items():
            ids = list(reps)
            pick = np.sum([reps[i] for i in rng.choice(ids, size=len(ids), replace=True)], axis=0)
            tot += pick
            lv_tot[lv] += pick[3:5]
        draws["dJ"].append(tot[0] / tot[2])
        draws["rel"].append(tot[0] / tot[1])
        for lv in core.CV_LEVELS:
            draws[f"harm_{lv}"].append(lv_tot[lv][0] / lv_tot[lv][1] if lv_tot[lv][1] > 0 else np.nan)
    return {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) for k, v in draws.items()}


def summarize(out):
    rows = out["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    S = dict(econ=out["econ"], n_jobs=out["n_jobs"], n_usable=out["n_usable"], missing=len(out["missing"]),
             excluded=out["excluded"], contexts={})
    expected = len(core.CV_LEVELS) * len(core.N_REPORTS) * core.R * len(core.SHIFTS)
    complete = True
    passes_i, passes_ii, passes_iii = [], [], []
    for z in ("A", "B"):
        R = [r for r in rows if r["ctx"] == z]
        # Every registered cell of both contexts must be present; otherwise the verdict is INCOMPLETE.
        if len(R) != expected:
            complete = False
        if not R:
            S["contexts"][z] = dict(n_rows=0, expected_rows=expected)
            continue
        mcsp = _mean(R, "J_CSP")
        dj = _mean(R, "J_CSP") - _mean(R, "J_CDRO")                     # primary pair: present in every row
        ci = boot(rows, z, rng)
        nom = [r for r in R if r["shift"] == 0]
        harm_lv = {lv: matched([r for r in nom if r["level"] == lv], "CDRO", "CSP")["diff"]
                   / _mean([r for r in nom if r["level"] == lv], "J_CSP") for lv in core.CV_LEVELS}
        ok_i = dj >= MIN_REL * mcsp and dj >= MIN_ABS and ci["dJ"][0] > 0
        ok_ii = all(ci[f"harm_{lv}"][1] <= NO_HARM for lv in core.CV_LEVELS)
        ok_iii = ci["rel"][0] >= -NO_HARM
        passes_i.append(ok_i)
        passes_ii.append(ok_ii)
        passes_iii.append(ok_iii)
        methods = sorted({k[2:] for r in R for k in r if k.startswith("J_")})
        gap = matched(R, "CSP", "oracle")
        cap_rows = [r for r in R if np.isfinite(r.get("oracle", np.nan))]
        ctxs = dict(n_rows=len(R), expected_rows=expected, mean_J_CSP=mcsp, primary_dJ=dj, primary_rel=dj / mcsp,
                    mc_ci95=ci["dJ"], rel_ci95=ci["rel"], nominal_harm_rel_by_level=harm_lv,
                    nominal_harm_rel_ci95={lv: ci[f"harm_{lv}"] for lv in core.CV_LEVELS},
                    crit_i=ok_i, crit_ii=ok_ii, crit_iii=ok_iii,
                    vs_CSP={m: matched(R, "CSP", m) for m in methods if m != "CSP"},
                    by_shift={s: {m: matched([r for r in R if r["shift"] == s], "CSP", m) for m in methods
                                  if m != "CSP"} for s in core.SHIFTS},
                    by_level={lv: matched([r for r in R if r["level"] == lv], "CSP", "CDRO") for lv in core.CV_LEVELS},
                    by_n_rep={n: matched([r for r in R if r["n_rep"] == n], "CSP", "CDRO") for n in core.N_REPORTS},
                    eps_star_freq={t: sum(1 for s in out["sel_ctx"] if s["ctx"] == z and s["tag"] == t)
                                   for t in ("0", "50", "80")},
                    mean_premium=_mean(R, "premium"),
                    oracle_gap_CSP=gap,
                    captured_by_CDRO=(matched(cap_rows, "CSP", "CDRO")["diff"] / gap["diff"]
                                      if gap["n"] and gap["diff"] > 1e-9 else float("nan")),
                    certificate_positive=sum(1 for r in R if r["shift"] == 0 and r["certificate"] > 1e-9),
                    worst_truth=min(((s, lv, matched([r for r in R if r["shift"] == s and r["level"] == lv],
                                                     "CSP", "CDRO")["diff"])
                                     for s in core.SHIFTS for lv in core.CV_LEVELS), key=lambda t: t[2]),
                    factorial=dict(conditioning_SP=matched(R, "PSP", "CSP"), conditioning_DRO=matched(R, "PDRO", "CDRO"),
                                   robustness_contextual=matched(R, "CSP", "CDRO"),
                                   robustness_pooled=matched(R, "PSP", "PDRO")))
        S["contexts"][z] = ctxs
    S["complete"] = complete and not out["excluded"]
    S["crit_i_any_context"] = any(passes_i)
    S["crit_ii_all_contexts"] = len(passes_ii) == 2 and all(passes_ii)
    S["crit_iii_all_contexts"] = len(passes_iii) == 2 and all(passes_iii)
    if not S["complete"]:
        S["verdict"] = "INCOMPLETE"
    else:
        S["verdict"] = "PASS" if (S["crit_i_any_context"] and S["crit_ii_all_contexts"]
                                  and S["crit_iii_all_contexts"]) else "FAIL"
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1a_main")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = analyze(a.run)
    S = summarize(out)
    path = pathlib.Path(a.out) if a.out else ROOT / "data" / f"p1a_summary_{a.run}.json"
    path.write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (path.with_suffix(".rows.json")).write_text(json.dumps(out["rows"], default=float), encoding="utf-8")
    print(f"{a.run} ({S['econ']}): jobs {S['n_jobs']}, usable {S['n_usable']}, missing {S['missing']}, "
          f"excluded cells {len(S['excluded'])}")
    for z, c in S["contexts"].items():
        if not c.get("n_rows"):
            print(f"  {z}: no admissible rows")
            continue
        harm = {lv: tuple(round(100 * x, 2) for x in v) for lv, v in c["nominal_harm_rel_ci95"].items()}
        print(f"  {z}: rows {c['n_rows']}/{c['expected_rows']} | E[J_CSP] {c['mean_J_CSP']:.1f} $M | CSP - CDRO "
              f"{c['primary_dJ']:+.2f} $M ({100 * c['primary_rel']:+.2f} %), MC 95% CI [{c['mc_ci95'][0]:+.2f}, "
              f"{c['mc_ci95'][1]:+.2f}] | nominal harm CI % {harm} | eps* {c['eps_star_freq']} | "
              f"crit (i) {c['crit_i']} (ii) {c['crit_ii']} (iii) {c['crit_iii']}")
    print(f"VERDICT: {S['verdict']}")


if __name__ == "__main__":
    main()
