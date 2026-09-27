"""Phase-1a finite-sample experiment (data/p1a_preregistration.md).

Stage 1, `plan(econ)`: for every evidence cell (context, dispersion level, number of reports, replicate):
  - draw the reports with a fixed seed;
  - build the plug-in centre, the calibration family (seeded by the reports, so identical training information gives
    identical inputs), the radii and the predictive mixture;
  - collect the policy jobs, keyed by a hash of their inputs, so each distinct policy is solved once.
  Oracle jobs (SAA on each latent truth) are added for the oracle-gap diagnostic.
Stage 2, `run_job(spec)`: the harness (kind p1a) runs every job. A job is a policy optimized on its inputs, then
  branch-polished, and yields its cost vector C over T = 1..Q+1.
Stage 3: experiments/analyze_p1a.py (evaluation under the latent truths, radius selection, success rule).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import evidence as ev  # noqa: E402
from model.bridge import (make_params, Model, policy_rolling_det, polish, recompute_costs,  # noqa: E402
                          backup_shortfall, percentile_quarter)

# ---- frozen design (data/p1a_preregistration.md) -----------------------------------------------------------------
BASE = dict(p_R=25.0, cap_R=100.0, delay_mult=1.0, residence_on=True, backup_rating="DCC", backup_sym=True,
            age_salvage=True, backup_ramp=True, service_hard=True)
ECON = {"main": dict(BASE), "salv05": dict(BASE, salv_mult=0.5), "salv0": dict(BASE, salv_mult=0.0)}
CV_LEVELS = ("low", "high")
N_REPORTS = (1, 3)
R = 24
SHIFTS = (-2, -1, 0, 1, 2)
N_CAL = 400
N_VAL = 400
MASTER_SEED = 20260923


def cv_of(ctx, level):
    return ev.CV_RANGE[ctx][CV_LEVELS.index(level)]


def _seed(*parts):
    s = json.dumps([MASTER_SEED, *parts], sort_keys=True)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


def reports_of(ctx, level, n_rep, r):
    return ev.draw_reports(ctx, cv_of(ctx, level), n_rep, np.random.default_rng(_seed("reports", ctx, level, n_rep, r)))


_TRAINING = {}


def training(ctx, reports):
    """Everything a planner derives from its training information alone. Deterministic in (ctx, reports), so it is
    cached by them; callers must not mutate the result."""
    key = json.dumps([ctx, reports], sort_keys=True)
    if key not in _TRAINING:
        centre = ev.estimate(ctx, reports)
        fam = ev.calibration_family(ctx, reports, N_CAL, np.random.default_rng(_seed("cal", ctx, reports)))
        val = ev.calibration_family(ctx, reports, N_VAL, np.random.default_rng(_seed("val", ctx, reports)))
        _TRAINING[key] = dict(centre=centre, fam=fam, val=val, radii=ev.radii(centre, fam),
                              p_mix=ev.clean(fam.mean(axis=0)))
    return _TRAINING[key]


def pooled_training(tA, tB):
    """Context-blind pooling: the same procedure applied to the pooled evidence (equal weights)."""
    centre = ev.clean(0.5 * (tA["centre"] + tB["centre"]))
    fam, val = 0.5 * (tA["fam"] + tB["fam"]), 0.5 * (tA["val"] + tB["val"])
    return dict(centre=centre, fam=fam, val=val, radii=ev.radii(centre, fam), p_mix=ev.clean(fam.mean(axis=0)))


def job_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def _spec(econ, ctx, method, **kw):
    s = dict(econ=econ, ctx=ctx, method=method)
    for k, v in kw.items():
        s[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return s


def contextual_specs(econ, ctx, t):
    """Contextual methods for a campus in `ctx` trained on `t` -> {method name: spec}."""
    c, allowed = t["centre"], ev.allowed_states(ctx)
    lo, hi = percentile_quarter(t["p_mix"], 0.05), percentile_quarter(t["p_mix"], 0.95)
    out = {"CSP": _spec(econ, ctx, "saa", centre=c),
           "BMIX": _spec(econ, ctx, "saa", centre=t["p_mix"]),
           "BOX": _spec(econ, ctx, "box", lo=int(lo), hi=int(hi)),
           "B1": _spec(econ, ctx, "rolling", centre=c)}
    for q, tag in ((0.5, "50"), (0.8, "80")):
        e = t["radii"][q]
        out[f"CDRO{tag}"] = _spec(econ, ctx, "dro", centre=c, eps=e, allowed=allowed, later_only=False)
        out[f"CDRO{tag}_nobounds"] = _spec(econ, ctx, "dro", centre=c, eps=e, allowed=None, later_only=False)
        out[f"CDRO{tag}_later"] = _spec(econ, ctx, "dro", centre=c, eps=e, allowed=allowed, later_only=True)
    return out


def pooled_specs(econ, ctx, tp):
    c, allowed = tp["centre"], ev.allowed_states("pooled")
    out = {"PSP": _spec(econ, ctx, "saa", centre=c)}
    for q, tag in ((0.5, "50"), (0.8, "80")):
        out[f"PDRO{tag}"] = _spec(econ, ctx, "dro", centre=c, eps=tp["radii"][q], allowed=allowed, later_only=False)
    return out


def plan(econ="main"):
    """-> (cells, jobs). cells: one per (ctx, level, n_rep, r) with the job key of every method; jobs: key -> spec."""
    cells, jobs = [], {}

    def add(spec):
        k = job_key(spec)
        jobs.setdefault(k, spec)
        return k

    for level in CV_LEVELS:
        for n_rep in N_REPORTS:
            for r in range(R):
                tr = {z: training(z, reports_of(z, level, n_rep, r)) for z in ("A", "B")}
                tp = pooled_training(tr["A"], tr["B"])
                for z in ("A", "B"):
                    keys = {name: add(s) for name, s in contextual_specs(econ, z, tr[z]).items()}
                    keys.update({name: add(s) for name, s in pooled_specs(econ, z, tp).items()})
                    keys["B7"] = add(_spec(econ, z, "det", T_hat=ev.Q + 1))
                    cells.append(dict(ctx=z, level=level, cv=cv_of(z, level), n_rep=n_rep, r=r,
                                      reports=reports_of(z, level, n_rep, r),
                                      radii={str(q): v for q, v in tr[z]["radii"].items()},
                                      radii_pooled={str(q): v for q, v in tp["radii"].items()}, jobs=keys))
    for z in ("A", "B"):
        for level in CV_LEVELS:
            for s in SHIFTS:
                add(_spec(econ, z, "saa", centre=ev.truth(z, cv_of(z, level), s), oracle=[level, s]))
    return cells, jobs


def oracle_key(econ, ctx, level, shift):
    return job_key(_spec(econ, ctx, "saa", centre=ev.truth(ctx, cv_of(ctx, level), shift), oracle=[level, shift]))


# ---- stage 2 --------------------------------------------------------------------------------------------------
def _summary(P, m):
    sv = m.spine_values()
    return dict(n_GE=sv.get("n|GE"), n_DG=sv.get("n|DG"), z=sv.get("z"),
                orders_MW={j: round(sum(sv[f"x|{j}|{k}"] for k in range(P.Q + 1)), 3) for j in P.assets},
                root_MW={j: round(sv[f"x|{j}|0"], 3) for j in P.assets},
                filings=[k for k in range(P.Q) if sv[f"f|{k}"] == 1],
                rent_ST_MWq=round(sum(sv[f"RST|{q}"] for q in range(1, P.Q + 1)), 3),
                rent_NR_MWq=round(sum(sv[f"RNR|{q}"] for q in range(1, P.Q + 1)), 3))


def run_job(spec, time_limit=600.0):
    """`time_limit` (s per MILP) is not part of the job key: the pre-registered remedy for a timed-out job is a re-run
    with a longer limit and otherwise identical inputs (data/p1a_preregistration.md §10). An exception makes the job
    not ok (so the remedy and the INCOMPLETE rule handle it) instead of stopping the whole run."""
    try:
        return _run_job(spec, time_limit)
    except Exception as e:                                        # noqa: BLE001
        return dict(spec_key=job_key(spec), ok=False, status=dict(ok=False), error=f"{type(e).__name__}: {e}")


def _run_job(spec, time_limit):
    P = make_params(spec["ctx"], **ECON[spec["econ"]])
    kind = spec["method"]
    c = np.array(spec["centre"]) if "centre" in spec else None
    t0 = time.time()
    extra = {}
    if kind == "rolling":
        m, stats = policy_rolling_det(P, c, time_limit=time_limit)
        st = dict(ok=m is not None and all(s.get("ok", False) for s in stats),
                  tiebreak_ok=all(s.get("tiebreak_ok") is not False for s in stats),
                  primary_gap=max((s.get("primary_gap") or 0.0) for s in stats), n_solves=len(stats))
    else:
        m = Model(P, time_limit=time_limit)
        if kind == "saa":
            st = m.solve_saa(c)
        elif kind == "dro":
            st = m.solve_dro(c, spec["eps"], allowed=spec["allowed"], later_only=spec["later_only"])
        elif kind == "box":
            st = m.solve_box(spec["lo"], spec["hi"])
        elif kind == "det":
            st = m.solve_det(spec["T_hat"])
        else:
            raise ValueError(kind)
        if st.get("ok") and kind == "dro":
            from model.bridge import worst_case_expectation
            wc = worst_case_expectation(m.C_values(), c, spec["eps"], allowed=spec["allowed"],
                                        later_only=spec["later_only"])
            extra["dual_primal_gap"] = float(abs(wc - st.get("primary_obj", st["obj"])))
    status = {k: st.get(k) for k in ("ok", "status", "primary_obj", "primary_status", "primary_gap", "tiebreak_ok",
                                     "tiebreak_status", "n_solves")}
    if m is None or not st.get("ok", False):
        return dict(spec_key=job_key(spec), ok=False, status=status)
    pm, pst = polish(P, m.spine_values(), time_limit=time_limit)
    C = pm.C_values()
    return dict(spec_key=job_key(spec), ok=bool(pst["ok"]), status=status, polish_ok=bool(pst["ok"]),
                tiebreak_ok=st.get("tiebreak_ok"), C=C.tolist(), C_unpolished=m.C_values().tolist(),
                bookkeeping_maxabs=float(np.max(np.abs(C - recompute_costs(P, pm)))),
                backup_shortfall=float(backup_shortfall(P, pm)), summary=_summary(P, m), **extra)
