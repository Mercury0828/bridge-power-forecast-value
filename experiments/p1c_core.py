"""Phase 1c: confirmation of the implemented information value (data/p1c_preregistration.md).

Training (the planner's information): one rounded report of its context's evidence population (the Phase-1a
observation mechanism), and the plug-in centre p~ = model.evidence.estimate(report).
Repeated estimation is handled by INTEGRATION over the report-outcome distribution (exact for
A, Monte Carlo with stated coverage for B):
- every report outcome of the (context, dispersion level) is one cell, weighted by its probability
  (model.evidence.report_distribution);
- A is exact, with coverage >= 1 - 1e-6;
- B is simulated from 400,000 cohorts, with a Monte Carlo standard error.

Signal.
- At node TAU, if the campus is not yet connected (T > TAU), the utility issues an estimated energization quarter
  Y = T + eps.
- The planner maps Y into K = 3 categories, with thresholds at the terciles of its own centre p~ (training
  information).

Frozen reliability model of the planner.
- eps is symmetric, unbiased and a discretized normal with sd sigma.
- sigma is uniform on SIGMA_PRIOR (1 to 4 quarters).
- The planner's category kernel Kbar is the average over that prior.

Policies (same training information, same economics):
- nosig: SP on p~ (Model): the no-signal predictive planner;
- sig: SP on the planner's joint law p~(t) Kbar(c | t) in the signal tree (SignalModel). It anticipates the signal in
  its root decisions but not its realization.
Both use the same lexicographic tie-break (D-008) in its LP form: the first stage's integer decisions are fixed, and the
continuous decisions minimize the uniform sum of scenario costs among primary-optimal solutions.

Evaluation families (fixed before any run; analysed exactly in experiments/analyze_p1c.py):
- sym1, sym2, sym4: symmetric unbiased errors with sd 1, 2 and 4 quarters;
- opt2: optimistic, eps = −2 + symmetric sd 2 (a labelled assumption);
- uninf: Y independent of T (a control).
The truth marginal is the evidence-population law of the cell's dispersion level: Δ = 0 is primary, and
Δ = −2, +2 are secondary.

Oracle diagnostic: SP on each true joint law (Δ = 0) in the signal tree, with its certified dual bound.
Every joint law that reaches the solver or the evaluation is floored at 1e-9 and renormalized (model.signal.clean_law;
a declared numerical convention, as in Phase 1a v2.2).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import evidence as ev  # noqa: E402
from model.bridge import (make_params, Model, polish, recompute_costs, backup_shortfall, percentile_quarter,  # noqa: E402
                          cost_components, policy_detail)
from experiments import p1a_core as p1a  # noqa: E402

MASTER_SEED = 20260924
TAU = 4
NCAT = 3
LEVELS = ("low", "high")
REPORT_COVERAGE = 1 - 1e-6          # A: report outcomes are added until this probability is covered
REPORT_MC = 400_000                 # B: simulated cohorts for the report-outcome probabilities
REPORT_SEED = 31337
J_BOUND = 1000.0                    # $M: a bound on |Δ_info| for the uncovered report mass (lifecycle-cost scale)
CONFIGS = {"A": {}, "A+staged": dict(stage_len=4, stage_frac=0.25),
           "B": {}, "B+staged": dict(stage_len=4, stage_frac=0.25)}
SIGMA_PRIOR = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
FAMILIES = {"sym1": dict(sd=1.0, bias=0.0), "sym2": dict(sd=2.0, bias=0.0), "sym4": dict(sd=4.0, bias=0.0),
            "opt2": dict(sd=2.0, bias=-2.0), "uninf": dict(sd=2.0, bias=0.0, independent=True)}
SHIFTS = (0, -2, 2)
EPS_SUPPORT = np.arange(-16, 17)


def base_ctx(cfg):
    return cfg.partition("+")[0]


def params(cfg):
    return make_params(base_ctx(cfg), **dict(p1a.ECON["main"], **CONFIGS[cfg]))


def _seed(*parts):
    s = json.dumps([MASTER_SEED, *parts], sort_keys=True)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


_OUTCOMES = {}


def report_outcomes(ctx, level):
    """[(report, p, se)], covered mass. Cached; deterministic."""
    key = (ctx, level)
    if key not in _OUTCOMES:
        outs, cov = ev.report_distribution(ctx, p1a.cv_of(ctx, level), coverage=REPORT_COVERAGE, n_mc=REPORT_MC,
                                           seed=REPORT_SEED)
        _OUTCOMES[key] = (outs, cov)
    return _OUTCOMES[key]


# ---- signal model ---------------------------------------------------------------------------------------------
def error_pmf(sd, bias=0.0):
    """Discretized normal error on integer quarters, mean `bias`, sd `sd`."""
    e = EPS_SUPPORT
    w = stats.norm.cdf((e + 0.5 - bias) / sd) - stats.norm.cdf((e - 0.5 - bias) / sd)
    return w / w.sum()


def thresholds(centre):
    """Tercile thresholds (in T units) of the planner's centre: training information."""
    return percentile_quarter(centre, 1 / 3), percentile_quarter(centre, 2 / 3)


def category(y, th):
    q1, q2 = th
    return 0 if y <= q1 else (1 if y <= q2 else 2)


def kernel(th, sd, bias=0.0):
    """K[c][i] = P(category c | T = i + 1) for an error law N(bias, sd) on integer quarters."""
    pe = error_pmf(sd, bias)
    K = np.zeros((NCAT, ev.Q + 1))
    for i in range(ev.Q + 1):
        t = i + 1
        for e, w in zip(EPS_SUPPORT, pe):
            K[category(t + e, th), i] += w
    return K


def planner_kernel(th):
    return np.mean([kernel(th, s) for s in SIGMA_PRIOR], axis=0)


def exact_kernel(th):
    """Phase 1f E-N: a planner that treats the estimate category as exact, K[c][i] = 1 if category(i + 1) = c."""
    K = np.zeros((NCAT, ev.Q + 1))
    for i in range(ev.Q + 1):
        K[category(i + 1, th), i] = 1.0
    return K


def true_kernel(th, family, p_true):
    f = FAMILIES[family]
    K = kernel(th, f["sd"], f["bias"])
    if f.get("independent"):
        # Y independent of T: the category law is the marginal category law of an independent draw.
        m = K @ np.asarray(p_true)
        K = np.repeat(m[:, None], ev.Q + 1, axis=1)
    return K


def truth(cfg, level, shift):
    return ev.truth(base_ctx(cfg), p1a.cv_of(base_ctx(cfg), level), shift)


# ---- plan -----------------------------------------------------------------------------------------------------
def job_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def plan():
    """-> (cells, jobs). A cell is one report outcome of one configuration and dispersion level, with its probability."""
    from model.signal import joint_law, clean_law
    cells, jobs = [], {}

    def add(spec):
        k = job_key(spec)
        jobs.setdefault(k, spec)
        return k

    for cfg in CONFIGS:
        ctx = base_ctx(cfg)
        for level in LEVELS:
            outs, cov = report_outcomes(ctx, level)
            for r, o in enumerate(outs):
                rep = o["report"]
                c = ev.estimate(ctx, rep)
                th = thresholds(c)
                Kp = planner_kernel(th)
                keys = dict(nosig=add(dict(kind="nosig", cfg=cfg, centre=c.tolist())),
                            sig=add(dict(kind="sig", cfg=cfg, law=clean_law(joint_law(c, Kp, TAU)))))
                orc = {}
                for fam in FAMILIES:
                    pt = truth(cfg, level, 0)
                    orc[fam] = add(dict(kind="oracle", cfg=cfg,
                                        law=clean_law(joint_law(pt, true_kernel(th, fam, pt), TAU))))
                keys["oracle"] = orc
                cells.append(dict(cfg=cfg, level=level, r=r, p=o["p"], se=o["se"], covered=cov,
                                  reports=json.loads(json.dumps(rep)), thresholds=list(th), jobs=keys))
    return cells, jobs


# ---- run ------------------------------------------------------------------------------------------------------
def _check(P, m):
    return float(np.max(np.abs(m.C_values() - recompute_costs(P, m)))), float(backup_shortfall(P, m))


def run_job(spec, time_limit=1800.0, tiebreak_mode="lp", tiebreak_presolve=True):
    """`tiebreak_mode` stays "lp" (pre-registration §4). The D-022 repair (experiments/p1c_repair.py) is the only caller
    that changes these: it tried "lp_pure" (the same LP solved as a pure LP), then, as amended, "lp" with the
    second-stage presolve off."""
    try:
        return _run_job(spec, time_limit, tiebreak_mode, tiebreak_presolve)
    except Exception as e:                                        # noqa: BLE001
        return dict(spec_key=job_key(spec), ok=False, error=f"{type(e).__name__}: {e}")


def _run_job(spec, time_limit, tiebreak_mode="lp", tiebreak_presolve=True, P=None):
    """`P` overrides the configuration's parameters (Phase-1d exploratory variants; experiments/p1d_core.py)."""
    from model.signal import SignalModel
    P = params(spec["cfg"]) if P is None else P
    if spec["kind"] == "nosig":
        m = Model(P, time_limit=time_limit)
        st = m.solve_saa(np.array(spec["centre"]), tiebreak_mode=tiebreak_mode, tiebreak_presolve=tiebreak_presolve)
        if not st["ok"]:
            return dict(spec_key=job_key(spec), ok=False, status=st["status"])
        pm, pst = polish(P, m.spine_values(), time_limit=time_limit)
        bk, sh = _check(P, pm)
        return dict(spec_key=job_key(spec), ok=bool(pst["ok"]), tiebreak_ok=st.get("tiebreak_ok"),
                    C=pm.C_values().tolist(), bookkeeping_maxabs=bk, backup_shortfall=sh,
                    summary=p1a._summary(P, m), primary_gap=st.get("primary_gap"),
                    components={k: v.tolist() for k, v in cost_components(P, pm).items()},
                    detail=policy_detail(P, pm))
    gap = 1e-5 if spec["kind"] == "oracle" else 1e-4
    sm = SignalModel(P, NCAT, spec.get("tau", TAU), gap=gap, time_limit=time_limit)
    st = sm.solve_saa(spec["law"], tiebreak=(spec["kind"] == "sig"), tiebreak_mode=tiebreak_mode,
                      tiebreak_presolve=tiebreak_presolve)
    if not st["ok"]:
        return dict(spec_key=job_key(spec), ok=False, status=st["status"])
    costs, bks, shs, comps, details = [], [], [], [], []
    for c in sm.copies:
        pm, pst = polish(P, c.spine_values(), time_limit=time_limit)
        if not pst["ok"]:
            return dict(spec_key=job_key(spec), ok=False, status="polish failed")
        costs.append(pm.C_values().tolist())
        if spec["kind"] == "sig":
            comps.append({k: v.tolist() for k, v in cost_components(P, pm).items()})
            details.append(policy_detail(P, pm))
        bk, sh = _check(P, pm)
        bks.append(bk)
        shs.append(sh)
    out = dict(spec_key=job_key(spec), ok=True, tiebreak_ok=st.get("tiebreak_ok"), costs=costs,
               bookkeeping_maxabs=max(bks), backup_shortfall=max(shs), primary_obj=st.get("primary_obj", st["obj"]),
               gap=st["gap"], dual_bound=st["dual_bound"])
    if spec.get("variant") and st.get("primary_gap") is not None:          # Phase-1f runs onward (not Phase 1c items)
        out.update(primary_gap=st["primary_gap"], primary_dual_bound=st["primary_dual_bound"])
    if spec["kind"] == "sig":
        out["summary"] = [p1a._summary(P, c) for c in sm.copies]
        out["components"] = comps
        out["detail"] = details
    return out
