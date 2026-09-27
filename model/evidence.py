"""Phase-1a evidence model (data/p1a_preregistration.md): clock-aligned connection-time laws, the report (observation)
mechanism, the plug-in estimator, the evidence-consistent calibration family, and Wasserstein-1 radii.

Clock: t = time of the campus's first grid energization, in quarters after the planning origin.
T = ceil(t) is the first quarter with grid power. Support {1..Q} plus the overflow state Q+1 ("T > Q").

Context A (ERCOT; the campus's requested in-service date E0 is known):
    t = E0 + D, D = delay beyond the requested date. R08: cohort-mean delay 220 days (applied once; this is the
    bias correction).
Context B (Dominion, engineering-study stage):
    t = S + O, S = engineering study (R11: "9 to 12 months"), O = order to energization (R12: "typically 3 years after
    the point of order"), conditioned on t <= 28 quarters (R02: N. Virginia "up to 7 years"). R12 describes realized
    energization, so no delay correction is added (DEV-5 kept: immediate progression from study to order).
The dispersion of D and O is not identified by the evidence. It is declared as a range: A's range brackets the R07
consistency check, and B's upper end respects R02.

Durations are Gamma distributions: A by mean and CV, B by median ("typically") and CV.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

Q = 28
DAYS_PER_Q = 91.3125
E0_A = 8                                   # DEV-4: requested in-service quarter of the context-A campus
MU_D_A = 220.0 / DAYS_PER_Q                # R08
MED_O_B = 12.0                             # R12: typically 3 years after the point of order
S_RANGE_B = (3.0, 4.0)                     # R11: 9-12 months
T_MAX_B = 28.0                             # R02: up to 7 years
CV_RANGE = {"A": (0.5, 1.0), "B": (0.25, 0.5)}
CV_HAT = {z: 0.5 * (lo + hi) for z, (lo, hi) in CV_RANGE.items()}
N_COHORT = 40                              # declared cohort size behind each report
# Stage/support bounds: the allowed connection quarters, inclusive; None = up to the overflow state.
# Only structural lower bounds: no energization before the requested date (A) or during the study (B). R02's "up to
# 7 years" is a regional summary, not a probability-zero tail, so it bounds no adversary.
BOUNDS = {"A": (E0_A, None), "B": (4, None), "pooled": (4, None)}


# ---------------------------------------------------------------------------------------------------------------
# Gamma helpers
# ---------------------------------------------------------------------------------------------------------------
def _k_theta_mean(mean, cv):
    k = 1.0 / cv ** 2
    return k, mean / k


def _k_theta_median(median, cv):
    k = 1.0 / cv ** 2
    return k, median / stats.gamma.median(k)


def _G(x, k, th):
    return stats.gamma.cdf(np.maximum(x, 0.0), k, scale=th) * (np.asarray(x) > 0)


def _H(x, k, th):
    """Integral of the Gamma CDF from 0 to x: x G_k(x) - k th G_{k+1}(x) (0 for x <= 0)."""
    x = np.asarray(x, dtype=float)
    xp = np.maximum(x, 0.0)
    return np.where(x > 0, xp * stats.gamma.cdf(xp, k, scale=th) - k * th * stats.gamma.cdf(xp, k + 1, scale=th), 0.0)


PMF_FLOOR = 1e-9                           # HiGHS drops or rejects coefficients below its 1e-9 small_matrix_value


def clean(p):
    """Declared numerical convention: masses below PMF_FLOOR are set to zero and the pmf is renormalized. Applied to
    every law and mixture that reaches the solver or the evaluation, so both see the same pmf."""
    p = np.where(np.asarray(p, dtype=float) < PMF_FLOOR, 0.0, p)
    return p / p.sum()


def pmf_from_cdf(F):
    """F = [P(t <= q) for q = 0..Q] -> pmf over T = 1..Q and the overflow state (T = ceil(t))."""
    F = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)
    F = np.maximum.accumulate(F)
    p = np.append(np.diff(F), 1.0 - F[-1])
    p = np.maximum(p, 0.0)
    return clean(p / p.sum())


# ---------------------------------------------------------------------------------------------------------------
# Connection-time laws
# ---------------------------------------------------------------------------------------------------------------
def law_A(mu, cv, shift=0.0, E0=E0_A):
    """t = E0 + max(0, D + shift), D ~ Gamma(mean mu, CV cv). A negative shift piles mass onto the requested date."""
    k, th = _k_theta_mean(mu, cv)
    qs = np.arange(0, Q + 1, dtype=float)
    F = np.where(qs < E0, 0.0, _G(qs - E0 - shift, k, th))
    return pmf_from_cdf(F)


def _F_B_raw(x, med, cv, s_lo, s_hi):
    k, th = _k_theta_median(med, cv)
    return (_H(x - s_lo, k, th) - _H(x - s_hi, k, th)) / (s_hi - s_lo)


def law_B(med, cv, shift=0.0, s_range=S_RANGE_B, t_max=T_MAX_B):
    """t = t_ev + shift, t_ev = S + O conditioned on t_ev <= t_max; S ~ U[s_range], O ~ Gamma(median med, CV cv)."""
    s_lo, s_hi = s_range
    qs = np.arange(0, Q + 1, dtype=float)
    x = qs - shift
    Fmax = _F_B_raw(t_max, med, cv, s_lo, s_hi)
    F = np.where(x >= t_max, 1.0, _F_B_raw(np.minimum(x, t_max), med, cv, s_lo, s_hi) / Fmax)
    return pmf_from_cdf(F)


def truth(ctx, cv, shift):
    """Latent campus law of the Phase-1a grid: the evidence-population law shifted by `shift` quarters."""
    return law_A(MU_D_A, cv, shift) if ctx == "A" else law_B(MED_O_B, cv, shift)


# ---------------------------------------------------------------------------------------------------------------
# Observation mechanism (reports) and plug-in estimator
# ---------------------------------------------------------------------------------------------------------------
def draw_reports(ctx, cv, n_rep, rng, n_c=N_COHORT):
    """Reports as the real evidence states them, drawn from the evidence-population law (no campus shift).
    A: a cohort-mean delay in days, rounded to 10 days (R08 form).
    B: a study-duration range in whole months (cohort 10th-90th percentiles, rounded outward; R11 form) and a
       typical order-to-energization time in whole years (cohort median; R12 form), with t = S + O <= 28 (R02)."""
    reps = []
    for _ in range(n_rep):
        if ctx == "A":
            k, th = _k_theta_mean(MU_D_A, cv)
            d = rng.gamma(k, th, size=n_c)
            reps.append(dict(mean_delay_days=10.0 * round(d.mean() * DAYS_PER_Q / 10.0)))
        else:
            k, th = _k_theta_median(MED_O_B, cv)
            S, O = np.empty(0), np.empty(0)
            while S.size < n_c:
                s = rng.uniform(*S_RANGE_B, size=n_c)
                o = rng.gamma(k, th, size=n_c)
                ok = s + o <= T_MAX_B
                S, O = np.append(S, s[ok]), np.append(O, o[ok])
            S, O = S[:n_c], O[:n_c]
            reps.append(dict(study_months=(float(np.floor(3 * np.quantile(S, 0.1))),
                                           float(np.ceil(3 * np.quantile(S, 0.9)))),
                             typical_years=float(round(np.median(O) / 4.0))))
    return reps


def report_distribution(ctx, cv, coverage=1 - 1e-6, n_mc=400_000, seed=31337, n_c=N_COHORT):
    """The distribution of ONE report (n_rep = 1) under the evidence population with dispersion cv (Phase 1c).

    A, exact. The report is 10 * round(mean_days / 10), and the cohort mean of n_c Gamma(k, th) delays is
       Gamma(n_c k, th / n_c). Outcomes are added from the most probable outward until `coverage` is reached. The
       omitted mass (1 - covered) is returned.
    B, simulation. n_mc cohorts of the registered mechanism (fixed seed): the study range (floor and ceil of the cohort
       10-90 % percentiles, in months) and the typical years (round of the cohort median / 4), with t = S + O <= 28.
       Probabilities carry their Monte Carlo standard error.

    Returns (outcomes, covered). outcomes = [dict(report=[...], p=..., se=...)], with reports in the draw_reports format."""
    if ctx == "A":
        k, th = _k_theta_mean(MU_D_A, cv)
        K, TH = n_c * k, th / n_c

        def pr(x):
            lo, hi = max(x - 5.0, 0.0), x + 5.0
            return float(stats.gamma.cdf(hi / DAYS_PER_Q, K, scale=TH) - stats.gamma.cdf(lo / DAYS_PER_Q, K, scale=TH))

        x0 = 10.0 * round(MU_D_A * DAYS_PER_Q / 10.0)
        outs = {x0: pr(x0)}
        lo = hi = x0
        tot = outs[x0]
        while tot < coverage:
            pl = pr(lo - 10.0) if lo - 10.0 >= 0 else -1.0
            ph = pr(hi + 10.0)
            if pl >= ph:
                lo -= 10.0
                outs[lo] = pl
                tot += pl
            else:
                hi += 10.0
                outs[hi] = ph
                tot += ph
        res = [dict(report=[dict(mean_delay_days=x)], p=p, se=0.0) for x, p in sorted(outs.items())]
        return res, float(tot)
    k, th = _k_theta_median(MED_O_B, cv)
    rng = np.random.default_rng(seed)
    counts = {}
    done, chunk, width = 0, 20000, int(np.ceil(1.6 * n_c)) + 10
    while done < n_mc:
        m = min(chunk, n_mc - done)
        S_sel, O_sel = np.empty((m, n_c)), np.empty((m, n_c))
        todo = np.arange(m)
        while todo.size:
            s = rng.uniform(*S_RANGE_B, size=(todo.size, width))
            o = rng.gamma(k, th, size=(todo.size, width))
            acc = s + o <= T_MAX_B
            cnt = np.cumsum(acc, axis=1)
            full = cnt[:, -1] >= n_c
            sel = acc & (cnt <= n_c)
            S_sel[todo[full]] = s[full][sel[full]].reshape(-1, n_c)
            O_sel[todo[full]] = o[full][sel[full]].reshape(-1, n_c)
            todo = todo[~full]
        a = np.floor(3 * np.quantile(S_sel, 0.1, axis=1))
        b = np.ceil(3 * np.quantile(S_sel, 0.9, axis=1))
        y = np.round(np.median(O_sel, axis=1) / 4.0)
        for key in zip(a, b, y):
            counts[key] = counts.get(key, 0) + 1
        done += m
    res = []
    for (a_, b_, y_), n in sorted(counts.items()):
        p = n / n_mc
        res.append(dict(report=[dict(study_months=(float(a_), float(b_)), typical_years=float(y_))], p=p,
                        se=float(np.sqrt(p * (1 - p) / n_mc))))
    return res, 1.0


REAL_REPORTS = {"A": [dict(mean_delay_days=220.0)],
                "B": [dict(study_months=(9.0, 12.0), typical_years=3.0)]}


def estimate(ctx, reports):
    """Plug-in, bias-corrected predictive law (the common centre of SP and DRO)."""
    if ctx == "A":
        mu = np.mean([r["mean_delay_days"] for r in reports]) / DAYS_PER_Q
        return law_A(mu, CV_HAT["A"])
    a = np.mean([r["study_months"][0] for r in reports]) / 3.0
    b = np.mean([r["study_months"][1] for r in reports]) / 3.0
    med = 4.0 * np.mean([r["typical_years"] for r in reports])
    return law_B(med, CV_HAT["B"], s_range=(a, b))


# ---- calibration: the posterior of the latent parameters given the reports ---------------
PRIOR_MU_A = (0.25, 8.0)                   # uniform prior on A's mean delay, quarters (23 to 730 days)
PRIOR_MED_B = (4.0, 24.0)                  # uniform prior on B's order-to-energization median, quarters (1 to 6 years)
N_PRIOR_A = 40000
GRID_B = (np.arange(PRIOR_MED_B[0], PRIOR_MED_B[1] + 1e-9, 0.25),
          np.round(np.arange(CV_RANGE["B"][0], CV_RANGE["B"][1] + 1e-9, 0.025), 6))
M_SIM_B, SEED_LIK_B = 4000, 424242


def _loglik_A(reports, mu, cv, n_c):
    """Exact: the mean of n_c iid Gamma(k, th) is Gamma(n_c k, th / n_c); a report is that mean in days, rounded to
    10 days."""
    k = 1.0 / cv ** 2
    th = mu / k
    ll = np.zeros_like(mu)
    for r in reports:
        x = r["mean_delay_days"]
        lo, hi = (x - 5.0) / DAYS_PER_Q, (x + 5.0) / DAYS_PER_Q
        p = stats.gamma.cdf(hi, n_c * k, scale=th / n_c) - stats.gamma.cdf(lo, n_c * k, scale=th / n_c)
        ll = ll + np.log(np.maximum(p, 1e-300))
    return ll


_LIK_B_CACHE = {}


def lik_table_B(s_range, n_c=N_COHORT):
    """P(reported typical years = y | median, CV) on GRID_B, by simulation of the registered observation model (a
    cohort of n_c accepted pairs with S ~ U[s_range], O ~ Gamma, S + O <= 28; the report is round(median(O) / 4)).
    Fixed seed; cached per study range."""
    key = (round(s_range[0], 9), round(s_range[1], 9), n_c)
    if key in _LIK_B_CACHE:
        return _LIK_B_CACHE[key]
    rng = np.random.default_rng(SEED_LIK_B)
    meds, cvs = GRID_B
    tab = np.zeros((len(meds), len(cvs), 12))
    for i, med in enumerate(meds):
        for j, cv in enumerate(cvs):
            k, th = _k_theta_median(med, cv)
            p_acc = max(float(_F_B_raw(T_MAX_B, med, cv, *s_range)), 1e-3)
            width = int(np.ceil(1.5 * n_c / p_acc)) + 10
            O_sel = np.empty((M_SIM_B, n_c))
            todo = np.arange(M_SIM_B)
            while todo.size:
                s = rng.uniform(*s_range, size=(todo.size, width))
                o = rng.gamma(k, th, size=(todo.size, width))
                acc = s + o <= T_MAX_B
                cnt = np.cumsum(acc, axis=1)
                full = cnt[:, -1] >= n_c
                sel = acc & (cnt <= n_c)
                O_sel[todo[full]] = o[full][sel[full]].reshape(-1, n_c)
                todo = todo[~full]
            y = np.round(np.median(O_sel, axis=1) / 4.0).astype(int)
            tab[i, j] = np.bincount(np.clip(y, 0, 11), minlength=12)[:12] / M_SIM_B
    _LIK_B_CACHE[key] = tab
    return tab


def calibration_family(ctx, reports, n, rng, n_c=N_COHORT, return_params=False):
    """Pseudo-truth laws drawn from the posterior of the latent parameters given the reports. It uses the registered
    observation model and uniform priors (training information only):
    A: mean ~ U[PRIOR_MU_A], CV ~ U[CV_RANGE]; exact likelihood; importance resampling from N_PRIOR_A prior draws.
    B: median ~ U[PRIOR_MED_B], CV ~ U[CV_RANGE] on GRID_B; simulated likelihood of the reported typical years; draws
       from the grid posterior, jittered within their cell. The study range is taken as reported.
    The posterior mean law is the Bayesian predictive distribution (BMIX)."""
    out = np.empty((n, Q + 1))
    lo, hi = CV_RANGE[ctx]
    if ctx == "A":
        mu = rng.uniform(*PRIOR_MU_A, size=N_PRIOR_A)
        cv = rng.uniform(lo, hi, size=N_PRIOR_A)
        ll = _loglik_A(reports, mu, cv, n_c)
        w = np.exp(ll - ll.max())
        idx = rng.choice(N_PRIOR_A, size=n, replace=True, p=w / w.sum())
        params = np.column_stack([mu[idx], cv[idx]])
        for i, (m_, c_) in enumerate(params):
            out[i] = law_A(m_, c_)
    else:
        a = np.mean([r["study_months"][0] for r in reports]) / 3.0
        b = np.mean([r["study_months"][1] for r in reports]) / 3.0
        tab = lik_table_B((a, b), n_c)
        ll = np.zeros(tab.shape[:2])
        for r in reports:
            ll = ll + np.log(np.maximum(tab[:, :, int(r["typical_years"])], 1e-12))
        w = np.exp(ll - ll.max()).ravel()
        idx = rng.choice(w.size, size=n, replace=True, p=w / w.sum())
        meds, cvs = GRID_B
        mi, ci = np.unravel_index(idx, tab.shape[:2])
        m_ = np.clip(meds[mi] + rng.uniform(-0.125, 0.125, size=n), *PRIOR_MED_B)
        c_ = np.clip(cvs[ci] + rng.uniform(-0.0125, 0.0125, size=n), lo, hi)
        params = np.column_stack([m_, c_])
        for i in range(n):
            out[i] = law_B(m_[i], c_[i], s_range=(a, b))
    return (out, params) if return_params else out


# ---------------------------------------------------------------------------------------------------------------
# Wasserstein-1 on the ordered support (state Q+1 at unit distance beyond Q; a declared convention)
# ---------------------------------------------------------------------------------------------------------------
def w1(p, r):
    return float(np.abs(np.cumsum(p)[:-1] - np.cumsum(r)[:-1]).sum())


def radii(p_tilde, family, qs=(0.5, 0.8)):
    d = np.array([w1(p, p_tilde) for p in family])
    return {q: float(np.quantile(d, q)) for q in qs}


def allowed_states(ctx):
    """0-based indices of the allowed connection states (T = index + 1) under the stage bounds."""
    lo, hi = BOUNDS[ctx]
    hi = Q + 1 if hi is None else hi
    return [i for i in range(Q + 1) if lo <= i + 1 <= hi]
