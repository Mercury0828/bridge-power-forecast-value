"""Build the Phase-0 ambiguity-set centres for T_grid from the R1 evidence table (v2).

v2 (2026-09-22) changes:
  1.   The horizon differs from the pre-registered {1..12, >12}. It is now declared as DEV-3. The generator reports
       censoring-robust statistics (median, P(T<=12), P(T>Q)) and never treats the overflow state as an exact date.
  2.   R07 (ERCOT officer-letter cohort, 55.4 % in service by Feb 2025) has no common stated date, so it is no longer
       a centre component. It is used only as a CONSISTENCY CHECK against the R08-based delay component.
  3.   R14 (AEP aggregate ramp onset) is removed from the centres. AEP Ohio is a separately registered z1 level
       ("PJM-other"), and pooling it into context B broke the pre-registered context definition.

Construction rules come from data/r1_preregistration.md section 4 (commit 723d6eb, made before the evidence):
  rule (1) stated range [a, b] years -> uniform over integer quarters in [a, b], with overflow to "> Q"
  rule (2) stated central value only  -> symmetric discrete triangle, +/-50 % of the stated statistic
  rule (3) realization rate by a stated date (not used in v2: no record supplies a common stated date)
  several records per context         -> equal-weight mixture
Declared deviations (all printed at run time):
  DEV-1 "up to X years" (no lower bound; R02) -> range [1 year, X years], using the US-general lower bound (R01).
        This is a declared exploratory assumption.
  DEV-3 horizon Q = 28 quarters instead of the pre-registered 12. Reason: context-B evidence (R02 up to 7 years;
        R11+R12 about 4 years) lies mostly beyond 12 quarters; at Q = 12 about 80 % of B's mass would be censored,
        leaving no decision-relevant support. Made after compiling evidence, so it is declared, and the R1
        classification does not depend on it (see data/r1_result.md).
  DEV-4 context A's requested in-service quarter = 8, a declared campus input chosen inside R01's US range
        (1-3 years). R08's delay is measured relative to the requested date.
  DEV-5 R11+R12 give a derived central value: study 9-12 months (midpoint 10.5 months) + "typically 3 years after
        the point of order" = 15.5 quarters. This assumes immediate progression from study to order, which
        SHORTENS context B, i.e. it works against the A-vs-B gap. Dominion's two 90-day CLOA windows are omitted
        for the same conservative reason.

Quarter convention: q = 1 is the first quarter after the planning date. T_grid = q means grid power is available
from the start of quarter q. Support is {1..Q} plus an overflow state standing for "> Q".

Usage: python data/make_r1_centres.py --Q 28 --out data/r1_centres.csv
"""
import argparse
import csv

import numpy as np


def uniform(a_q, b_q, Q):
    p = np.zeros(Q + 1)
    qs = list(range(int(a_q), int(b_q) + 1))
    for q in qs:
        p[min(q, Q + 1) - 1] += 1.0 / len(qs)
    return p


def triangle(center_q, half_width_q, Q):
    """Symmetric discrete triangle on integer quarters within [c - h, c + h] (inclusive after rounding
    outward), weight proportional to (h + 1 - |q - c|). The outward rounding is documented."""
    lo = int(np.floor(center_q - half_width_q))
    hi = int(np.ceil(center_q + half_width_q))
    qs = np.arange(max(lo, 1), hi + 1)
    w = np.maximum(half_width_q + 1.0 - np.abs(qs - center_q), 0.0)
    w = w / w.sum()
    p = np.zeros(Q + 1)
    for q, wq in zip(qs, w):
        p[min(int(q), Q + 1) - 1] += wq
    return p


def stats(p, Q):
    """Censoring-robust summaries. The overflow state is never used as an exact date."""
    cdf = np.cumsum(p)
    med = int(np.searchsorted(cdf, 0.5)) + 1
    med_s = f">{Q}" if med > Q else str(med)
    within = p[:Q].sum()
    mean_within = float(np.dot(np.arange(1, Q + 1), p[:Q]) / within) if within > 0 else float("nan")
    return dict(median=med_s, p_le8=cdf[7], p_le12=cdf[11], p_gt_Q=p[Q], mean_given_le_Q=mean_within)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=28)
    ap.add_argument("--out", default="data/r1_centres.csv")
    a = ap.parse_args()
    Q = a.Q
    comps = {}

    # ---- Context A: ERCOT (z1), mixed stages (z3), 150 MW ---------------------------------------------------
    comps[("A", "R01")] = uniform(4, 12, Q)                     # rule (1): US 1-3 yr -> quarters 4..12
    req_A = 8                                                    # DEV-4
    d = 220 / 91.3125                                            # R08: mean delay 220 days = 2.41 quarters
    comps[("A", "R08")] = triangle(req_A + d, 0.5 * d, Q)        # rule (2) on the stated statistic (the delay)

    # ---- Context B: PJM-Dominion (N. Virginia) (z1), early / engineering-study stage (z3), 150 MW -----------
    comps[("B", "R02")] = uniform(4, 28, Q)                      # DEV-1: "up to 7 yr" -> [1, 7] yr -> q 4..28
    c = 10.5 / 3.0 + 12.0                                        # DEV-5: 15.5 quarters
    comps[("B", "R11+R12")] = triangle(c, 0.5 * c, Q)            # rule (2)

    centres = {}
    for ctx in ("A", "B"):
        parts = [v for (k, _), v in comps.items() if k == ctx]
        centres[ctx] = np.mean(parts, axis=0)
    centres["pooled"] = 0.5 * (centres["A"] + centres["B"])     # declared: unconditional centre for B3/B4

    # ---- Consistency check with R07 (not a centre component) --------------------------------------
    # ERCOT: officer-letter loads with 2024 in-service dates -> 55.4 % in service by Feb 2025. With requested dates
    # spread uniformly over 2024, the follow-up lag runs from about 1 to 13 months (0.33-4.33 quarters). The cohort
    # fraction implied by a delay CDF F is then the average of F over that lag range. Compute it for the R08
    # component's delay distribution.
    delay_support = np.arange(0, 12)
    comp = comps[("A", "R08")]
    delay_pmf = np.array([comp[req_A + k - 1] if req_A + k - 1 < Q + 1 else 0.0 for k in delay_support])
    lags = np.linspace(0.33, 4.33, 401)
    Fvals = [delay_pmf[: int(np.floor(x)) + 1].sum() for x in lags]
    implied = float(np.mean(Fvals))
    # Stated difference: R07 is load-weighted (or project-weighted; the NERC/ERCOT denominators differ) and covers
    # officer-letter loads only, whereas R08 covers all new large loads, so agreement is only indicative.

    support = list(range(1, Q + 1)) + [f">{Q}"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quarter"] + [f"{k}_{r}" for (k, r) in comps] + ["centre_A", "centre_B", "centre_pooled"])
        for i, q in enumerate(support):
            w.writerow([q] + [f"{comps[k][i]:.6f}" for k in comps] + [f"{centres[x][i]:.6f}" for x in ("A", "B", "pooled")])
    print(f"Q = {Q} (DEV-3)")
    for ctx in ("A", "B", "pooled"):
        s = stats(centres[ctx], Q)
        print(f"centre {ctx:6s}: median q={s['median']:>4s}  P(T<=8)={s['p_le8']:.3f}  P(T<=12)={s['p_le12']:.3f}  "
              f"P(T>Q)={s['p_gt_Q']:.3f}  mean|T<=Q={s['mean_given_le_Q']:.2f}")
    sA, sB = stats(centres["A"], Q), stats(centres["B"], Q)
    print(f"A-vs-B gap in mean|T<=Q (no censoring in either centre if P(T>Q)=0): "
          f"{sB['mean_given_le_Q'] - sA['mean_given_le_Q']:.2f} quarters")
    print(f"R07 consistency check: implied officer-letter cohort fraction from the R08 delay component = {implied:.3f} "
          f"(observed 0.554; indicative only)")
    print("written", a.out)


if __name__ == "__main__":
    main()
