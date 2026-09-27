"""Phase-1a real-evidence centres (S2 of the Phase-1a plan; the R1 files stay frozen).

Applies the clock-aligned estimator of model/evidence.py to the evidence as stated (R08; R11, R12 and R02), and
reports:
  - the centres p~_A, p~_B and p~_pooled;
  - their stage bounds;
  - the posterior calibration family under the registered observation model (radii at 50 % and 80 %);
  - the R07 consistency check that brackets context A's dispersion range.
No cost or policy outcome is computed here.

    python data/make_p1a_centres.py  ->  data/p1a_centres.csv, data/p1a_centres.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import evidence as ev  # noqa: E402

N_CAL = 400


def seed_of(ctx, reports):
    """The calibration family is a deterministic function of the training information (context + reports)."""
    s = json.dumps([ctx, reports], sort_keys=True)
    return int.from_bytes(__import__("hashlib").sha256(s.encode()).digest()[:8], "little")


def summary(p):
    t = np.arange(1, ev.Q + 2)
    cdf = np.cumsum(p)
    return dict(mean=float(t @ p), median=int(np.searchsorted(cdf, 0.5)) + 1,
                p5=int(np.searchsorted(cdf, 0.05 - 1e-12)) + 1, p95=int(np.searchsorted(cdf, 0.95 - 1e-12)) + 1,
                p_overflow=float(p[-1]))


def main():
    out, cen = {}, {}
    for ctx in ("A", "B"):
        rep = ev.REAL_REPORTS[ctx]
        c = ev.estimate(ctx, rep)
        fam = ev.calibration_family(ctx, rep, N_CAL, np.random.default_rng(seed_of(ctx, rep)))
        cen[ctx], cen[f"{ctx}_fam"] = c, fam
        out[ctx] = dict(reports=rep, centre=c.tolist(), summary=summary(c), radii=ev.radii(c, fam),
                        p_mix=fam.mean(axis=0).tolist(), summary_mix=summary(fam.mean(axis=0)),
                        bounds=ev.BOUNDS[ctx], cv_hat=ev.CV_HAT[ctx], cv_range=ev.CV_RANGE[ctx])
    pooled = 0.5 * (cen["A"] + cen["B"])
    fam_p = 0.5 * (cen["A_fam"] + cen["B_fam"])                   # context-blind: the same procedure on the pool
    out["pooled"] = dict(centre=pooled.tolist(), summary=summary(pooled), radii=ev.radii(pooled, fam_p),
                         bounds=ev.BOUNDS["pooled"])
    # R07 consistency (context A's dispersion range): officer-letter loads with 2024 in-service dates, 55.4 % in
    # service by Feb 2025. Follow-up lags 0.33-4.33 quarters (requested dates spread over 2024).
    lags = np.linspace(0.33, 4.33, 401)
    r07 = {}
    for cv in (ev.CV_RANGE["A"][0], ev.CV_HAT["A"], ev.CV_RANGE["A"][1]):
        k = 1.0 / cv ** 2
        r07[cv] = float(np.mean(stats.gamma.cdf(lags, k, scale=ev.MU_D_A / k)))
    out["R07_check"] = dict(observed=0.554, implied_by_cv=r07)

    (ROOT / "data" / "p1a_centres.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(ROOT / "data" / "p1a_centres.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "centre_A", "centre_B", "centre_pooled", "p_mix_A", "p_mix_B"])
        for i in range(ev.Q + 1):
            w.writerow([i + 1 if i < ev.Q else f">{ev.Q}", f"{cen['A'][i]:.6f}", f"{cen['B'][i]:.6f}",
                        f"{pooled[i]:.6f}", f"{out['A']['p_mix'][i]:.6f}", f"{out['B']['p_mix'][i]:.6f}"])
    for z in ("A", "B", "pooled"):
        s = out[z]["summary"]
        print(f"{z:6s} mean {s['mean']:.2f} median {s['median']} P5-P95 [{s['p5']}, {s['p95']}] "
              f"P(T>Q) {s['p_overflow']:.4f} radii eps50 {out[z]['radii'][0.5]:.3f} eps80 {out[z]['radii'][0.8]:.3f} "
              f"bounds {out[z]['bounds']}")
    print("R07 implied cohort fraction by CV:", {k: round(v, 3) for k, v in r07.items()}, "observed 0.554")


if __name__ == "__main__":
    main()
