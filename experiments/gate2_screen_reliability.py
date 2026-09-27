"""Gate 2 screen 2: the signal-reliability mechanism on real policy-cost vectors. Time-0
signal; EXPLORATORY; not used for any claim.

Setting (for one context and economic case):
- The marginal P_T is the real-evidence centre and is preserved.
- A 3-category signal y (the terciles of P_T) with the kernel K_r(y | t) = r if y is t's tercile, else (1 − r) / 2.
- The true reliability r is unknown in Γ = [R_LO, R_HI].

Policies: π_y(r̂) = SAA on P_{r̂}(T | y) for r̂ in RHAT; "ignore" = SAA on P_T for every y.
Evaluation: J(plan | r) = Σ_y Σ_t P(t) K_r(y | t) C_{π_y}(t). Planners:
- oracle(r): plans with the true r;
- predictive: a uniform prior on Γ. K is linear in r, so the predictive kernel is K at the mean of Γ;
- robust: chooses per-y policies from the candidate set {π_y(r̂), ignore} to minimize max over r in RGRID of
  J(plan | r). This is an exact min–max over that candidate set, a restriction of the full robust problem.
Reported per true r: V_signal(r) = J(ignore) − J(oracle(r)); G_practical(r) = J(predictive | r) − J(oracle(r));
and robust − predictive.
The EVPI is also recomputed with the primary objective of the deterministic solves.

    python experiments/gate2_screen_reliability.py  ->  data/gate2_screen_reliability.json (+ stdout)
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R_LO, R_HI = 0.6, 0.95
RHAT = (0.6, 0.7, 0.775, 0.85, 0.95)
RGRID = np.round(np.linspace(R_LO, R_HI, 8), 4)
ECONS = ("main", "salv0")


def _solve(args):
    from model.bridge import make_params, Model, polish
    from experiments import p1a_core as core
    econ, ctx, tag, kind, arg = args
    P = make_params(ctx, **core.ECON[econ])
    m = Model(P, gap=1e-6, time_limit=3600.0)
    if kind == "det":
        st = m.solve_det(int(arg), tiebreak=False)
        return (econ, ctx, tag), float(st["obj"]) if st["ok"] else float("nan")
    st = m.solve_saa(np.array(arg))
    if not st["ok"]:
        return (econ, ctx, tag), None
    pm, pst = polish(P, m.spine_values())
    return (econ, ctx, tag), pm.C_values().tolist()


def kernel(p, r):
    cdf = np.cumsum(p)
    cat = np.where(cdf <= 1 / 3 + 1e-12, 0, np.where(cdf <= 2 / 3 + 1e-12, 1, 2))
    return np.array([[r if cat[t] == y else (1 - r) / 2 for t in range(len(p))] for y in range(3)])


def main():
    from model import evidence as ev
    jobs, P_T = [], {}
    for ctx in ("A", "B"):
        p = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        P_T[ctx] = p
        for econ in ECONS:
            jobs.append((econ, ctx, "ignore", "saa", p.tolist()))
            for i in range(len(p)):
                if p[i] > 0:
                    jobs.append((econ, ctx, f"det|{i + 1}", "det", i + 1))
            for rh in RHAT:
                K = kernel(p, rh)
                for y in range(3):
                    jobs.append((econ, ctx, f"pi|{rh}|{y}", "saa", ev.clean(K[y] * p / (K[y] @ p)).tolist()))
    with mp.get_context("spawn").Pool(processes=24) as pool:
        V = dict(pool.map(_solve, jobs))
    out = {}
    for econ in ECONS:
        for ctx in ("A", "B"):
            p = P_T[ctx]
            Cign = np.array(V[(econ, ctx, "ignore")])
            v0 = float(p @ Cign)
            evpi = v0 - sum(p[i] * V[(econ, ctx, f"det|{i + 1}")] for i in range(len(p)) if p[i] > 0)
            cand = {rh: [np.array(V[(econ, ctx, f"pi|{rh}|{y}")]) for y in range(3)] for rh in RHAT}

            def J(Cy, r):
                K = kernel(p, r)
                return float(sum((K[y] * p) @ Cy[y] for y in range(3)))

            # per-y candidate set: ignore or pi_y(rh)
            per_y = [[("ign", Cign)] + [(f"r{rh}", cand[rh][y]) for rh in RHAT] for y in range(3)]
            best, best_combo = np.inf, None
            for combo in itertools.product(*per_y):
                worst = max(J([c[1] for c in combo], r) for r in RGRID)
                if worst < best:
                    best, best_combo = worst, combo
            rob = [c[1] for c in best_combo]
            pred = cand[0.775]
            rows = []
            for r in RGRID:
                orc = min(J(cand[rh], r) for rh in RHAT)          # the grid-restricted oracle (plans with r̂ nearest r)
                jo = J(cand[min(RHAT, key=lambda x: abs(x - r))], r)
                rows.append(dict(r=float(r), J_ignore=J([Cign] * 3, r), J_oracle=jo, J_best_rhat=orc,
                                 J_pred=J(pred, r), J_robust=J(rob, r)))
            out[f"{econ}|{ctx}"] = dict(v0=v0, EVPI=evpi, robust_combo=[c[0] for c in best_combo], rows=rows)
            print(f"== {econ} {ctx}: v0 {v0:.2f} | EVPI {evpi:.2f} ({100 * evpi / v0:.2f} %) | robust per-y choice "
                  f"{[c[0] for c in best_combo]}")
            print("   true r | V_signal = ign − oracle | G_practical = pred − oracle | robust − pred | robust vs ignore")
            for rw in rows:
                print(f"   {rw['r']:.3f}  | {rw['J_ignore'] - rw['J_best_rhat']:+8.2f} | "
                      f"{rw['J_pred'] - rw['J_best_rhat']:+8.2f} | {rw['J_robust'] - rw['J_pred']:+8.2f} | "
                      f"{rw['J_robust'] - rw['J_ignore']:+8.2f}")
    (ROOT / "data" / "gate2_screen_reliability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
