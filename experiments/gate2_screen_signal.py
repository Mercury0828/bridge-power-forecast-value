"""Gate 2 screen 1: upper bounds on the value of a utility signal, using the existing model and
before any signal-tree or DRO development. EXPLORATORY; not used for any claim.

For context z, the marginal P_T is the real-evidence centre, preserved. Quantities:
  v0      = v(P_T): the best no-signal policy (SAA on P_T);
  EVPI    = v0 − Σ_t P(t) v(δ_t): perfect knowledge of T at time 0;
  V0(r)   = v0 − Σ_y P(y) v(P(T | y)): the value of a 3-category signal revealed at time 0. The categories are the
            terciles of P_T. The kernel is K(y | t) = r if y is t's tercile, else (1 − r) / 2, for r in RELIAB
            (r = 1/3 is uninformative). A signal arriving later, at the end of the study, is worth at most V0(r).
Economic cases: main (must-energize) and zero salvage. Solver: gap 1e-6; incumbent values (screen only).

    python experiments/gate2_screen_signal.py  ->  data/gate2_screen_signal.json (+ stdout)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RELIAB = (1 / 3, 0.6, 0.8, 0.95, 1.0)
ECONS = ("main", "salv0")


def _solve(args):
    from model.bridge import make_params, Model
    from experiments import p1a_core as core
    econ, ctx, tag, kind, arg = args
    P = make_params(ctx, **core.ECON[econ])
    m = Model(P, gap=1e-6, time_limit=3600.0)
    # tiebreak=False: the tie-break stage's objective is not v(delta_t) (Phase 1f S4: the stored EVPI was invalid)
    st = m.solve_det(int(arg), tiebreak=False) if kind == "det" else m.solve_saa(np.array(arg), tiebreak=False)
    return (econ, ctx, tag), (float(st["obj"]) if st["ok"] else float("nan"))


def terciles(p):
    cdf = np.cumsum(p)
    cat = np.where(cdf <= 1 / 3 + 1e-12, 0, np.where(cdf <= 2 / 3 + 1e-12, 1, 2))
    return cat


def main():
    from model import evidence as ev
    jobs, meta = [], {}
    for ctx in ("A", "B"):
        p = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        cat = terciles(p)
        sup = [i for i in range(len(p)) if p[i] > 0]
        post = {}
        for r in RELIAB:
            K = np.array([[r if cat[t] == y else (1 - r) / 2 for t in range(len(p))] for y in range(3)])
            py = K @ p
            post[r] = [(float(py[y]), ev.clean(K[y] * p / py[y])) for y in range(3) if py[y] > 1e-12]
        meta[ctx] = dict(p=p, sup=sup, post=post)
        for econ in ECONS:
            jobs.append((econ, ctx, "v0", "saa", p.tolist()))
            for i in sup:
                jobs.append((econ, ctx, f"det|{i + 1}", "det", i + 1))
            for r in RELIAB:
                for y, (py, q) in enumerate(post[r]):
                    jobs.append((econ, ctx, f"sig|{r:.3f}|{y}", "saa", q.tolist()))
    with mp.get_context("spawn").Pool(processes=24) as pool:
        V = dict(pool.map(_solve, jobs))
    out = {}
    for econ in ECONS:
        for ctx in ("A", "B"):
            mt = meta[ctx]
            v0 = V[(econ, ctx, "v0")]
            evpi = v0 - sum(mt["p"][i] * V[(econ, ctx, f"det|{i + 1}")] for i in mt["sup"])
            vals = {}
            for r in RELIAB:
                vals[f"{r:.3f}"] = v0 - sum(py * V[(econ, ctx, f"sig|{r:.3f}|{y}")]
                                            for y, (py, q) in enumerate(mt["post"][r]))
            out[f"{econ}|{ctx}"] = dict(v0=v0, EVPI=evpi, V0=vals)
            print(f"{econ:6s} {ctx}: v0 {v0:8.2f} | EVPI {evpi:6.2f} ({100 * evpi / v0:.2f} %) | time-0 signal value by "
                  f"reliability: " + ", ".join(f"r={k}: {v:+.2f} ({100 * v / v0:.2f} %)" for k, v in vals.items()))
    (ROOT / "data" / "gate2_screen_signal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
