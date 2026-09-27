"""Gate 2 screen 3: does staged grid connection create consequential room? EXPLORATORY; not used
for any claim.

For each staged configuration (partial capacity stage_frac of peak demand, for stage_len quarters after initial
energization) and each context, with the main economics, the real-evidence centre and the Phase-1a truth grid:
- room      = mean over truths of (J_CSP − oracle);
- Bayes     = J_CSP − J_TMIX on the grid (the TMIX bound: the most any planner can gain without new information);
- V_sig(r)  = the value of a time-0 3-category (tercile) signal with reliability r, for r = 0.8 and r = 1.0.

    python experiments/gate2_screen_staged.py  ->  data/gate2_screen_staged.json (+ stdout)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = {"full": dict(), "L4_f50": dict(stage_len=4, stage_frac=0.5), "L4_f25": dict(stage_len=4, stage_frac=0.25),
           "L8_f50": dict(stage_len=8, stage_frac=0.5), "L8_f25": dict(stage_len=8, stage_frac=0.25)}
SHIFTS = (-2, -1, 0, 1, 2)


def _solve(args):
    from model.bridge import make_params, Model, polish
    from experiments import p1a_core as core
    cfg, ctx, tag, p = args
    P = make_params(ctx, **dict(core.ECON["main"], **CONFIGS[cfg]))
    m = Model(P, gap=1e-5, time_limit=3600.0)
    st = m.solve_saa(np.array(p), tiebreak=False)
    if not st["ok"]:
        return (cfg, ctx, tag), None
    pm, pst = polish(P, m.spine_values())
    return (cfg, ctx, tag), dict(C=pm.C_values().tolist(), obj=float(st["obj"]))


def kernel(p, r):
    cdf = np.cumsum(p)
    cat = np.where(cdf <= 1 / 3 + 1e-12, 0, np.where(cdf <= 2 / 3 + 1e-12, 1, 2))
    return np.array([[r if cat[t] == y else (1 - r) / 2 for t in range(len(p))] for y in range(3)])


def main():
    from model import evidence as ev
    jobs, meta = [], {}
    for ctx in ("A", "B"):
        c = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        tr = {(cv, s): ev.truth(ctx, cv, s) for cv in ev.CV_RANGE[ctx] for s in SHIFTS}
        tmix = ev.clean(np.mean(list(tr.values()), axis=0))
        meta[ctx] = dict(c=c, tr=tr)
        for cfg in CONFIGS:
            jobs.append((cfg, ctx, "CSP", c.tolist()))
            jobs.append((cfg, ctx, "TMIX", tmix.tolist()))
            for (cv, s), p in tr.items():
                jobs.append((cfg, ctx, f"orc|{cv}|{s}", p.tolist()))
            for r in (0.8, 1.0):
                K = kernel(c, r)
                for y in range(3):
                    jobs.append((cfg, ctx, f"sig|{r}|{y}", ev.clean(K[y] * c / (K[y] @ c)).tolist()))
    with mp.get_context("spawn").Pool(processes=20) as pool:
        V = dict(pool.map(_solve, jobs))
    out = {}
    for ctx in ("A", "B"):
        c, tr = meta[ctx]["c"], meta[ctx]["tr"]
        for cfg in CONFIGS:
            g = lambda tag: V.get((cfg, ctx, tag))                               # noqa: E731
            if g("CSP") is None or g("TMIX") is None:
                print(cfg, ctx, "failed")
                continue
            Ccsp, Ctm = np.array(g("CSP")["C"]), np.array(g("TMIX")["C"])
            Jcsp = np.mean([p @ Ccsp for p in tr.values()])
            Jtm = np.mean([p @ Ctm for p in tr.values()])
            room = np.mean([p @ Ccsp - g(f"orc|{cv}|{s}")["obj"] for (cv, s), p in tr.items()])
            v0 = float(c @ Ccsp)
            vs = {}
            for r in (0.8, 1.0):
                K = kernel(c, r)
                vs[r] = v0 - sum(float(K[y] @ c) * g(f"sig|{r}|{y}")["obj"] for y in range(3))
            out[f"{cfg}|{ctx}"] = dict(J_CSP_grid=float(Jcsp), room=float(room), bayes=float(Jcsp - Jtm), v0=v0,
                                       V_sig={str(k): float(x) for k, x in vs.items()})
            print(f"{cfg:7s} {ctx}: J_CSP(grid) {Jcsp:7.1f} | room {room:6.2f} ({100 * room / Jcsp:.2f} %) | Bayes bound "
                  f"{Jcsp - Jtm:+6.2f} | V_sig r=0.8 {vs[0.8]:+6.2f} ({100 * vs[0.8] / v0:.2f} %), perfect {vs[1.0]:+6.2f} "
                  f"({100 * vs[1.0] / v0:.2f} %)")
    (ROOT / "data" / "gate2_screen_staged.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
