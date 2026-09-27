"""EXPLORATORY protection-premium frontier (owner gate D-019; route 3 of data/diag_p1a.md). Not pre-registered and not
used for any claim. It checks, on the real-evidence centres and the Phase-1a truth grid (main economics), whether a
tail-consistent ambiguity set buys late-side protection more cheaply than the alternatives.

Families, each traced over a parameter:
  W1       : W1-DRO with structural lower bounds, eps in EPS
  W1band   : W1 intersected with posterior survival bands from training information: u_q = the 95th and l_q = the 5th
             percentile of S_q over the posterior draws, for every q; eps in EPS
  BOX      : box-robust over [P5, P_k] of the posterior predictive law, k in (0.90, 0.95, 0.99)
  SHIFT    : SP on the centre shifted later by d quarters, d in (1, 2) (a naive hedge)
Per policy:
  - the premium = grid-average J − grid-average J_CSP;
  - the protection = J_CSP − J at the late truths (shift +1 and +2, both CV ends, averaged);
  - the worst truth = max over the 10 truths of J.

    python experiments/diag_p1a_frontier.py  ->  data/diag_p1a_frontier.json (+ stdout)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ECON = dict(p_R=25.0, cap_R=100.0, delay_mult=1.0, residence_on=True, backup_rating="DCC", backup_sym=True,
            age_salvage=True, backup_ramp=True, service_hard=True)
EPS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
SHIFTS = (-2, -1, 0, 1, 2)


def _job(args):
    from model.bridge import make_params, Model, polish
    ctx, name, kind, p, kw = args
    P = make_params(ctx, **ECON)
    m = Model(P)
    p = np.array(p)
    st = m.solve_saa(p) if kind == "saa" else (m.solve_box(**kw) if kind == "box" else m.solve_dro(p, **kw))
    if not st["ok"]:
        return (ctx, name), None
    pm, pst = polish(P, m.spine_values())
    return (ctx, name), (pm.C_values().tolist() if pst["ok"] else None)


def shifted(p, d):
    q = np.zeros_like(p)
    q[d:-1] = p[:-1 - d]
    q[-1] = p[-1] + p[-1 - d:-1].sum()
    return q


def main():
    from model import evidence as ev
    from model.bridge import percentile_quarter
    jobs, truths = [], {}
    for ctx in ("A", "B"):
        c = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        fam = ev.calibration_family(ctx, ev.REAL_REPORTS[ctx], 400, np.random.default_rng(1))
        pmix = ev.clean(fam.mean(axis=0))
        S = np.array([[p[q:].sum() for q in range(1, ev.Q + 1)] for p in fam])        # S_q, q = 1..Q
        su = {q: float(np.quantile(S[:, q - 1], 0.95)) for q in range(1, ev.Q + 1)}
        sl = {q: float(np.quantile(S[:, q - 1], 0.05)) for q in range(1, ev.Q + 1)}
        Sc = {q: float(c[q:].sum()) for q in range(1, ev.Q + 1)}                          # keep the centre inside
        su = {q: max(su[q], Sc[q]) for q in su}
        sl = {q: min(sl[q], Sc[q]) for q in sl}
        allowed = ev.allowed_states(ctx)
        truths[ctx] = {(cv, s): ev.truth(ctx, cv, s) for cv in ev.CV_RANGE[ctx] for s in SHIFTS}
        jobs.append((ctx, "CSP", "saa", c.tolist(), None))
        for e in EPS:
            jobs.append((ctx, f"W1|{e}", "dro", c.tolist(), dict(eps=e, allowed=allowed)))
            jobs.append((ctx, f"W1band|{e}", "dro", c.tolist(), dict(eps=e, allowed=allowed, surv_upper=su,
                                                                     surv_lower=sl)))
        for k in (0.90, 0.95, 0.99):
            jobs.append((ctx, f"BOX|{k}", "box", None, dict(lo=percentile_quarter(pmix, 0.05),
                                                           hi=percentile_quarter(pmix, k))))
        for d in (1, 2):
            jobs.append((ctx, f"SHIFT|{d}", "saa", ev.clean(shifted(c, d)).tolist(), None))
    with mp.get_context("spawn").Pool(processes=24) as pool:
        res = dict(pool.map(_job, jobs))
    out = {}
    for ctx in ("A", "B"):
        C = {k[1]: np.array(v) for k, v in res.items() if k[0] == ctx and v is not None}
        tr = truths[ctx]
        J = {m: {t: float(p @ Cm) for t, p in tr.items()} for m, Cm in C.items()}
        base_avg = np.mean(list(J["CSP"].values()))
        late = [t for t in tr if t[1] > 0]
        print(f"== {ctx}: grid-average J_CSP {base_avg:.1f}; late-truth J_CSP "
              f"{np.mean([J['CSP'][t] for t in late]):.1f}; worst-truth J_CSP {max(J['CSP'].values()):.1f}")
        rows = []
        for m in C:
            prem = np.mean(list(J[m].values())) - base_avg
            prot = np.mean([J["CSP"][t] - J[m][t] for t in late])
            worst = max(J[m].values())
            rows.append(dict(method=m, premium=prem, late_protection=prot, worst_truth=worst))
            print(f"  {m:14s} premium {prem:+8.2f} | late protection {prot:+8.2f} | worst truth {worst:8.1f}")
        out[ctx] = rows
    (ROOT / "data" / "diag_p1a_frontier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
