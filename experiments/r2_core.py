"""R2 core: compute every method's polished scenario-cost vector for one sweep item (data/r2_preregistration.md).

run_item(ctx, p_R, cap_R, delay_mult, residence_on, eps_mult=1.0) -> dict (JSON-serializable)
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bridge import (make_params, Model, policy_rolling_det, polish, recompute_costs, radius,  # noqa: E402
                          worst_case_expectation, percentile_quarter, ASSETS)

N_RECORDS = {"A": 2, "B": 2, "pooled": 4}          # R1 v2 records per centre


def load_centres(path=ROOT / "data" / "r1_centres.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for k in ("A", "B", "pooled"):
        p = np.array([float(r[f"centre_{k}"]) for r in rows])
        out[k] = p / p.sum()          # CSV rounding leaves sums of 1.000002-1.000004
    return out


def _summ(P, m):
    sv = m.spine_values()
    return dict(n_GE=sv.get("n|GE"), n_DG=sv.get("n|DG"), z=sv.get("z"),
                orders_MW={j: round(sum(sv[f"x|{j}|{k}"] for k in range(P.Q + 1)), 3) for j in ASSETS},
                root_MW={j: round(sv[f"x|{j}|0"], 3) for j in ASSETS},
                filings=[k for k in range(P.Q) if sv[f"f|{k}"] == 1],
                rent_NR_MWq=round(sum(sv[f"RNR|{q}"] for q in range(1, P.Q + 1)), 3),
                rent_ST_MWq=round(sum(sv[f"RST|{q}"] for q in range(1, P.Q + 1)), 3),
                served_IT_MWq=round(sum(sv[f"S|{q}"] for q in range(1, P.Q + 1)), 3),
                demand_IT_MWq=round(sum(P.L(q) for q in range(1, P.Q + 1)), 3))


def _finish(P, m_policy, st, name, t0):
    """Polish the branches of a solved policy and return its cost vector plus diagnostics."""
    if m_policy is None or not st.get("ok", False):
        return dict(name=name, ok=False, status=st)
    spine = m_policy.spine_values()
    mp, stp = polish(P, spine)
    C_pol = mp.C_values()
    C_chk = recompute_costs(P, mp)
    return dict(name=name, ok=bool(stp["ok"]), status=st, polish_status=stp,
                tiebreak_ok=st.get("tiebreak_ok"),
                C=C_pol.tolist(), C_unpolished=m_policy.C_values().tolist(),
                bookkeeping_maxabs=float(np.max(np.abs(C_pol - C_chk))),
                summary=_summ(P, m_policy))


TOY = dict(Q=8, it_ramp=((2, 20.0), (5, 40.0)), lead={"GE": 3, "DG": 2, "ABS": 1, "BESS": 1}, L_permit=2)


def toy_centres():
    p = np.array([0.0, 0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05])
    q = np.array([0.0, 0.0, 0.05, 0.05, 0.10, 0.20, 0.25, 0.20, 0.15])
    return {"A": p, "B": q, "pooled": 0.5 * (p + q)}


def run_item(ctx, p_R, cap_R, delay_mult, residence_on, eps_mult=1.0, methods=None, toy=False):
    cen = toy_centres() if toy else load_centres()
    P = make_params(ctx, p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on, **(TOY if toy else {}))
    eps_z = radius(cen[ctx], N_RECORDS[ctx]) * eps_mult
    eps_pool = radius(cen["pooled"], N_RECORDS["pooled"]) * eps_mult
    methods = methods or ["CDRO", "B4", "B3", "CSAA", "B1", "B7"]
    out = dict(ctx=ctx, p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on, eps_mult=eps_mult,
               eps_z=eps_z, eps_pooled=eps_pool, methods={})
    for name in methods:
        t0 = time.time()
        if name == "CDRO":
            m = Model(P); st = m.solve_dro(cen[ctx], eps_z)
            res = _finish(P, m, st, name, t0)
            if res["ok"]:  # V3 duality check on the unpolished policy's own cost vector
                wc = worst_case_expectation(m.C_values(), cen[ctx], eps_z)
                res["dual_primal_gap"] = float(abs(wc - st.get("primary_obj", st["obj"])))
        elif name == "B4":
            m = Model(P); st = m.solve_dro(cen["pooled"], eps_pool)
            res = _finish(P, m, st, name, t0)
            if res["ok"]:
                wc = worst_case_expectation(m.C_values(), cen["pooled"], eps_pool)
                res["dual_primal_gap"] = float(abs(wc - st.get("primary_obj", st["obj"])))
        elif name == "B3":
            m = Model(P); st = m.solve_saa(cen["pooled"]); res = _finish(P, m, st, name, t0)
        elif name == "CSAA":
            m = Model(P); st = m.solve_saa(cen[ctx]); res = _finish(P, m, st, name, t0)
        elif name == "B1":
            m, stats = policy_rolling_det(P, cen[ctx])
            compact = [dict(k=s.get("k"), T_hat=s.get("T_hat"), ok=s.get("ok"), primary_status=s.get("primary_status"),
                            primary_gap=s.get("primary_gap"), tiebreak_ok=s.get("tiebreak_ok"),
                            tiebreak_status=s.get("tiebreak_status")) for s in stats]
            st = dict(ok=m is not None and all(s.get("ok", False) for s in stats), rolling=compact,
                      n_solves=len(stats), tiebreak_ok=all(s.get("tiebreak_ok") is not False for s in stats),
                      max_primary_gap=max((s.get("primary_gap") or 0.0) for s in stats))
            res = _finish(P, m, st, name, t0)
        elif name == "B7":
            m = Model(P); st = m.solve_det(P.Q + 1); res = _finish(P, m, st, name, t0)
        else:
            raise ValueError(name)
        out["methods"][name] = res
    return out


if __name__ == "__main__":
    import json
    r = run_item(sys.argv[1] if len(sys.argv) > 1 else "A", 25.0, 100.0, 1.0, True)
    print(json.dumps({k: (v if k != "methods" else {n: {kk: vv for kk, vv in d.items() if kk not in ("C", "C_unpolished")}
                                                     for n, d in v.items()}) for k, v in r.items()}, indent=1)[:6000])


# ---------------------------------------------------------------------------------------------------------------
# R2b: leave-one-record-out (data/r2b_preregistration.md)
# ---------------------------------------------------------------------------------------------------------------
COMPONENTS = {"A": ("A_R01", "A_R08"), "B": ("B_R02", "B_R11+R12")}


def load_components(path=ROOT / "data" / "r1_centres.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for z, names in COMPONENTS.items():
        for n in names:
            p = np.array([float(r[n]) for r in rows])
            out[n] = p / p.sum()
    return out


def run_item_loro(ctx, p_R, cap_R, delay_mult, residence_on, toy=False):
    comps = load_components()
    P = make_params(ctx, p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on)
    out = dict(kind="loro", ctx=ctx, p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on,
               eps_mult=1.0, train={})
    for r in COMPONENTS[ctx]:
        ph = comps[r]
        eps = radius(ph, 1)
        lo, hi = percentile_quarter(ph, 0.05), percentile_quarter(ph, 0.95)
        res = {}
        t0 = time.time()
        m = Model(P); st = m.solve_dro(ph, eps); res["CDRO"] = _finish(P, m, st, "CDRO", t0)
        m = Model(P); st = m.solve_saa(ph); res["CSAA"] = _finish(P, m, st, "CSAA", t0)
        m, stats = policy_rolling_det(P, ph)
        compact = [dict(k=s.get("k"), T_hat=s.get("T_hat"), ok=s.get("ok"), primary_gap=s.get("primary_gap"),
                        tiebreak_ok=s.get("tiebreak_ok")) for s in stats]
        st = dict(ok=m is not None and all(s.get("ok", False) for s in stats), rolling=compact,
                  tiebreak_ok=all(s.get("tiebreak_ok") is not False for s in stats))
        res["B1"] = _finish(P, m, st, "B1", t0)
        m = Model(P); st = m.solve_box(lo, hi); res["B5"] = _finish(P, m, st, "B5", t0)
        out["train"][r] = dict(eps=eps, box=[lo, hi], methods=res)
    return out
