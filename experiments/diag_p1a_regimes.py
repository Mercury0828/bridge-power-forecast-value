"""EXPLORATORY regime map after the Phase-1a FAIL (owner gate D-019: method repair). Not pre-registered and not used for
any claim. Its only purpose is to locate economic regimes where distributional ambiguity has consequences, before a
new test is designed.

For each economic regime and context:
  - Training centre: the real-evidence plug-in centre (model/evidence.py, REAL_REPORTS); its posterior radius eps80.
  - Truths: the Phase-1a latent grid (both CV ends x shifts -2..+2).
  - Methods: CSP; CDRO at eps80, 1.0 and 2.0 (structural bounds); BMIX (posterior predictive).
    TMIX is a REFERENCE ONLY: SP on the uniform mixture of the grid truths. It is the Bayes policy if the grid were
    the prior, so it uses evaluation knowledge and is not a method.
  - The oracle on each truth.
Output: mean over the 10 truths, and by shift, of J_CSP − J_m and of the room J_CSP − v.

    python experiments/diag_p1a_regimes.py  ->  data/diag_p1a_regimes.json (+ stdout)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = dict(p_R=25.0, cap_R=100.0, delay_mult=1.0, residence_on=True, backup_rating="DCC", backup_sym=True,
            age_salvage=True, backup_ramp=True)
REGIMES = {"must": dict(BASE, service_hard=True),
           "soft_P6": dict(BASE, service_hard=False, vod_base=0.471),
           "soft_AI_low": dict(BASE, service_hard=False, vod_base=1.4290),
           "soft_AI_central": dict(BASE, service_hard=False, vod_base=1.7775),
           "soft_AI_high": dict(BASE, service_hard=False, vod_base=2.1718)}
SHIFTS = (-2, -1, 0, 1, 2)


def _job(args):
    from model.bridge import make_params, Model, polish
    regime, ctx, name, kind, p, eps, allowed, lohi = args
    P = make_params(ctx, **REGIMES[regime])
    m = Model(P)
    p = np.array(p)
    if kind == "saa":
        st = m.solve_saa(p)
    else:
        st = m.solve_dro(p, eps, allowed=allowed)
    if not st["ok"]:
        return (regime, ctx, name), None
    pm, pst = polish(P, m.spine_values())
    return (regime, ctx, name), (pm.C_values().tolist() if pst["ok"] else None)


def main():
    from model import evidence as ev
    jobs, truths = [], {}
    for ctx in ("A", "B"):
        c = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        fam = ev.calibration_family(ctx, ev.REAL_REPORTS[ctx], 400, np.random.default_rng(1))
        e80 = ev.radii(c, fam)[0.8]
        allowed = ev.allowed_states(ctx)
        tr = {(cv, s): ev.truth(ctx, cv, s) for cv in ev.CV_RANGE[ctx] for s in SHIFTS}
        truths[ctx] = tr
        tmix = ev.clean(np.mean(list(tr.values()), axis=0))
        for regime in REGIMES:
            jobs.append((regime, ctx, "CSP", "saa", c.tolist(), None, None, None))
            jobs.append((regime, ctx, "BMIX", "saa", ev.clean(fam.mean(axis=0)).tolist(), None, None, None))
            jobs.append((regime, ctx, "TMIX(ref)", "saa", tmix.tolist(), None, None, None))
            for tag, e in (("CDRO_e80", e80), ("CDRO_e1", 1.0), ("CDRO_e2", 2.0)):
                jobs.append((regime, ctx, tag, "dro", c.tolist(), e, allowed, None))
            for (cv, s), p in tr.items():
                jobs.append((regime, ctx, f"oracle|{cv}|{s}", "saa", p.tolist(), None, None, None))
    with mp.get_context("spawn").Pool(processes=24) as pool:
        res = dict(pool.map(_job, jobs))
    out = {}
    for regime in REGIMES:
        for ctx in ("A", "B"):
            C = {k[2]: np.array(v) for k, v in res.items() if k[0] == regime and k[1] == ctx and v is not None}
            if "CSP" not in C:
                print(regime, ctx, "CSP failed")
                continue
            rows = []
            for (cv, s), p in truths[ctx].items():
                ok = f"oracle|{cv}|{s}"
                row = dict(cv=cv, shift=s, J_CSP=float(p @ C["CSP"]),
                           oracle=float(p @ C[ok]) if ok in C else float("nan"))
                for m in ("BMIX", "TMIX(ref)", "CDRO_e80", "CDRO_e1", "CDRO_e2"):
                    if m in C:
                        row[m] = float(p @ C[m])
                rows.append(row)
            out[f"{regime}|{ctx}"] = rows
            J = np.mean([r["J_CSP"] for r in rows])
            room = np.mean([r["J_CSP"] - r["oracle"] for r in rows])
            gains = {m: np.mean([r["J_CSP"] - r[m] for r in rows if m in r])
                     for m in ("BMIX", "TMIX(ref)", "CDRO_e80", "CDRO_e1", "CDRO_e2")}
            print(f"{regime:16s} {ctx}: E[J_CSP] {J:7.1f} | room {room:6.2f} ({100 * room / J:5.2f} %) | gains vs CSP: "
                  + ", ".join(f"{m} {g:+6.2f} ({100 * g / J:+5.2f} %)" for m, g in gains.items()))
            for s in SHIFTS:
                R = [r for r in rows if r["shift"] == s]
                print(f"      shift {s:+d}: room {np.mean([r['J_CSP'] - r['oracle'] for r in R]):7.2f} | "
                      + ", ".join(f"{m} {np.mean([r['J_CSP'] - r[m] for r in R if m in r]):+7.2f}"
                                  for m in ("CDRO_e80", "CDRO_e1", "CDRO_e2", "TMIX(ref)")))
    (ROOT / "data" / "diag_p1a_regimes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
