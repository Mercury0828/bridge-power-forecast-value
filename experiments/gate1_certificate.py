"""Gate 1: certify the Phase-1a obstruction on the actual repeated-estimation endpoint.

Information-matched lower bound.
- In the Phase-1a experiment, T depends only on (dispersion level, campus shift). The reports D depend only on the
  level and on sampling noise, and the shift is independent of D.
- Hence E[C | D, level] = E[C | level] = E_{M_level}[C], with M_level the equal-weight mixture of the five shifted truths
  at that level.
- Conditioning on the level as well can only lower the Bayes risk. So

      J_info* >= mean over levels of  min over pi of E_{M_level}[C_pi]  >=  mean over levels of L_level,

  where L_level is HiGHS's certified global lower bound (the MIP dual bound) for that minimization.
- Certificate, for any admissible data-dependent method A:

      J(CSP) − J(A) <= J(CSP) − mean_level L_level,

  with J(CSP) the registered endpoint (the grid average over the frozen report-dependent CSP policies of every cell).

Also reported (§1.4): how close CSP is to the policy that is optimal for its own validation target. The validation
score of BMIX (SP on the posterior predictive) is compared with that of CSP in every cell.

    python experiments/gate1_certificate.py  ->  data/gate1_certificate.json (+ stdout)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bound(args):
    from model.bridge import make_params, Model
    from experiments import p1a_core as core
    from model import evidence as ev
    econ, ctx, level = args
    P = make_params(ctx, **core.ECON[econ])
    mix = ev.clean(np.mean([ev.truth(ctx, core.cv_of(ctx, level), s) for s in core.SHIFTS], axis=0))
    m = Model(P, time_limit=3600.0, gap=1e-6)
    st = m.solve_saa(mix, tiebreak=False)
    info = m.h.getInfo()
    return (econ, ctx, level), dict(status=st["status"], incumbent=float(info.objective_function_value),
                                    dual_bound=float(info.mip_dual_bound), gap=float(info.mip_gap))


def main():
    from experiments import p1a_core as core
    from experiments.analyze_p1a import load, usable
    jobs = [(e, z, lv) for e in ("main", "salv05", "salv0") for z in ("A", "B") for lv in core.CV_LEVELS]
    with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
        B = dict(pool.map(_bound, jobs))
    out = {}
    for econ in ("main", "salv05", "salv0"):
        rows = json.loads((ROOT / "data" / f"p1a_summary_p1a_{econ}.rows.json").read_text(encoding="utf-8"))
        plan, res = load(f"p1a_{econ}")
        for z in ("A", "B"):
            R = [r for r in rows if r["ctx"] == z]
            J_csp = float(np.mean([r["J_CSP"] for r in R]))
            per_level = {}
            for lv in core.CV_LEVELS:
                b = B[(econ, z, lv)]
                J_lv = float(np.mean([r["J_CSP"] for r in R if r["level"] == lv]))
                per_level[lv] = dict(J_CSP=J_lv, **b, cert_gap=J_lv - b["dual_bound"])
            L = float(np.mean([per_level[lv]["dual_bound"] for lv in core.CV_LEVELS]))
            cert = J_csp - L
            # §1.4: validation-target optimality of CSP versus BMIX, cell by cell
            dv = []
            for cell in plan["cells"]:
                if cell["ctx"] != z:
                    continue
                tr = core.training(z, cell["reports"])
                cs, bm = cell["jobs"]["CSP"], cell["jobs"]["BMIX"]
                if usable(plan["jobs"][cs], res.get(cs)) and usable(plan["jobs"][bm], res.get(bm)):
                    v = tr["val"]
                    dv.append(float((v @ np.array(res[cs]["C"])).mean() - (v @ np.array(res[bm]["C"])).mean()))
            out[f"{econ}|{z}"] = dict(J_CSP=J_csp, mean_dual_bound=L, certificate=cert, cert_rel=cert / J_csp,
                                      rules_out_threshold=bool(cert < max(3.0, 0.005 * J_csp)), levels=per_level,
                                      val_CSP_minus_BMIX=dict(mean=float(np.mean(dv)), max=float(np.max(dv)),
                                                              min=float(np.min(dv)), n=len(dv)))
            o = out[f"{econ}|{z}"]
            gaps = ", ".join("%.1e" % per_level[lv]["gap"] for lv in core.CV_LEVELS)
            vd = o["val_CSP_minus_BMIX"]
            print(f"{econ:6s} {z}: J_CSP {J_csp:8.2f} | certified bound {L:8.2f} | max possible gain of ANY method "
                  f"<= {cert:6.3f} $M ({100 * cert / J_csp:.3f} %) | threshold ruled out: {o['rules_out_threshold']} | "
                  f"MIP gaps {gaps} | validation CSP-BMIX mean {vd['mean']:+.3f} (max {vd['max']:+.3f})")
    (ROOT / "data" / "gate1_certificate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
