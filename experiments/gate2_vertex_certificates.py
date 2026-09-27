"""Family-wide certificates for the signal extensions. No optimizer search.

Signal families (as in experiments/gate2_screen_tree.py; main economics; tau = 4). The marginal P_T is the
real-evidence training centre, and signal categories are its terciles / median (training information). The
reliability box R is:
  one : r in [0.6, 0.95]                         (vertices 0.6, 0.95)
  two : r_s in [0.8, 0.95], r_w in [0.55, 0.95]   (4 vertices)
The joint law Q_r is affine in r, so for any frozen policy J_r(pi) is affine in r.

Frozen policies:
  - pi_ignore: the no-signal SP on P_T (standard Model). J_Q(pi_ignore) = P_T' C for every Q in the family.
  - pi_P: predictive signal SP at the prior-mean reliabilities.
  - pi_R: robust signal use, min over policies of max over the vertices.
Certified oracle lower bounds L_j: the HiGHS dual bound of min over signal-aware policies of J_{r_j}, at each vertex,
with MIP gap 1e-6.

Certificates, valid on the whole box:
  (1) oracle gap   sup_r [J_r(pi_P) − v(r)]           <= max_j [J_{r_j}(pi_P) − L_j]    (convexity of g)
  (2) information  inf_r [J(pi_ignore) − J_r(pi_P)]    =  min_j [J(pi_ignore) − J_{r_j}(pi_P)]   (affinity)
  (3) the same two quantities for pi_R, and the trade-off J_r(pi_R) − J_r(pi_P) at the vertices.

    python experiments/gate2_vertex_certificates.py [config ...]  ->  data/gate2_vertex_certificates.json
    configs: A, A+staged, B (default: all three)
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
from experiments.gate2_screen_tree import DESIGNS, TAU, kernel  # noqa: E402

GAP, TLIM = 1e-6, 14400.0


def _params(cfg):
    from model.bridge import make_params
    from experiments import p1a_core as core
    base, _, extra = cfg.partition("+")
    over = dict(stage_len=4, stage_frac=0.25) if extra == "staged" else {}
    return make_params(base, **dict(core.ECON["main"], **over))


def _job(args):
    from model.bridge import Model, polish
    from model.signal import SignalModel
    cfg, design, tag, laws, n_sig, pT = args
    P = _params(cfg)
    if tag == "ignore":
        m = Model(P, gap=GAP, time_limit=TLIM)
        st = m.solve_saa(np.array(pT), tiebreak=False)
        if not st["ok"]:
            return (cfg, design, tag), dict(ok=False, status=st["status"])
        pm, pst = polish(P, m.spine_values())
        info = m.h.getInfo()
        return (cfg, design, tag), dict(ok=bool(pst["ok"]), C=pm.C_values().tolist(), incumbent=st["obj"],
                                        dual_bound=float(info.mip_dual_bound), gap=float(info.mip_gap))
    sm = SignalModel(P, n_sig, TAU, gap=GAP, time_limit=TLIM)
    st = sm.solve_saa(laws[0]) if len(laws) == 1 else sm.solve_robust(laws)
    if not st["ok"]:
        return (cfg, design, tag), dict(ok=False, status=st["status"])
    costs = sm.polished_costs()
    return (cfg, design, tag), dict(ok=costs is not None, costs=[c.tolist() for c in costs] if costs else None,
                                    incumbent=st["obj"], dual_bound=st["dual_bound"], gap=st["gap"])


def main(cfgs=("A", "A+staged", "B")):
    from model import evidence as ev
    from model.signal import joint_law, evaluate
    jobs, fam = [], {}
    for cfg in cfgs:
        p = ev.estimate(cfg.partition("+")[0], ev.REAL_REPORTS[cfg.partition("+")[0]])
        jobs.append((cfg, "-", "ignore", None, None, p.tolist()))
        for design, d in DESIGNS.items():
            n_sig = len(d["ranges"]) * d["cats"]
            verts = list(itertools.product(*d["ranges"]))
            mean = tuple(0.5 * (lo + hi) for lo, hi in d["ranges"])
            law = {v: joint_law(p, kernel(p, design, v), TAU) for v in verts + [mean]}
            fam[(cfg, design)] = dict(p=p, verts=verts, mean=mean, law=law, n=n_sig)
            jobs.append((cfg, design, "predictive", [law[mean]], n_sig, None))
            jobs.append((cfg, design, "robust", [law[v] for v in verts], n_sig, None))
            for v in verts:
                jobs.append((cfg, design, f"oracle|{v}", [law[v]], n_sig, None))
    with mp.get_context("spawn").Pool(processes=min(12, len(jobs))) as pool:
        R = dict(pool.map(_job, jobs))
    out = {}
    for cfg in cfgs:
        ign = R[(cfg, "-", "ignore")]
        for design in DESIGNS:
            f = fam[(cfg, design)]
            key = f"{cfg}|{design}"
            if not ign.get("ok"):
                out[key] = dict(error="ignore failed")
                continue
            J_ign = float(f["p"] @ np.array(ign["C"]))
            rows, ok = [], True
            for v in f["verts"]:
                lw, orc = f["law"][v], R[(cfg, design, f"oracle|{v}")]
                row = dict(vertex=list(v), J_ignore=J_ign)
                for name in ("predictive", "robust"):
                    r_ = R[(cfg, design, name)]
                    if r_.get("ok"):
                        row[f"J_{name}"] = evaluate(np.array(r_["costs"]), lw, TAU)
                    else:
                        ok = False
                row.update(oracle_incumbent=orc.get("incumbent"), oracle_bound=orc.get("dual_bound"),
                           oracle_gap=orc.get("gap"), oracle_ok=orc.get("ok"))
                ok = ok and bool(orc.get("ok"))
                rows.append(row)
            res = dict(rows=rows, complete=ok, J_ignore=J_ign, ignore_bound=ign["dual_bound"])
            if ok:
                res["cert_oracle_gap_predictive"] = max(r["J_predictive"] - r["oracle_bound"] for r in rows)
                res["cert_oracle_gap_robust"] = max(r["J_robust"] - r["oracle_bound"] for r in rows)
                res["cert_info_value_predictive"] = min(r["J_ignore"] - r["J_predictive"] for r in rows)
                res["cert_info_value_robust"] = min(r["J_ignore"] - r["J_robust"] for r in rows)
                res["oracle_info_value_range"] = (min(r["J_ignore"] - r["oracle_incumbent"] for r in rows),
                                                  max(r["J_ignore"] - r["oracle_incumbent"] for r in rows))
                res["robust_minus_predictive"] = [r["J_robust"] - r["J_predictive"] for r in rows]
                print(f"{key:12s}: J_ignore {J_ign:8.2f} | CERT room over predictive <= "
                      f"{res['cert_oracle_gap_predictive']:6.3f} | CERT implemented info value (predictive) >= "
                      f"{res['cert_info_value_predictive']:+7.3f} | robust: room <= {res['cert_oracle_gap_robust']:6.3f}, "
                      f"info >= {res['cert_info_value_robust']:+7.3f} | robust − predictive at vertices "
                      f"{[round(x, 2) for x in res['robust_minus_predictive']]} | max MIP gap "
                      f"{max(r['oracle_gap'] for r in rows):.1e}")
            else:
                print(f"{key:12s}: INCOMPLETE (a solve failed)")
            out[key] = res
    tag = "_".join(c.replace("+", "-") for c in cfgs)
    (ROOT / "data" / f"gate2_vertex_certificates_{tag}.json").write_text(
        json.dumps(dict(results=out, raw={"|".join(k): v for k, v in R.items()}), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("A", "A+staged", "B"))
