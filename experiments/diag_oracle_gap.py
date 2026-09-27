"""EXPLORATORY diagnostic (not pre-registered; not used for any claim). Run on frozen Phase-0 R2 outputs, before any confirmatory phase is
designed.

For the base point (p_R 25, cap 100, residence on) at delay x1 and x3, in contexts A and B:
  * premium a = p_hat' (C_CDRO - C_CSAA)            (the nominal cost of robustness under the training centre)
  * oracle v(p) = min_pi p' C_pi (SAA solved on p)   (knows the distribution, not the realized date)
  * SP gap = p' C_CSAA - v(p): the largest expected-cost improvement ANY feasible method could make over same-centre SP
  * CDRO gap = p' C_CDRO - v(p)
  * whether each policy's cost vector C(t) is nondecreasing in t
  * the T1 barycentre vs p_hat
Evaluation distributions p: p_hat, T1 barycentre (0.95 p_hat + 0.05 u), p_hat shifted -2/+2/+4, and each LORO record
component.

python experiments/diag_oracle_gap.py  ->  data/diag_oracle_gap.json (+ stdout table)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oracle(args):
    from model.bridge import make_params, Model
    ctx, dm, name, p = args
    P = make_params(ctx, p_R=25.0, cap_R=100.0, delay_mult=dm, residence_on=True)
    m = Model(P)
    st = m.solve_saa(np.array(p))
    return (ctx, dm, name), dict(ok=st["ok"], v=st.get("primary_obj", st["obj"]))


def main():
    from experiments.r2_core import load_centres, load_components, COMPONENTS
    from experiments.analyze_r2 import shifted, MIX
    cen, comps = load_centres(), load_components()
    items = ROOT / "experiments" / "r2_runs" / "r2_main" / "items"
    jobs, meta = [], {}
    for ctx in ("A", "B"):
        ph = cen[ctx]
        n = len(ph)
        dists = {"p_hat": ph, "T1_barycentre": (1 - MIX) * ph + MIX * np.ones(n) / n,
                 "shift-2": shifted(ph, -2), "shift+2": shifted(ph, 2), "shift+4": shifted(ph, 4)}
        for r in COMPONENTS[ctx]:
            dists[f"component_{r}"] = comps[r]
        for dm in (1.0, 3.0):
            it = json.loads((items / f"{ctx}_pR25_cap100_d{dm:g}_res1_e1.json").read_text(encoding="utf-8"))
            C = {k: np.array(v["C"]) for k, v in it["methods"].items()}
            meta[(ctx, dm)] = dict(C=C, dists=dists)
            for name, p in dists.items():
                jobs.append((ctx, dm, name, list(p)))
    with mp.get_context("spawn").Pool(processes=min(16, len(jobs))) as pool:
        oracle = dict(pool.map(_oracle, jobs))
    out = []
    print(f"{'ctx':3s} {'dly':>4s} {'distribution':>22s} | {'E[CSAA]':>9s} {'E[CDRO]':>9s} {'v(p)':>9s} | "
          f"{'SP gap':>8s} {'SP gap %':>8s} {'CDRO-CSAA %':>11s}")
    for (ctx, dm), d in meta.items():
        C = d["C"]
        ph = cen[ctx]
        premium = float(ph @ (C["CDRO"] - C["CSAA"]))
        mono = {k: bool(np.all(np.diff(C[k]) >= -1e-6)) for k in C}
        for name, p in d["dists"].items():
            o = oracle[(ctx, dm, name)]
            Es, Ed = float(p @ C["CSAA"]), float(p @ C["CDRO"])
            row = dict(ctx=ctx, delay_mult=dm, dist=name, E_CSAA=Es, E_CDRO=Ed, v=o["v"], oracle_ok=o["ok"],
                       SP_gap=Es - o["v"], SP_gap_pct=(Es - o["v"]) / o["v"], CDRO_vs_CSAA_pct=(Ed - Es) / Es,
                       premium_at_centre=premium, monotone=mono)
            out.append(row)
            print(f"{ctx:3s} {dm:4.0f} {name:>22s} | {Es:9.1f} {Ed:9.1f} {o['v']:9.1f} | {Es - o['v']:8.2f} "
                  f"{100 * (Es - o['v']) / o['v']:7.2f}% {100 * (Ed - Es) / Es:10.2f}%")
        print(f"    premium a (CDRO - CSAA at p_hat) = {premium:.2f} $M; monotone C(t): "
              + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in mono.items()))
    bary = {z: float(np.abs((1 - MIX) * cen[z] + MIX / len(cen[z]) - cen[z]).sum()) for z in ("A", "B")}
    zeros = {z: int((cen[z] == 0).sum()) for z in ("A", "B")}
    print(f"T1 barycentre vs p_hat: L1 distance A={bary['A']:.3f}, B={bary['B']:.3f}; zero-mass states in p_hat: {zeros}")
    (ROOT / "data" / "diag_oracle_gap.json").write_text(json.dumps(dict(rows=out, T1_barycentre_L1=bary,
                                                                        zero_states=zeros), indent=1),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
