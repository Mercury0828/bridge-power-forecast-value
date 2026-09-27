"""Gate 2 screen 4: the signal arrives at node tau = 4 (end of the engineering study), in
the signal tree (model/signal.py). EXPLORATORY; not used for any claim.

Designs (the marginal P_T is the real-evidence centre and is preserved; main economics):
  one  : a single forecast type; 3 categories (terciles of P_T); reliability r in [0.6, 0.95]
  two  : two forecast types observed with the forecast (weight 1/2 each), 2 categories (T at or before vs after the
         median of P_T); strong r_s in [0.8, 0.95], weak r_w in [0.55, 0.95]
Planners, all signal-aware:
  - predictive: SP on the joint law at the prior-mean reliabilities (expected cost is linear in r, so this is exact);
  - robust: min over policies of max over the vertices of the reliability box (exact, by linearity in r);
  - oracle(truth): SP on the true joint law;
  - ignore: no-signal SP.
Reported at each truth: V_signal = ignore − oracle; G_practical = predictive − oracle; robust − predictive.

    python experiments/gate2_screen_tree.py [ctx ...]  ->  data/gate2_screen_tree_<ctxs>.json (+ stdout)
    A context "A+staged" uses staged connection (25 % of peak for 4 quarters after T).
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
TAU = 4
DESIGNS = {
    "one": dict(ranges=[(0.6, 0.95)], cats=3),
    "two": dict(ranges=[(0.8, 0.95), (0.55, 0.95)], cats=2),
}
TRUTHS = {"one": [(0.6,), (0.775,), (0.95,)],
          "two": [(0.8, 0.55), (0.8, 0.95), (0.95, 0.55), (0.95, 0.95), (0.875, 0.75)]}


def categories(p, n):
    cdf = np.cumsum(p)
    if n == 3:
        return np.where(cdf <= 1 / 3 + 1e-12, 0, np.where(cdf <= 2 / 3 + 1e-12, 1, 2))
    return np.where(cdf <= 0.5 + 1e-12, 0, 1)


def kernel(p, design, rs):
    """Signal values y = (type, category), flattened. K(y | t) = w_type * accuracy term."""
    d = DESIGNS[design]
    n, cat = d["cats"], categories(p, d["cats"])
    wt = 1.0 / len(d["ranges"])
    rows = []
    for ty, r in enumerate(rs):
        for y in range(n):
            rows.append([wt * (r if cat[i] == y else (1 - r) / (n - 1)) for i in range(len(p))])
    return np.array(rows)


def _job(args):
    from model.bridge import make_params
    from model.signal import SignalModel, joint_law, evaluate
    from experiments import p1a_core as core
    ctx, design, tag, laws, n_sig = args
    base, _, cfg = ctx.partition("+")
    over = dict(stage_len=4, stage_frac=0.25) if cfg == "staged" else {}
    P = make_params(base, **dict(core.ECON["main"], **over))
    sm = SignalModel(P, n_sig, TAU, gap=1e-5, time_limit=7200.0)
    st = sm.solve_saa(laws[0]) if len(laws) == 1 else sm.solve_robust(laws)
    if not st["ok"]:
        return (ctx, design, tag), None
    costs = sm.polished_costs()
    return (ctx, design, tag), [c.tolist() for c in costs] if costs is not None else None


def main(ctxs=("A", "B")):
    from model import evidence as ev
    from model.signal import joint_law, evaluate
    jobs, meta = [], {}
    for ctx in ctxs:
        p = ev.estimate(ctx.partition("+")[0], ev.REAL_REPORTS[ctx.partition("+")[0]])
        for design, d in DESIGNS.items():
            n_sig = len(d["ranges"]) * d["cats"]
            means = tuple(0.5 * (lo + hi) for lo, hi in d["ranges"])
            verts = list(itertools.product(*d["ranges"]))
            law = lambda rs, p=p, design=design: joint_law(p, kernel(p, design, rs), TAU)   # noqa: E731 (bound now)
            meta[(ctx, design)] = dict(p=p, law=law, n=n_sig)
            ignore_law = joint_law(p, np.full((n_sig, len(p)), 1.0 / n_sig), TAU)   # uninformative = ignore
            jobs.append((ctx, design, "ignore", [ignore_law], n_sig))
            jobs.append((ctx, design, "predictive", [law(means)], n_sig))
            jobs.append((ctx, design, "robust", [law(v) for v in verts], n_sig))
            for t in TRUTHS[design]:
                jobs.append((ctx, design, f"oracle|{t}", [law(t)], n_sig))
    with mp.get_context("spawn").Pool(processes=min(12, len(jobs))) as pool:
        R = dict(pool.map(_job, jobs))
    out = {}
    for ctx in ctxs:
        for design in DESIGNS:
            mt = meta[(ctx, design)]
            get = lambda tag: R.get((ctx, design, tag))                     # noqa: E731
            rows = []
            for t in TRUTHS[design]:
                lw = mt["law"](t)
                J = {k: evaluate(np.array(get(k)), lw, TAU) for k in ("ignore", "predictive", "robust")
                     if get(k) is not None}
                o = get(f"oracle|{t}")
                J["oracle"] = evaluate(np.array(o), lw, TAU) if o is not None else float("nan")
                rows.append(dict(truth=list(t), **J))
            out[f"{ctx}|{design}"] = rows
            print(f"== {ctx} {design} (tau = {TAU}): truth | V_signal = ignore − oracle | G_practical = pred − oracle | "
                  f"robust − pred | robust − ignore")
            for r in rows:
                print(f"   {r['truth']} | {r['ignore'] - r['oracle']:+7.2f} | {r['predictive'] - r['oracle']:+7.2f} | "
                      f"{r['robust'] - r['predictive']:+7.2f} | {r['robust'] - r['ignore']:+7.2f}")
    tag = "_".join(c.replace("+", "-") for c in ctxs)
    (ROOT / "data" / f"gate2_screen_tree_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "data" / f"gate2_screen_tree_{tag}_costs.json").write_text(
        json.dumps({"|".join(k): v for k, v in R.items()}), encoding="utf-8")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("A", "B"))
