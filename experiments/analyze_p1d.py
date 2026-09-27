"""Phase-1d exploratory analysis (docs/phase1d_plan.md v2). No claims and no thresholds: every number is a sensitivity
or mechanism diagnostic, reported against the matched Phase-1c cells.

    python experiments/analyze_p1d.py --variant stage --run p1d_stage   -> data/p1d/<variant>_summary.json

Evaluation is the Phase-1c exact evaluation:
- the true joint law is the base context's truth marginal (level, shift) × the family kernel under the cell's own
  thresholds, cleaned;
- both frozen policies are evaluated under it.

The grid statistic is the Phase-1c one: the equal-weight mean over levels of Σ_outcomes p × effect.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments.analyze_p1c import _stat  # noqa: E402
from model.signal import joint_law, evaluate, clean_law  # noqa: E402

OUT = ROOT / "data" / "p1d"


def usable(spec, r):
    if r is None or not r.get("ok"):
        return False
    if spec["kind"].startswith(("units_", "batch_")):
        return (r.get("cost_match", 1.0) <= 1e-6 and r.get("bookkeeping", 1.0) <= 1e-6
                and r.get("shortfall", 1.0) <= 1e-6)
    if r.get("tiebreak_ok") is False:
        return False
    return r.get("bookkeeping_maxabs", 1.0) <= 1e-6 and r.get("backup_shortfall", 1.0) <= 1e-6


def load(run):
    d = ROOT / "experiments" / "r2_runs" / run
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    res = {}
    for k in plan["jobs"]:
        f = d / "items" / f"J_{k}.json"
        if f.exists():
            res[k] = json.loads(f.read_text(encoding="utf-8"))
    return plan, res


def _tau(cell):
    return int(cell.get("tau", core.TAU))                   # Phase 1f E-T; Phase 1c-1e cells: TAU


def _law(cell, fam, s):
    pt = core.truth(cell["base"], cell["level"], s)
    return clean_law(joint_law(pt, core.true_kernel(tuple(cell["thresholds"]), fam, pt), _tau(cell)))


def _rsb_branch(d, T):
    """Rented standby (MW) held after connection in scenario T, summed over quarters T+1..Q (quarter T is on the
    spine)."""
    b = d["branches"].get(str(T), d["branches"].get(T))
    return float(sum((b or {}).get("rsb") or [0.0])) - float(((b or {}).get("rsb") or [0.0])[0])


def _realized(detail, law, per_copy, tau=None):
    """Expected realized spine orders by asset (orders at node k happen iff T > k) and ST rental MW-quarters (quarter q
    rented iff T > q), under the joint law. per_copy: signal policy (details per category).
    With rented standby (E1): RSB_MWq, the spine rentals (quarter q contracted at node q - 1, so held iff T > q - 1),
    and RSBb_MWq, the rentals held after connection within the horizon (minimum commitment; the residual beyond Q is
    not counted)."""
    tau = core.TAU if tau is None else tau
    pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
    n = len(pre)
    surv_all = lambda w: np.array([w[k:].sum() for k in range(n)])            # noqa: E731  P(T > k) with T = i+1
    out = {}
    if not per_copy:
        m = pre + sum(post)
        s = surv_all(m)
        for j, v in detail["orders"].items():
            out[j] = float(sum(v[k] * s[k] for k in range(len(v))))
        for key, lab in (("RST", "ST_MWq"), ("RNR", "NR_MWq")):
            out[lab] = float(sum(detail[key][q - 1] * s[q] for q in range(1, len(detail[key]) + 1)))
        if "RSB" in detail:
            out["RSB_MWq"] = float(sum(detail["RSB"][q - 1] * s[q - 1] for q in range(1, len(detail["RSB"]) + 1)))
            out["RSBb_MWq"] = float(sum(m[T - 1] * _rsb_branch(detail, T) for T in range(1, n + 1)))
        return out
    # signal policy: nodes k < TAU are common (copy 0); nodes k >= TAU follow the category's copy
    s_pre = surv_all(pre + sum(post))
    s_y = [surv_all(p) for p in post]
    for j in detail[0]["orders"]:
        tot = 0.0
        for k in range(len(detail[0]["orders"][j])):
            if k < tau:
                tot += detail[0]["orders"][j][k] * s_pre[k]
            else:
                tot += sum(detail[y]["orders"][j][k] * s_y[y][k] for y in range(core.NCAT))
        out[j] = float(tot)
    for key, lab in (("RST", "ST_MWq"), ("RNR", "NR_MWq")):
        tot = 0.0
        for q in range(1, len(detail[0][key]) + 1):
            if q - 1 < tau:                               # the rental for quarter q is decided at node q - 1
                tot += detail[0][key][q - 1] * s_pre[q]
            else:
                tot += sum(detail[y][key][q - 1] * s_y[y][q] for y in range(core.NCAT))
        out[lab] = float(tot)
    if "RSB" in detail[0]:
        tot = 0.0
        for q in range(1, len(detail[0]["RSB"]) + 1):
            if q - 1 < tau:
                tot += detail[0]["RSB"][q - 1] * s_pre[q - 1]
            else:
                tot += sum(detail[y]["RSB"][q - 1] * s_y[y][q - 1] for y in range(core.NCAT))
        out["RSB_MWq"] = float(tot)
        out["RSBb_MWq"] = float(sum(pre[T - 1] * _rsb_branch(detail[0], T)
                                    + sum(post[y][T - 1] * _rsb_branch(detail[y], T) for y in range(core.NCAT))
                                    for T in range(1, n + 1)))
    return out


def analyze(variant, run):
    plan, res = load(run)
    jobs = plan["jobs"]
    ok = {k: usable(jobs[k], res.get(k)) for k in jobs}
    rows = []
    for cell in plan["cells"]:
        J = cell["jobs"]
        if not (ok[J["nosig"]] and ok[J["sig"]]):
            continue
        ns, sg = res[J["nosig"]], res[J["sig"]]
        th = tuple(cell["thresholds"])
        for fam in core.FAMILIES:
            for s in core.SHIFTS:
                law = _law(cell, fam, s)
                row = dict(cfg=cell["cfg"], base=cell["base"], level=cell["level"], r=cell["r"], p=cell["p"],
                           family=fam, shift=s)
                if jobs[J["nosig"]]["kind"].startswith(("units_", "batch_")):
                    for tag, a, b in (("orig", ns["costs"][0], sg["costs"]), ("rounded", ns["costs_rounded"][0],
                                                                                sg["costs_rounded"])):
                        jn = evaluate(np.array([a] * core.NCAT), law, _tau(cell))
                        js = evaluate(np.array(b), law, _tau(cell))
                        row.update({f"J_nosig_{tag}": jn, f"J_sig_{tag}": js, f"d_info_{tag}": jn - js})
                    row.update(J_nosig=row["J_nosig_rounded"], J_sig=row["J_sig_rounded"],
                               d_info=row["d_info_rounded"])
                else:
                    jn = evaluate(np.array([ns["C"]] * core.NCAT), law, _tau(cell))
                    js = evaluate(np.array(sg["costs"]), law, _tau(cell))
                    row.update(J_nosig=jn, J_sig=js, d_info=jn - js)
                    if fam == "sym2" and s == 0:
                        row["realized_nosig"] = _realized(ns["detail"], law, False, _tau(cell))
                        row["realized_sig"] = _realized(sg["detail"], law, True, _tau(cell))
                rows.append(row)
    return dict(rows=rows, n_jobs=len(jobs), n_usable=int(sum(ok.values())),
                unusable=[k for k in jobs if not ok[k]], complete=all(ok.values()))


def _series(rows, cfg, fam, s=0, key="d_info"):
    return {(r["level"], r["r"]): (r["p"], r[key], r["J_nosig"]) for r in rows
            if r["cfg"] == cfg and r["family"] == fam and r["shift"] == s}


def summarize(variant, out):
    rows = out["rows"]
    p1c_rows = json.loads((ROOT / "data" / "p1c_rows.json").read_text(encoding="utf-8"))
    names = sorted({r["cfg"] for r in rows})
    base_of = {r["cfg"]: r["base"] for r in rows}
    S = dict(variant=variant, complete=out["complete"], n_jobs=out["n_jobs"], n_usable=out["n_usable"],
             unusable=out["unusable"], table={}, vs_phase1c={}, costs={}, realized={})
    for name in names:
        base = base_of[name]
        ctx = core.base_ctx(base)
        for fam in core.FAMILIES:
            for s in core.SHIFTS:
                ser = _series(rows, name, fam, s)
                if not ser:
                    continue
                st = _stat(ser, ctx)
                S["table"][f"{name}|{fam}|{s}"] = st
                ref = _series(p1c_rows, base, fam, s)
                common = sorted(set(ser) & set(ref))
                if common:
                    diff = {k: (ser[k][0], ser[k][1] - ref[k][1], ser[k][2]) for k in common}
                    S["vs_phase1c"][f"{name}|{fam}|{s}"] = dict(d_info_main=_stat({k: ref[k] for k in common}, ctx),
                                                              change=_stat(diff, ctx))
        # expected costs of each policy at Δ = 0, sym2 (equal level weights), against Phase 1c
        for pol in ("J_nosig", "J_sig"):
            ser = {(r["level"], r["r"]): (r["p"], r[pol], r["J_nosig"]) for r in rows
                   if r["cfg"] == name and r["family"] == "sym2" and r["shift"] == 0}
            ref = {(r["level"], r["r"]): (r["p"], r[pol], r["J_nosig"]) for r in p1c_rows
                   if r["cfg"] == base and r["family"] == "sym2" and r["shift"] == 0}
            common = sorted(set(ser) & set(ref))
            S["costs"][f"{name}|{pol}"] = dict(
                variant=_stat({k: ser[k] for k in common}, ctx)["mean"],
                phase1c=_stat({k: ref[k] for k in common}, ctx)["mean"],
                change=_stat({k: (ser[k][0], ser[k][1] - ref[k][1], ser[k][2]) for k in common}, ctx)["mean"])
        if any(r.get("d_info_orig") is not None for r in rows if r["cfg"] == name):
            for fam in core.FAMILIES:
                ser_o = {(r["level"], r["r"]): (r["p"], r["d_info_orig"], r["J_nosig"]) for r in rows
                         if r["cfg"] == name and r["family"] == fam and r["shift"] == 0}
                inc_n = {(r["level"], r["r"]): (r["p"], r["J_nosig_rounded"] - r["J_nosig_orig"], r["J_nosig"])
                         for r in rows if r["cfg"] == name and r["family"] == fam and r["shift"] == 0}
                inc_s = {(r["level"], r["r"]): (r["p"], r["J_sig_rounded"] - r["J_sig_orig"], r["J_nosig"])
                         for r in rows if r["cfg"] == name and r["family"] == fam and r["shift"] == 0}
                S["table"][f"{name}|{fam}|0"]["d_info_orig"] = _stat(ser_o, ctx)["mean"]
                S["table"][f"{name}|{fam}|0"]["rounding_cost_nosig"] = _stat(inc_n, ctx)["mean"]
                S["table"][f"{name}|{fam}|0"]["rounding_cost_sig"] = _stat(inc_s, ctx)["mean"]
        rz = [r for r in rows if r["cfg"] == name and r["family"] == "sym2" and r["shift"] == 0 and "realized_nosig" in r]
        if rz:
            by = {}
            for r in rz:
                by.setdefault(r["level"], []).append(r)
            agg = {}
            for pol in ("realized_nosig", "realized_sig"):
                keys = rz[0][pol].keys()
                agg[pol] = {k: float(np.mean([sum(r["p"] * r[pol][k] for r in v) / sum(r["p"] for r in v)
                                              for v in by.values()])) for k in keys}
            S["realized"][name] = agg
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    out = analyze(a.variant, a.run)
    S = summarize(a.variant, out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.variant}_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (OUT / f"{a.variant}_rows.json").write_text(json.dumps(out["rows"], default=float), encoding="utf-8")
    print(f"{a.variant}: complete {S['complete']} | jobs {S['n_jobs']} usable {S['n_usable']}")
    for k, v in S["table"].items():
        if k.endswith("|0") and ("|sym1|" in k or "|sym2|" in k or "|uninf|" in k):
            ch = S["vs_phase1c"].get(k, {}).get("change", {}).get("mean", float("nan"))
            print(f"  {k:32s} E[d_info] {v['mean']:+8.3f} (vs Phase 1c: {ch:+8.3f})")


if __name__ == "__main__":
    main()
