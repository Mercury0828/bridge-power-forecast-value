"""Phase 1e E0b/E0c: exact evaluation of frozen policies under the evidence-anchored proxy error families `txUS` and
`tpit_*` (the ERCOT TPIT vintage-controlled schedule errors, experiments/p1e_tpit.py)
(docs/phase1e_plan.md; sources docs/phase1e_sources.md). No new policies are solved.

    python experiments/p1e_eval.py   -> data/p1e/txus_summary.json

`txUS` (ε = Y − T in quarters; ε = −d, where d is the delay of the actual date after the utility's estimate):
- P(d <= 0) = 0.444, at d = 0 (base). Variant "txUS-early": half of it at d = −1 (one quarter early).
- Late (0.556):
  - 32 % at d ∈ {1, 2} (≤ 6 months);
  - 16 % at d ∈ {3, 4} (6–12 months);
  - 52 % at d ≥ 5 (> 12 months). This bin is a geometric on {5, …, 16}, truncated, with its parameter set so that the
    late mean is 16.4 months (S4).
- The within-bin shapes and the truncation at 16 quarters (the model's error support) are declared.
- "txUS-uninf" keeps the signal (category) marginal of txUS with Y independent of T: the matched control.

The planner's reliability model is unchanged (unbiased, σ uniform on 1–4 quarters), so this is also a misspecification
test.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments.analyze_p1c import _stat  # noqa: E402
from model import evidence as ev  # noqa: E402
from model.signal import joint_law, evaluate, clean_law  # noqa: E402

OUT = ROOT / "data" / "p1e"
RUNS = ROOT / "experiments" / "r2_runs"
P_LATE = 0.556
BINS = (0.32, 0.16, 0.52)
LATE_MEAN_Q = 16.4 / 3.0                      # 16.4 months in quarters
D_MAX = int(core.EPS_SUPPORT.max())


def _tail_pmf():
    """Geometric on d = 5..D_MAX (truncated), with its parameter set so the overall late mean is 16.4 months."""
    target_bin3 = (LATE_MEAN_Q - BINS[0] * 1.5 - BINS[1] * 3.5) / BINS[2]
    d = np.arange(5, D_MAX + 1)
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        r = 0.5 * (lo + hi)
        w = r ** (d - 5)
        w = w / w.sum()
        if (w @ d) < target_bin3:
            lo = r
        else:
            hi = r
    return d, w, float(w @ d), target_bin3


def txus_pmf(early_split=False):
    """-> pmf over EPS_SUPPORT (ε = −d)."""
    pe = {int(e): 0.0 for e in core.EPS_SUPPORT}
    p_on = 1.0 - P_LATE
    if early_split:
        pe[0] += p_on / 2
        pe[1] += p_on / 2                    # d = −1: actual one quarter before the estimate
    else:
        pe[0] += p_on
    for d in (1, 2):
        pe[-d] += P_LATE * BINS[0] / 2
    for d in (3, 4):
        pe[-d] += P_LATE * BINS[1] / 2
    dt, wt, _, _ = _tail_pmf()
    for d, w in zip(dt, wt):
        pe[-int(d)] += P_LATE * BINS[2] * w
    v = np.array([pe[int(e)] for e in core.EPS_SUPPORT])
    return v / v.sum()


def kernel_from_pmf(th, pe):
    K = np.zeros((core.NCAT, ev.Q + 1))
    for i in range(ev.Q + 1):
        t = i + 1
        for e, w in zip(core.EPS_SUPPORT, pe):
            K[core.category(t + e, th), i] += w
    return K


TPIT, TPIT_EXT = {}, {}


def tpit_pmf(key, ext=False):
    """E0c empirical pmf (data/p1e/tpit_errors.json; experiments/p1e_tpit.py) on EPS_SUPPORT. With `ext`, the Phase-1f
    E0-ext pmf (data/p1e/tpit_errors_ext.json: the pre-2015 workbooks pooled with the primary ones)."""
    store, name = (TPIT_EXT, "tpit_errors_ext.json") if ext else (TPIT, "tpit_errors.json")
    if not store:
        store.update(json.loads((OUT / name).read_text(encoding="utf-8"))["summary"])
    pmf = store[key]["pmf"]
    v = np.array([pmf.get(str(int(e)), pmf.get(int(e), 0.0)) for e in core.EPS_SUPPORT], dtype=float)
    return v / v.sum()


def family_pmf(fam):
    base = fam.replace("-uninf", "")
    if base.startswith("txUS"):
        return txus_pmf(early_split=(base == "txUS-early"))
    ext = base.startswith("tpit_ext_")
    key = base.replace("tpit_ext_", "tpit_")
    return tpit_pmf({"tpit_h8": "h8|all", "tpit_h4": "h4|all", "tpit_h12": "h12|all",
                     "tpit_h8_load": "h8|load"}[key], ext=ext)


def family_kernel(th, fam, pt):
    K = kernel_from_pmf(th, family_pmf(fam))
    if fam.endswith("-uninf"):
        m = K @ np.asarray(pt)                  # the family's own signal marginal, with Y independent of T
        K = np.repeat(m[:, None], ev.Q + 1, axis=1)
    return K


FAMS = ("txUS", "txUS-early", "txUS-uninf", "tpit_h8", "tpit_h8-uninf", "tpit_h4", "tpit_h12", "tpit_h8_load")
FAMS_EXT = ("tpit_ext_h4", "tpit_ext_h8", "tpit_ext_h8-uninf", "tpit_ext_h12", "tpit_ext_h8_load")   # Phase 1f E0-ext


def evaluate_run(run, base_of=None, fams=FAMS):
    """Exact evaluation of every cell's frozen nosig/sig policies of a run under the proxy families (Δ = 0, ±2)."""
    d = RUNS / run
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    rows = []
    for cell in plan["cells"]:
        J = cell["jobs"]
        fn, fs = d / "items" / f"J_{J['nosig']}.json", d / "items" / f"J_{J['sig']}.json"
        if not (fn.exists() and fs.exists()):
            continue
        ns, sg = json.loads(fn.read_text(encoding="utf-8")), json.loads(fs.read_text(encoding="utf-8"))
        if not (ns.get("ok") and sg.get("ok")):
            continue
        base = cell.get("base", cell["cfg"])
        th = tuple(cell["thresholds"])
        if "costs_rounded" in ns:          # batch whole-unit runs (E2, E4b): the implemented whole-unit policies
            cn, cs = ns["costs_rounded"][0], sg["costs_rounded"]
        else:
            cn, cs = ns["C"], sg["costs"]
        for fam in fams:
            for s in core.SHIFTS:
                pt = core.truth(base, cell["level"], s)
                law = clean_law(joint_law(pt, family_kernel(th, fam, pt), core.TAU))
                jn = evaluate(np.array([cn] * core.NCAT), law, core.TAU)
                js = evaluate(np.array(cs), law, core.TAU)
                rows.append(dict(run=run, cfg=cell["cfg"], base=base, level=cell["level"], r=cell["r"], p=cell["p"],
                                 family=fam, shift=s, J_nosig=jn, J_sig=js, d_info=jn - js))
    return rows


def summarize(rows):
    out = {}
    for key in sorted({(r["run"], r["cfg"], r["family"], r["shift"]) for r in rows}):
        run, cfg, fam, s = key
        ser = {(r["level"], r["r"]): (r["p"], r["d_info"], r["J_nosig"]) for r in rows
               if (r["run"], r["cfg"], r["family"], r["shift"]) == key}
        base = next(r["base"] for r in rows if (r["run"], r["cfg"]) == (run, cfg))
        out[f"{run}|{cfg}|{fam}|{s}"] = _stat(ser, core.base_ctx(base))
    return out


def main_ext():
    """Phase 1f E0-ext: the same six policy sets under the extended TPIT families; a separate output."""
    runs = ("p1c_main", "p1d_spine", "p1e_e1", "p1e_e2", "p1e_e4b_spine", "p1e_e4b_rsb")
    rows = []
    for run in runs:
        if (RUNS / run / "plan.json").exists():
            rows += evaluate_run(run, fams=FAMS_EXT)
    S = dict(source="data/p1e/tpit_errors_ext.json (pre-declared rules: docs/phase1f_plan.md E0-ext)",
             table=summarize(rows))
    (OUT / "e0_ext_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (OUT / "e0_ext_rows.json").write_text(json.dumps(rows, default=float), encoding="utf-8")
    for k, v in S["table"].items():
        if k.endswith("|0"):
            print(f"{k:50s} E[d_info] {v['mean']:+8.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dt, wt, m3, target = _tail_pmf()
    pe = txus_pmf()
    e = core.EPS_SUPPORT
    info = dict(P_late=P_LATE, bins=BINS, tail_mean_q=m3, tail_target_q=target,
                late_mean_months=float(3 * (BINS[0] * 1.5 + BINS[1] * 3.5 + BINS[2] * m3)),
                mean_error_q=float(pe @ e), sd_error_q=float(np.sqrt(pe @ e ** 2 - (pe @ e) ** 2)),
                pmf={int(k): float(v) for k, v in zip(e, pe) if v > 0})
    rows = []
    # the three continuous policy sets, and their batched whole-unit implementations
    extra = ("p1e_e1", "p1e_e2", "p1e_e4b_spine", "p1e_e4b_rsb")
    for run in ("p1c_main", "p1d_spine") + tuple(r for r in extra if (RUNS / r / "plan.json").exists()):
        rows += evaluate_run(run)
    S = dict(family=info, table=summarize(rows))
    (OUT / "txus_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (OUT / "txus_rows.json").write_text(json.dumps(rows, default=float), encoding="utf-8")
    print(json.dumps(info, indent=1, default=float)[:800])
    for k, v in S["table"].items():
        if k.endswith("|0"):
            print(f"{k:45s} E[d_info] {v['mean']:+8.3f}  by level {({a: round(b, 2) for a, b in v['by_level'].items()})}")


if __name__ == "__main__":
    main_ext() if "--ext" in sys.argv[1:] else main()
