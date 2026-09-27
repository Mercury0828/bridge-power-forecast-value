"""Phase 1f S4 checks (the *Energy* #1 self-check, lenses 3, 4 and 11; `selfcheck/expected.md` "S4 checks").

    python experiments/p1f_checks.py balances   -> data/p1f/balances.json
    python experiments/p1f_checks.py gaps       -> data/p1f/gap_summary.json
    python experiments/p1f_checks.py rerun      -> data/p1f/rerun_check.json
    python experiments/p1f_checks.py evpi       -> annotates data/gate2_screen_signal.json (invalid EVPI field)
    python experiments/p1f_checks.py plannerlaw -> data/p1f/planner_law.json (embedding check; value by τ)

balances. The 8 representative Phase-1c cells (each configuration and dispersion level: the highest-probability report
outcome, as in `experiments/p1d_accounting.py repcells`) are re-solved exactly as registered. The no-signal policy
and every signal copy are polished (spine fixed, branches optimized). For every block of every quarter:
- pre-connection (spine, q = 1..Q): electricity gGE + gDG + gNR + gST = (1 + a_oth) S + e / COP; cooling
  a + e = kappa S; heat a <= cop_abs eta_h gGE;
- post-connection (branch T, q = T..Q): electricity m + g (+ gd + gs while partial) - e / COP = (1 + a_oth) L_q;
  cooling a + e = kappa L_q; heat a <= cop_abs eta_h g.
It reports the largest absolute equality residual, the most negative inequality slack, and the recovered heat that is
rejected (eta_h g - a / cop_abs), plus the first-law and unit audit of the declared conversion factors.

gaps. The stored solver outcome of every job in the Phase-1c, 1d and 1e runs: the no-signal primary gap, the oracle gap,
and the status. Signal jobs record only the tie-break stage. Their primary stage is optimal at mip_rel_gap 1e-4 by
construction (a job is usable only if that stage is kOptimal).

rerun. 8 Phase-1c jobs (for each configuration, the no-signal and signal jobs of the highest-probability low-dispersion
cell; the D-022a job is excluded) are re-run from their frozen specs with the harness call, and compared with the stored
items for exact JSON equality.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from model.bridge import Model, polish  # noqa: E402

OUT = ROOT / "data" / "p1f"
RUNS = ROOT / "experiments" / "r2_runs"
P1C = RUNS / "p1c_main"
MMBTU_PER_MWH = 3.412
DIESEL_MMBTU_PER_GAL = 5.772 / 42


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"written data/p1f/{name}")


def _load_plan(run=P1C):
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    return plan, (lambda k: json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8")))


def _rep_cells(plan):
    out = []
    for cfg in core.CONFIGS:
        for lv in core.LEVELS:
            cs = [c for c in plan["cells"] if c["cfg"] == cfg and c["level"] == lv]
            out.append(max(cs, key=lambda x: x["p"]))
    return out


# ---- balances -----------------------------------------------------------------------------------------------------
def balance_report(P, pm):
    """Largest residuals and slacks of the energy balances of a polished single-copy policy (MW / MWth)."""
    val, Q = pm.val, P.Q
    r = dict(elec_eq=0.0, cool_eq=0.0, heat_slack_min=np.inf, nonneg_min=np.inf, heat_rejected_MWth_max=0.0,
             blocks_checked=0)
    blocks = getattr(P, "blocks", (0, 1))
    for q in range(1, Q + 1):
        for b in blocks:
            g = {k: val(pm.v[k][q, b]) for k in ("gGE", "gDG", "gNR", "gST", "a", "e")}
            S = val(pm.v["S"][q])
            c = P.cop(q, b)
            r["elec_eq"] = max(r["elec_eq"], abs(g["gGE"] + g["gDG"] + g["gNR"] + g["gST"]
                                                  - (1 + P.a_oth) * S - g["e"] / c))
            r["cool_eq"] = max(r["cool_eq"], abs(g["a"] + g["e"] - P.kappa * S))
            r["heat_slack_min"] = min(r["heat_slack_min"], P.cop_abs * P.eta_h * g["gGE"] - g["a"])
            r["nonneg_min"] = min(r["nonneg_min"], *g.values())
            r["heat_rejected_MWth_max"] = max(r["heat_rejected_MWth_max"], P.eta_h * g["gGE"] - g["a"] / P.cop_abs)
            r["blocks_checked"] += 1
    for T in range(1, Q + 1):
        for q in range(T, Q + 1):
            Lq = P.L(q)
            for b in blocks:
                m, g, a, e = (val(pm.v[k][T, q, b]) for k in ("m", "bg", "ba", "be"))
                extra = 0.0
                if (T, q, b) in pm.v.get("bgd", {}):
                    extra = val(pm.v["bgd"][T, q, b]) + val(pm.v["bgs"][T, q, b])
                c = P.cop(q, b)
                r["elec_eq"] = max(r["elec_eq"], abs(m + g + extra - e / c - (1 + P.a_oth) * Lq))
                r["cool_eq"] = max(r["cool_eq"], abs(a + e - P.kappa * Lq))
                r["heat_slack_min"] = min(r["heat_slack_min"], P.cop_abs * P.eta_h * g - a)
                r["nonneg_min"] = min(r["nonneg_min"], m, g, a, e)
                r["heat_rejected_MWth_max"] = max(r["heat_rejected_MWth_max"], P.eta_h * g - a / P.cop_abs)
                r["blocks_checked"] += 1
    return r


def _balance_job(args):
    from model.signal import SignalModel
    cell, spec_ns, spec_sg, item_ns = args
    P = core.params(cell["cfg"])
    m = Model(P, time_limit=1800)
    m.solve_saa(np.array(spec_ns["centre"]), tiebreak_mode="lp",
                tiebreak_presolve=not (item_ns.get("repair", {}).get("tiebreak_presolve") == "off"))
    pms = [("nosig", polish(P, m.spine_values(), time_limit=1800)[0])]
    sm = SignalModel(P, core.NCAT, core.TAU, time_limit=1800)
    sm.solve_saa(spec_sg["law"], tiebreak=True, tiebreak_mode="lp")
    pms += [(f"sig_y{y}", polish(P, c.spine_values(), time_limit=1800)[0]) for y, c in enumerate(sm.copies)]
    return dict(cfg=cell["cfg"], level=cell["level"], r=cell["r"],
                policies={name: balance_report(P, pm) for name, pm in pms})


def unit_audit(P):
    """First-law and unit audit of the declared conversion factors (HHV)."""
    ge_in = P.ge_hr / MMBTU_PER_MWH                                   # MWh fuel (HHV) per MWh_e
    dg_in = P.dg_gal_per_mwh * DIESEL_MMBTU_PER_GAL / MMBTU_PER_MWH
    return dict(ge_electrical_eff_HHV=1 / ge_in, ge_recovered_heat_frac_of_fuel_HHV=P.eta_h / ge_in,
                ge_total_eff_HHV=(1 + P.eta_h) / ge_in, dg_electrical_eff_HHV=1 / dg_in,
                abs_cop=P.cop_abs, heat_to_cooling_per_MWh_e=P.cop_abs * P.eta_h,
                electric_chiller_cop_range=[min(min(c) for c in P.cop_e), max(max(c) for c in P.cop_e)],
                hours_per_quarter=P.H * len(getattr(P, "blocks", (0, 1))),
                checks=dict(ge_total_eff_below_1=(1 + P.eta_h) / ge_in < 1.0, dg_eff_below_1=1 / dg_in < 1.0,
                            abs_cop_below_1=P.cop_abs < 1.0, electric_cop_above_1=min(min(c) for c in P.cop_e) > 1.0,
                            hours_per_quarter_eq_91_3125_days=abs(P.H * len(P.blocks) - 91.3125 * 24) < 1e-9))


def cmd_balances():
    plan, load = _load_plan()
    jobs = [(c, plan["jobs"][c["jobs"]["nosig"]], plan["jobs"][c["jobs"]["sig"]], load(c["jobs"]["nosig"]))
            for c in _rep_cells(plan)]
    with mp.get_context("spawn").Pool(min(8, len(jobs))) as pool:
        res = pool.map(_balance_job, jobs)
    worst = dict(elec_eq=max(p["elec_eq"] for r in res for p in r["policies"].values()),
                 cool_eq=max(p["cool_eq"] for r in res for p in r["policies"].values()),
                 heat_slack_min=min(p["heat_slack_min"] for r in res for p in r["policies"].values()),
                 nonneg_min=min(p["nonneg_min"] for r in res for p in r["policies"].values()))
    audit = {ctx: unit_audit(core.params(ctx)) for ctx in ("A", "B")}
    _dump("balances.json", dict(cells=res, worst=worst, unit_audit=audit, tolerance=1e-6,
                                passed=bool(worst["elec_eq"] <= 1e-6 and worst["cool_eq"] <= 1e-6
                                            and worst["heat_slack_min"] >= -1e-6 and worst["nonneg_min"] >= -1e-6)))
    print(json.dumps(worst, indent=1, default=float))


# ---- gaps -------------------------------------------------------------------------------------------------------
def cmd_gaps():
    out = {}
    for d in sorted(RUNS.glob("p1[cdef]_*")):
        if not (d / "plan.json").exists() or "." in d.name:
            continue
        plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        acc = {}
        for k, spec in plan["jobs"].items():
            f = d / "items" / f"J_{k}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            kind = spec.get("kind", "?")
            a = acc.setdefault(kind, dict(n=0, ok=0, primary_gap_max=None, gap_max=None, statuses={}))
            a["n"] += 1
            a["ok"] += bool(r.get("ok"))
            pg = r.get("primary_gap")
            if isinstance(pg, (int, float)) and np.isfinite(pg):
                a["primary_gap_max"] = pg if a["primary_gap_max"] is None else max(a["primary_gap_max"], pg)
            g = r.get("gap")
            # a signal job's stored gap belongs to its tie-break stage, so it is not a solution-quality measure
            if kind != "sig" and isinstance(g, (int, float)) and np.isfinite(g):
                a["gap_max"] = g if a["gap_max"] is None else max(a["gap_max"], g)
            st = r.get("status")
            if st is not None:
                a["statuses"][str(st)] = a["statuses"].get(str(st), 0) + 1
        out[d.name] = acc
    note = ("Signal jobs store the tie-break stage's status and gap; their primary stage is optimal at mip_rel_gap 1e-4 "
            "because a job is usable only if that stage is kOptimal (model/signal.py solve_saa). No-signal jobs store the "
            "primary gap; oracle jobs were solved at 1e-5.")
    _dump("gap_summary.json", dict(runs=out, note=note))
    for run, acc in out.items():
        for kind, a in acc.items():
            print(f"{run:16s} {kind:12s} n {a['n']:4d} ok {a['ok']:4d} primary_gap_max {a['primary_gap_max']} "
                  f"gap_max {a['gap_max']}")


# ---- rerun ------------------------------------------------------------------------------------------------------
def _rerun_job(args):
    key, spec = args
    return key, core.run_job(spec, time_limit=1800.0)


def cmd_rerun():
    plan, load = _load_plan()
    sample = []
    for cfg in core.CONFIGS:
        cs = [c for c in plan["cells"] if c["cfg"] == cfg and c["level"] == "low"]
        for c in sorted(cs, key=lambda x: -x["p"]):
            ks = [c["jobs"]["nosig"], c["jobs"]["sig"]]
            if not any(load(k).get("repair") for k in ks):
                sample += [(k, plan["jobs"][k]) for k in ks]
                break
    with mp.get_context("spawn").Pool(min(8, len(sample))) as pool:
        res = dict(pool.map(_rerun_job, sample))
    def schema_normalized(item):
        """Drop the branch key `rsb` when it is None: Phase 1e added it to `policy_detail` (always None when rented
        standby is off), so the Phase-1c items, written earlier, lack it. Nothing numerical is dropped."""
        it = json.loads(json.dumps(item))
        dets = it.get("detail")
        for d in (dets if isinstance(dets, list) else [dets]):
            for b in ((d or {}).get("branches") or {}).values():
                if b.get("rsb", 0) is None:
                    b.pop("rsb")
        return it

    rows = []
    for k, _ in sample:
        new = json.loads(json.dumps(res[k], default=float))
        old = load(k)
        raw = sorted(x for x in set(old) | set(new) if old.get(x) != new.get(x))
        n_old, n_new = schema_normalized(old), schema_normalized(new)
        diff = sorted(x for x in set(n_old) | set(n_new) if n_old.get(x) != n_new.get(x))
        rows.append(dict(job=k, kind=plan["jobs"][k]["kind"], cfg=plan["jobs"][k]["cfg"], identical=not diff,
                         raw_differing_keys=raw, differing_keys_after_schema_normalization=diff))
    _dump("rerun_check.json", dict(jobs=rows, all_identical=all(r["identical"] for r in rows),
                                   note="Identity is exact equality of the JSON after removing the null `rsb` branch "
                                        "key that Phase 1e added to policy_detail; raw differences are listed per job."))
    for r in rows:
        print(r)


# ---- evpi -------------------------------------------------------------------------------------------------------
def cmd_evpi():
    p = ROOT / "data" / "gate2_screen_signal.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    rel = json.loads((ROOT / "data" / "gate2_screen_reliability.json").read_text(encoding="utf-8"))
    d["_note_EVPI"] = ("INVALID EVPI fields in this file (Phase 1f S4 check, 2026-09-25): experiments/gate2_screen_signal.py "
                       "took the deterministic solves' objective after the tie-break stage (the uniform sum of scenario "
                       "costs), not the primary objective. The EVPI values of record are in data/gate2_screen_reliability.json "
                       "and data/gate2_screens.md: " + ", ".join(f"{k} {v['EVPI']:.3f}" for k, v in rel.items())
                       + ". Every other field of this file is unaffected (the SAA solves used no tie-break).")
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(d["_note_EVPI"])


# ---- planner law ------------------------------------------------------------------------------------------------
def cmd_plannerlaw():
    """Both policies of every cell under the planner's own law: the no-signal job's centre marginal applied to its
    scenario costs, and the signal job's primary objective (its planner joint law has the same T-marginal up to the
    1e-9 law cleaning, at most 1.7e-9 per entry; the no-signal costs are the branch-polished ones).
    - Embedding (Lemma 0): the signal-aware optimum is never above the no-signal optimum under the planner law, up to
      the solver's relative gap of 1e-4. A larger excess would flag a solve.
    - The planner-law information value per configuration (equal-weight level mean of the p-weighted gain). It cannot
      increase with τ. The earlier signal tree contains every policy of the later one (link the copies at the extra
      nodes), and the planner's T-marginal does not depend on τ (Phase 1f E-T diagnostic)."""
    out = {}
    for d in [P1C] + sorted(RUNS.glob("p1f_*")):
        if not (d / "plan.json").exists() or "." in d.name:
            continue
        plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        acc = {}
        for c in plan["cells"]:
            kn, ks = c["jobs"]["nosig"], c["jobs"]["sig"]
            fn, fs = d / "items" / f"J_{kn}.json", d / "items" / f"J_{ks}.json"
            if not (fn.exists() and fs.exists()):
                continue
            n, s = json.loads(fn.read_text(encoding="utf-8")), json.loads(fs.read_text(encoding="utf-8"))
            if not (n.get("ok") and s.get("ok")) or s.get("primary_obj") is None:
                continue
            jn = float(np.dot(plan["jobs"][kn]["centre"], n["C"][:len(plan["jobs"][kn]["centre"])]))
            gain = jn - s["primary_obj"]
            a = acc.setdefault(c["cfg"], dict(levels={}, n=0, min_gain=np.inf, max_excess_rel_gap=-np.inf))
            lv = a["levels"].setdefault(c["level"], [0.0, 0.0])
            lv[0] += c["p"] * gain
            lv[1] += c["p"]
            a["n"] += 1
            a["min_gain"] = min(a["min_gain"], gain)
            a["max_excess_rel_gap"] = max(a["max_excess_rel_gap"], -gain / (1e-4 * abs(jn)))
        out[d.name] = {cfg: dict(value=float(np.mean([v[0] / v[1] for v in a["levels"].values()])), n_pairs=a["n"],
                                 min_gain=float(a["min_gain"]),
                                 max_excess_in_gap_units=float(a["max_excess_rel_gap"]),
                                 embedding_ok=bool(a["max_excess_rel_gap"] <= 1.0))
                       for cfg, a in acc.items()}
    note = ("Planner-law values ($M): the equal-weight level mean of the p-weighted gain J_nosig - J_sig under the "
            "planner's own law. max_excess_in_gap_units = max over cells of (J_sig - J_nosig) / (1e-4 |J_nosig|); the "
            "embedding holds within the solver gap when it is <= 1.")
    _dump("planner_law.json", dict(runs=out, note=note))
    for run, acc in out.items():
        for cfg, a in sorted(acc.items()):
            print(f"{run:16s} {cfg:22s} value {a['value']:8.3f}  min gain {a['min_gain']:+8.3f}  "
                  f"excess/gap {a['max_excess_in_gap_units']:+7.2f}  {'ok' if a['embedding_ok'] else 'VIOLATION'}")


if __name__ == "__main__":
    cmds = dict(balances=cmd_balances, gaps=cmd_gaps, rerun=cmd_rerun, evpi=cmd_evpi, plannerlaw=cmd_plannerlaw)
    cmds[sys.argv[1]]()
