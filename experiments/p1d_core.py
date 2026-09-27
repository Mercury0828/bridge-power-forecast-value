"""Phase 1d exploratory runs (docs/phase1d_plan.md v2). No claims.

Each variant re-solves the no-signal and signal policies of the frozen Phase-1c cells (experiments/r2_runs/p1c_main/
plan.json) under new model settings. Every comparison is therefore matched by (base configuration, level, report
outcome). There are no oracle jobs. Every evaluation family is evaluated exactly from the same frozen policies
(experiments/analyze_p1d.py).

Variants (configuration name = "<Phase-1c configuration>|<tag>"):
- stage (B1): the cap and full-backup delay crossed:
  - A|cap, B|cap: stage_len 4, stage_frac 0.25, backup_delay 0 (cell 10);
  - A|bkdelay, B|bkdelay: backup_delay 4 (cell 01, a diagnostic counterfactual).
  - Cells 00 and 11 are Phase-1c A/B and A+staged/B+staged.
- spine (B3): backup_mode "spine" for A, A+staged, B, B+staged.
- refurb (B5): moderate (dg_refurb 10 % of DG capex, refurb_delay 1) and stress (25 %, delay 1, derate 0.8) for the
  four configurations.
- permit (B6): residence_on False for the four configurations.
- units (B4): the Phase-1c policies re-solved exactly as registered, then the rounding-and-repair rule (round_repair).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments import p1a_core as p1a  # noqa: E402
import highspy  # noqa: E402
from model.bridge import make_params, Model, polish, recompute_costs, backup_shortfall, BLOCKS  # noqa: E402

P1C_RUN = ROOT / "experiments" / "r2_runs" / "p1c_main"
DG_CAPEX = 2.0 / 2.1                                    # $M per MW-COP (Params.capex["DG"])
STAGED = dict(stage_len=4, stage_frac=0.25)
MOD = dict(dg_refurb=round(0.10 * DG_CAPEX, 6), refurb_delay=1)
STRESS = dict(dg_refurb=round(0.25 * DG_CAPEX, 6), refurb_delay=1, refurb_derate=0.8)

# configuration name -> (Phase-1c configuration whose cells are reused, overrides on top of the Phase-1c settings)
VARIANTS = {
    "stage": {"A|cap": ("A", dict(STAGED, backup_delay=0)), "A|bkdelay": ("A", dict(backup_delay=4)),
              "B|cap": ("B", dict(STAGED, backup_delay=0)), "B|bkdelay": ("B", dict(backup_delay=4))},
    "spine": {f"{c}|spine": (c, dict(backup_mode="spine")) for c in core.CONFIGS},
    "refurb": dict([(f"{c}|refurb_mod", (c, MOD)) for c in core.CONFIGS]
                   + [(f"{c}|refurb_stress", (c, STRESS)) for c in core.CONFIGS]),
    "permit": {f"{c}|permit": (c, dict(residence_on=False)) for c in core.CONFIGS},
    "units": {f"{c}|units": (c, {}) for c in core.CONFIGS},
    # ---- Phase 1e (docs/phase1e_plan.md v2) ----
    # E1: purchased standby on the five-quarter spine lead (the lead-consistent reference) plus rented standby
    "e1": {f"{c}|spine+rsb": (c, dict(backup_mode="spine", rsb_on=True)) for c in core.CONFIGS},
    # E1 paired stresses, A+staged only
    "e1s": {"A+staged|spine+rsb|low": ("A+staged", dict(backup_mode="spine", rsb_on=True, rsb_price_mult=0.5,
                                                          rsb_install=0.0)),
            "A+staged|spine+rsb|high": ("A+staged", dict(backup_mode="spine", rsb_on=True, rsb_price_mult=1.5,
                                                           rsb_install=30.0))},
    # E4a: the staging 2x2 under the lead-consistent spine reference (cells 10 and 01; 00 and 11 are p1d_spine)
    "e4a": {"A|cap|spine": ("A", dict(STAGED, backup_delay=0, backup_mode="spine")),
            "A|bkdelay|spine": ("A", dict(backup_delay=4, backup_mode="spine")),
            "B|cap|spine": ("B", dict(STAGED, backup_delay=0, backup_mode="spine")),
            "B|bkdelay|spine": ("B", dict(backup_delay=4, backup_mode="spine"))},
    # ---- Phase 1f (docs/phase1f_plan.md; D-026). A third entry is the signal setting of the name's signal jobs. ----
    # E-C: the CCHP ablation (no absorption chillers; cooling by electric chillers)
    "nocchp": {f"{c}|nocchp": (c, dict(abs_on=False)) for c in core.CONFIGS},
    # E-N: the naive planner that treats the estimate category as exact
    "naive": {f"{c}|naive": (c, {}, dict(planner="exact")) for c in core.CONFIGS},
    # E-T: the estimate arrives at node tau (Phase 1c: 4); same accuracy families and thresholds
    "tau2": {f"{c}|tau2": (c, {}, dict(tau=2)) for c in core.CONFIGS},
    "tau6": {f"{c}|tau6": (c, {}, dict(tau=6)) for c in core.CONFIGS},
    # E-Z: campus size (IT ramp and the rental cap scaled together)
    "size075": {f"{c}|size075": (c, dict(it_ramp=((5, 25.0), (9, 50.0), (13, 75.0)), cap_R=50.0))
                for c in core.CONFIGS},
    "size300": {f"{c}|size300": (c, dict(it_ramp=((5, 100.0), (9, 200.0), (13, 300.0)), cap_R=200.0))
                for c in core.CONFIGS},
}

# ---- Phase 1f E-S: techno-economic one-at-a-time levels (docs/phase1f_sources.md §2; fixed before any run) ----------
ES_PRICE = {"gas": {"A": (2.50, 6.50), "B": (4.00, 8.00)},          # $/MMBtu, lo / hi
            "diesel": {"A": (2.50, 4.70), "B": (2.70, 4.90)},       # $/gal
            "grid": {"A": (35.0, 70.0), "B": (55.0, 100.0)}}        # $/MWh


def _default_params():
    from model.bridge import Params
    return Params()


ES_CAPEX_MULT = {"lo": {"GE": 2280 / 2662.5, "DG": 800 / (2000 / 2.1), "ABS": 1630 / 1749, "BESS": 850 / 915.6},
                 "hi": {"GE": 3150 / 2662.5, "DG": 1190 / (2000 / 2.1), "ABS": 2360 / 1749, "BESS": 1320 / 915.6}}
ES_LIFE_Q = {"lo": {"GE": 60, "DG": 60, "ABS": 80, "BESS": 40}, "hi": {"GE": 100, "DG": 120, "ABS": 100, "BESS": 80}}
EF_GAS = 53.06e-3                                                     # t CO2 / MMBtu (EPA hub 2025)
EF_DIESEL_GAL = 73.96e-3 * 5.772 / 42                                 # t CO2 / gal
EF_GRID = {"A": 733.9 * 0.000453592, "B": 593.4 * 0.000453592}        # t CO2 / MWh (eGRID2023 ERCT, SRVC total)
ES_CO2 = (25.0, 286.0)                                                # $/t CO2, lo / hi


def _es_capex(level):
    d = _default_params()
    m = ES_CAPEX_MULT[level]
    return dict(capex={j: d.capex[j] * m[j] for j in ("GE", "DG", "ABS", "BESS")}, c_backup=d.c_backup * m["DG"])


def _es_co2(ctx, p):
    base = dict(A=dict(gas=3.50, grid=50.0, diesel=3.60), B=dict(gas=5.00, grid=75.0, diesel=3.70))[ctx]
    return dict(gas=base["gas"] + p * EF_GAS, diesel=base["diesel"] + p * EF_DIESEL_GAL,
                grid=base["grid"] + p * EF_GRID[ctx])


# ---- Phase 1f E-R: four operating blocks (morning, afternoon, evening, late night; docs/phase1f_sources.md §3) --------
# multipliers on the season's day / night COP; afternoon and evening derived so the half-mean of 1/COP is preserved
_BLOCK_M = {("A", "summer"): (1.070, 1.050), ("B", "summer"): (1.050, 1.040),
            ("A", "winter"): (1.020, 1.010), ("B", "winter"): (1.000, 1.000)}   # (morning, late night)


def _four_block_eir(ctx, season):
    """EIR multipliers (1/COP relative to the 2-block value) for (morning, afternoon, evening, late night)."""
    if season in ("spring", "fall"):
        s, w = _four_block_eir(ctx, "summer"), _four_block_eir(ctx, "winter")
        return tuple((a + b) / 2 for a, b in zip(s, w))
    morn, late = _BLOCK_M[ctx, season]
    return (1 / morn, 2 - 1 / morn, 2 - 1 / late, 1 / late)


def cop_four_blocks(ctx):
    d = _default_params()
    out = []
    for (day, night), season in zip(d.cop_e, ("winter", "spring", "summer", "fall")):
        r = _four_block_eir(ctx, season)
        out.append((day / r[0], day / r[1], night / r[2], night / r[3]))
    return tuple(out), d.H / 2


def _per_cfg(fn):
    return {c: fn(c, core.base_ctx(c)) for c in core.CONFIGS}


VARIANTS.update({
    "s_disc": {f"{c}|disc_{lv}": (c, dict(disc_year=r)) for c in core.CONFIGS for lv, r in (("lo", 0.05), ("hi", 0.11))},
    **{f"s_{par}": {f"{c}|{par}_{lv}": (c, {par: ES_PRICE[par][core.base_ctx(c)][i]})
                    for c in core.CONFIGS for i, lv in enumerate(("lo", "hi"))}
       for par in ("gas", "diesel", "grid")},
    "s_capex": {f"{c}|capex_{lv}": (c, _es_capex(lv)) for c in core.CONFIGS for lv in ("lo", "hi")},
    "s_life": {f"{c}|life_{lv}": (c, dict(life_q=dict(ES_LIFE_Q[lv]))) for c in core.CONFIGS for lv in ("lo", "hi")},
    "s_rent": {f"{c}|rent_{lv}": (c, dict(p_R=r)) for c in core.CONFIGS for lv, r in (("lo", 17.0), ("hi", 35.0))},
    "s_eff": {f"{c}|eff": (c, dict(ge_hr=round(8.404 * 1.10, 6), dg_gal_per_mwh=round(70.0 * 1.10, 6)))
              for c in core.CONFIGS},
    "s_co2": {f"{c}|co2_{lv}": (c, _es_co2(core.base_ctx(c), p)) for c in core.CONFIGS
              for lv, p in (("lo", ES_CO2[0]), ("hi", ES_CO2[1]))},
    "blocks4": {f"{c}|blocks4": (c, dict(zip(("cop_e", "H"), cop_four_blocks(core.base_ctx(c)))))
                for c in core.CONFIGS},
})

# Phase 1g (docs/phase1g_plan.md; D-030): G1 rental carry-over at staged connection (staged configurations only) and G2
# limited gas rentals at three declared price levels ($/kW-month).
GR_PRICES = (25, 35, 50)
VARIANTS.update({
    "carry": {f"{c}|carry": (c, dict(stage_rent_carry=True)) for c in ("A+staged", "B+staged")},
    **{f"gasrent_{p}": {f"{c}|gasrent_{p}": (c, dict(gr_on=True, p_GR=float(p))) for c in core.CONFIGS}
       for p in GR_PRICES},
})


# B4 declared units (docs/phase1d_plan.md v2): MW-e (GE, DG at COP), MWth (ABS), MW (BESS), counted MW-DCC (standby)
UNITS = {"GE": 4.455, "DG": 2.1, "ABS": 3.517, "BESS": 1.0, "SB": 2.5}
BU_UNIT = 2.5
RENT_UNIT = 2.0
SNAP = 1e-6


def params_for(cfg, over):
    return make_params(core.base_ctx(cfg), **dict(p1a.ECON["main"], **core.CONFIGS[cfg], **over))


def job_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def signal_law(centre, thresholds, sig):
    """Phase 1f E-N/E-T: the planner law of a signal job, recomputed from the cell's frozen centre and thresholds with
    the variant's signal time tau and planner kernel ("registered": the sigma-uniform mixture; "exact": the naive
    planner that treats the estimate category as exact)."""
    from model.signal import joint_law, clean_law
    th = tuple(thresholds)
    K = core.exact_kernel(th) if sig.get("planner", "registered") == "exact" else core.planner_kernel(th)
    tau = int(sig.get("tau", core.TAU))
    return dict(law=clean_law(joint_law(np.array(centre), K, tau)), tau=tau,
                planner=sig.get("planner", "registered"))


def frozen_p1c_plan():
    return json.loads((P1C_RUN / "plan.json").read_text(encoding="utf-8"))


def plan(variant):
    """-> (cells, jobs). Cells mirror the frozen Phase-1c cells of the reused configuration, one per variant
    configuration, with the same report outcome, probability, thresholds and planner laws. Batch variants (Phase 1e
    E2/E4) derive from existing runs (plan_batch)."""
    if variant in BATCH_SOURCES:
        return plan_batch(variant)
    p1c = frozen_p1c_plan()
    cells, jobs = [], {}
    for name, entry in VARIANTS[variant].items():
        base, over = entry[0], entry[1]
        sig = entry[2] if len(entry) > 2 else None             # Phase 1f: dict(tau=..., planner="exact")
        for c in p1c["cells"]:
            if c["cfg"] != base:
                continue
            keys = {}
            for kind in ("nosig", "sig"):
                src = p1c["jobs"][c["jobs"][kind]]
                spec = dict(src, variant=variant, name=name, over=over, p1c_key=c["jobs"][kind])
                if variant == "units":
                    spec["kind"] = f"units_{kind}"
                if sig and kind == "sig":
                    spec.update(signal_law(p1c["jobs"][c["jobs"]["nosig"]]["centre"], c["thresholds"], sig))
                k = job_key(spec)
                jobs.setdefault(k, spec)
                keys[kind] = k
            cell = dict(c, cfg=name, base=base, jobs=keys)
            if sig:
                cell["tau"] = int(sig.get("tau", core.TAU))
                cell["planner"] = sig.get("planner", "registered")
            cells.append(cell)
    return cells, jobs


# ---- B4: rounding and repair ------------------------------------------------------------------------------------
def ceil_units(x, unit, cap=None):
    n = math.ceil(x / unit - SNAP) if x > 0 else 0
    v = max(n, 0) * unit
    if cap is not None and v > cap + 1e-9:
        v = math.floor(cap / unit + SNAP) * unit if x <= cap else v
    return float(v)


def round_repair(P, pm, time_limit=1800.0):
    """The B4 rule applied to a polished single-copy policy `pm` (spine fixed, branches optimized). Every physical
    decision is rounded UP to whole units after a common snap:
    - spine orders (root included);
    - spine ST/NR rentals;
    - branch cohort retention (capped by the rounded order);
    - purchased standby;
    - staged branch rentals.
    Dispatch is then re-optimized (an LP) with every physical decision and the spine operations fixed.
    -> (repaired model, status, rounding summary)."""
    sv = pm.spine_values()
    rs = dict(sv)
    added = {}
    for j in P.assets:
        u = UNITS[j]
        for k in range(P.Q + 1):
            key = f"x|{j}|{k}"
            new = ceil_units(sv[key], u)
            added[j] = added.get(j, 0.0) + new - sv[key]
            rs[key] = new
        if P.int_root and j in ("GE", "DG"):
            rs[f"n|{j}"] = int(round(rs[f"x|{j}|0"] / (P.ge_unit if j == "GE" else P.dg_unit)))
    for q in range(1, P.Q + 1):
        for key in (f"RST|{q}", f"RNR|{q}"):
            if key in rs:
                new = ceil_units(sv[key], RENT_UNIT, cap=P.cap_R)
                added["rental_spine"] = added.get("rental_spine", 0.0) + new - sv[key]
                rs[key] = new
    m2 = Model(P, fix=rs, time_limit=time_limit)
    h = m2.h
    for T in range(1, P.Q + 1):
        for j in P.assets:
            for k in range(T):
                y = pm.val(pm.v["y"][T][j][k])
                yr = min(ceil_units(y, UNITS[j]), rs[f"x|{j}|{k}"])
                h.changeColBounds(m2.v["y"][T][j][k].index, yr, yr)
        for (_, _, var), (_, _, var2) in zip(pm.bu_orders[T], m2.bu_orders[T]):
            b = ceil_units(pm.val(var), BU_UNIT)
            added["standby"] = added.get("standby", 0.0) + b - pm.val(var)
            h.changeColBounds(var2.index, b, b)
        for q in range(T, min(T + P.stage_len, P.Q + 1)):
            if (T, q) in pm.v.get("brst", {}):
                r = ceil_units(pm.val(pm.v["brst"][T, q]), RENT_UNIT, cap=P.cap_R)
                added["rental_branch"] = added.get("rental_branch", 0.0) + r - pm.val(pm.v["brst"][T, q])
                h.changeColBounds(m2.v["brst"][T, q].index, r, r)
    h.minimize(sum(m2.C[T] for T in range(1, P.Q + 2)))
    return m2, m2._status(), added


def _units_job(spec, time_limit):
    """Re-solve the frozen Phase-1c policy exactly as registered, check it against the frozen item, then apply B4."""
    from model.signal import SignalModel
    P = core.params(spec["cfg"])
    item = json.loads((P1C_RUN / "items" / f"J_{spec['p1c_key']}.json").read_text(encoding="utf-8"))
    presolve = item.get("repair", {}).get("tiebreak_presolve") != "off"          # D-022a job only
    if spec["kind"] == "units_nosig":
        m = Model(P, time_limit=time_limit)
        m.solve_saa(np.array(spec["centre"]), tiebreak_mode="lp", tiebreak_presolve=presolve)
        pms = [polish(P, m.spine_values(), time_limit=time_limit)[0]]
        frozen = [item["C"]]
    else:
        sm = SignalModel(P, core.NCAT, spec.get("tau", core.TAU), time_limit=time_limit)
        sm.solve_saa(spec["law"], tiebreak=True, tiebreak_mode="lp")
        pms = [polish(P, c.spine_values(), time_limit=time_limit)[0] for c in sm.copies]
        frozen = item["costs"]
    out = dict(ok=True, costs=[], costs_rounded=[], cost_match=0.0, added=[], repair_status=[], bookkeeping=0.0,
               shortfall=0.0)
    for pm, fr in zip(pms, frozen):
        C = pm.C_values()
        out["cost_match"] = max(out["cost_match"], float(np.max(np.abs(C - np.array(fr)))))
        m2, st, added = round_repair(P, pm, time_limit)
        out["repair_status"].append(st["status"])
        if not st["ok"]:
            out["ok"] = False
            continue
        C2 = m2.C_values()
        out["costs"].append(C.tolist())
        out["costs_rounded"].append(C2.tolist())
        out["added"].append(added)
        out["bookkeeping"] = max(out["bookkeeping"], float(np.max(np.abs(C2 - recompute_costs(P, m2)))))
        out["shortfall"] = max(out["shortfall"], float(backup_shortfall(P, m2)))
    return out


# ---- Phase 1e E2: the "batch" whole-unit rule (docs/phase1e_plan.md) ----------------------------------------------------
# A batch variant re-solves every policy of an existing run exactly as it was produced, checks it against the stored item,
# and applies batch_repair. The sources are the frozen Phase-1c run and Phase-1d/1e runs.
BATCH_SOURCES = {
    "e2": ["p1c_main"],                      # the registered benchmark
    "e4b_spine": ["p1d_spine"],              # joint: lead-consistent standby + whole units
    "e4a2": ["p1d_stage"],                   # the staging 2x2 under whole units (cells 10 and 01)
    "e4b_rsb": ["p1e_e1"],                   # joint: rental-standby contract + whole units (after E1)
}
RUNS = ROOT / "experiments" / "r2_runs"


def plan_batch(variant):
    """-> (cells, jobs) for a batch variant: one batch job per job of each source run, and the source cells mirrored."""
    cells, jobs = [], {}
    for run in BATCH_SOURCES[variant]:
        src = json.loads((RUNS / run / "plan.json").read_text(encoding="utf-8"))
        keymap = {}
        for k, s0 in src["jobs"].items():
            kind = s0["kind"]
            if kind not in ("nosig", "sig"):
                continue
            spec = dict(s0, kind=f"batch_{kind}", variant=variant, source_run=run, source_key=k,
                        over=s0.get("over", {}), name=f"{s0.get('name', s0['cfg'])}|batch")
            nk = job_key(spec)
            jobs[nk] = spec
            keymap[k] = nk
        for c in src["cells"]:
            name = f"{c['cfg']}|batch"
            cells.append(dict(c, cfg=name, base=c.get("base", c["cfg"]),
                              jobs=dict(nosig=keymap[c["jobs"]["nosig"]], sig=keymap[c["jobs"]["sig"]])))
    return cells, jobs


def batch_repair(P, pm, time_limit=1800.0, gap=1e-6):
    """E2 rule on a polished single-copy policy `pm`:
    - spine orders are rounded up on the CUMULATIVE trajectory, so small orders are batched into whole units where the
      cumulative crosses a unit boundary; installed capacity never falls below the continuous policy's, and it depends
      only on the history;
    - spine rental levels are rounded up per quarter to RENT_UNIT;
    - branch retention, purchased standby and staged branch rentals are then re-optimized as whole units (a MILP with
      the batched spine fixed and continuous dispatch). The objective is the uniform sum over branches: training-free,
      since branches are independent given the spine.
    -> (model, status)."""
    sv = pm.spine_values()
    rs = dict(sv)
    for j in P.assets:
        u = UNITS[j]
        cum = prev = 0.0
        for k in range(P.Q + 1):
            cum += sv[f"x|{j}|{k}"]
            c_r = ceil_units(cum, u)
            rs[f"x|{j}|{k}"] = max(c_r - prev, 0.0)
            prev = max(prev, c_r)
        if P.int_root and j in ("GE", "DG"):
            rs[f"n|{j}"] = int(round(rs[f"x|{j}|0"] / (P.ge_unit if j == "GE" else P.dg_unit)))
    for q in range(1, P.Q + 1):
        for key in (f"RST|{q}", f"RNR|{q}"):
            if key in rs:
                rs[key] = ceil_units(sv[key], RENT_UNIT, cap=P.cap_R)
    if P.rsb_on:
        # rented standby on the spine: levels rounded up to standby units; minimal starts recomputed; a forward pass
        # restores the minimum commitment (a contract started in the last M - 1 quarters must still be held)
        r = {q: ceil_units(sv[f"RSB|{q}"], BU_UNIT, cap=P.rsb_cap) for q in range(1, P.Q + 1)}
        st = {}
        for q in range(1, P.Q + 1):
            held = sum(st.get(j, 0.0) for j in range(max(1, q - P.rsb_min_q + 1), q))
            r[q] = max(r[q], held)
            st[q] = max(0.0, r[q] - r.get(q - 1, 0.0))
            rs[f"RSB|{q}"], rs[f"RSBs|{q}"] = r[q], st[q]
    m2 = Model(P, fix=rs, time_limit=time_limit, gap=gap)
    h = m2.h
    for T in range(1, P.Q + 1):
        for j in P.assets:
            u = UNITS[j]
            for k in range(T):
                n = m2._var(0, int(round(rs[f"x|{j}|{k}"] / u)), integer=True)
                h.addConstr(m2.v["y"][T][j][k] - u * n == 0)
        for (_, _, var) in m2.bu_orders[T]:
            n = m2._var(0, highspy.kHighsInf, integer=True)
            h.addConstr(var - BU_UNIT * n == 0)
        for q in range(T, min(T + P.stage_len, P.Q + 1)):
            if (T, q) in m2.v.get("brst", {}):
                n = m2._var(0, int(P.cap_R // RENT_UNIT), integer=True)
                h.addConstr(m2.v["brst"][T, q] - RENT_UNIT * n == 0)
        if P.rsb_on:
            for q in range(T + 1, P.Q + 1):
                for key in ("RSBb", "RSBbs"):
                    n = m2._var(0, int(P.rsb_cap // BU_UNIT) + 1, integer=True)
                    h.addConstr(m2.v[key][T][q] - BU_UNIT * n == 0)
    h.minimize(sum(m2.C[T] for T in range(1, P.Q + 2)))
    return m2, m2._status()


QTY_KEYS = ("GE_MWh", "DG_MWh", "rental_MWh", "grid_MWh", "ABS_MWh_th", "chiller_MWh_th")


def scenario_quantities(P, m):
    """Per-scenario physical totals (index T - 1; undiscounted over quarters 1..Q). Scenario T uses spine quarters
    1..T-1 and branch quarters T..Q. MWh-e for generation and grid; MWh-th for cooling."""
    Q, H, val = P.Q, P.H, m.val
    sp = {k: np.zeros(Q + 1) for k in QTY_KEYS}
    for q in range(1, Q + 1):
        for b in P.blocks:
            sp["GE_MWh"][q] += H * val(m.v["gGE"][q, b])
            sp["DG_MWh"][q] += H * val(m.v["gDG"][q, b])
            sp["rental_MWh"][q] += H * (val(m.v["gNR"][q, b]) + val(m.v["gST"][q, b]))
            sp["ABS_MWh_th"][q] += H * val(m.v["a"][q, b])
            sp["chiller_MWh_th"][q] += H * val(m.v["e"][q, b])
    out = {k: np.zeros(Q + 1) for k in QTY_KEYS}
    for T in range(1, Q + 2):
        for k in QTY_KEYS:
            out[k][T - 1] = sp[k][1:min(T, Q + 1)].sum()
        if T <= Q:
            for q in range(T, Q + 1):
                for b in P.blocks:
                    out["GE_MWh"][T - 1] += H * val(m.v["bg"][T, q, b])
                    out["grid_MWh"][T - 1] += H * val(m.v["m"][T, q, b])
                    out["ABS_MWh_th"][T - 1] += H * val(m.v["ba"][T, q, b])
                    out["chiller_MWh_th"][T - 1] += H * val(m.v["be"][T, q, b])
                    if (T, q, b) in m.v.get("bgd", {}):
                        out["DG_MWh"][T - 1] += H * val(m.v["bgd"][T, q, b])
                        out["rental_MWh"][T - 1] += H * val(m.v["bgs"][T, q, b])
    return {k: v.tolist() for k, v in out.items()}


def resolve_policy(P, spec, item, time_limit=1800.0):
    """Re-solve a stored nosig/sig policy exactly as produced (the LP tie-break; presolve off in the second stage if the
    item records a repair, the D-022a rule). -> (list of polished models, cost_match against the item)."""
    from model.signal import SignalModel
    presolve = "repair" not in item
    rule = str(item.get("repair", {}).get("rule", ""))
    mode = "lp_pure" if "pure LP" in rule else "lp"              # Phase-1e escalation rule (p1d_repair.py)
    base_kind = spec["kind"].replace("batch_", "")
    if base_kind == "nosig":
        m = Model(P, time_limit=time_limit)
        m.solve_saa(np.array(spec["centre"]), tiebreak_mode=mode, tiebreak_presolve=presolve)
        pms = [polish(P, m.spine_values(), time_limit=time_limit)[0]]
        frozen = [item["C"]]
    else:
        sm = SignalModel(P, core.NCAT, spec.get("tau", core.TAU), time_limit=time_limit)
        sm.solve_saa(spec["law"], tiebreak=True, tiebreak_mode=mode, tiebreak_presolve=presolve)
        pms = [polish(P, c.spine_values(), time_limit=time_limit)[0] for c in sm.copies]
        frozen = item["costs"]
    cm = max(float(np.max(np.abs(pm.C_values() - np.array(fr)))) for pm, fr in zip(pms, frozen))
    return pms, cm


def _batch_job(spec, time_limit):
    P = params_for(spec["cfg"], spec["over"])
    item = json.loads((RUNS / spec["source_run"] / "items" / f"J_{spec['source_key']}.json").read_text(encoding="utf-8"))
    pms, cm = resolve_policy(P, spec, item, time_limit)
    out = dict(ok=True, costs=[], costs_rounded=[], cost_match=cm, repair_status=[], bookkeeping=0.0, shortfall=0.0,
               quantities=[], quantities_rounded=[])
    for pm in pms:
        m2, st = batch_repair(P, pm, time_limit)
        out["repair_status"].append(st["status"])
        if not st["ok"]:
            out["ok"] = False
            continue
        out["costs"].append(pm.C_values().tolist())
        out["costs_rounded"].append(m2.C_values().tolist())
        out["quantities"].append(scenario_quantities(P, pm))
        out["quantities_rounded"].append(scenario_quantities(P, m2))
        out["bookkeeping"] = max(out["bookkeeping"], float(np.max(np.abs(m2.C_values() - recompute_costs(P, m2)))))
        out["shortfall"] = max(out["shortfall"], float(backup_shortfall(P, m2)))
    return out


def run_job(spec, time_limit=1800.0):
    try:
        if spec["kind"].startswith("batch_"):
            return dict(spec_key=job_key(spec), **_batch_job(spec, time_limit))
        if spec["variant"] == "units":
            return dict(spec_key=job_key(spec), **_units_job(spec, time_limit))
        P = params_for(spec["cfg"], spec["over"])
        res = core._run_job(spec, time_limit, P=P)
        res["spec_key"] = job_key(spec)
        return res
    except Exception as e:                                        # noqa: BLE001
        return dict(spec_key=job_key(spec), ok=False, error=f"{type(e).__name__}: {e}")
