"""Phase-1d accounting items A1–A4 (docs/phase1d_plan.md). Nothing here re-tests a claim.

    python experiments/p1d_accounting.py v0          -> data/p1d/v0.json
    python experiments/p1d_accounting.py decomp      -> data/p1d/oracle_decomposition.json
    python experiments/p1d_accounting.py d022a       -> data/p1d/d022a_verification.json
    python experiments/p1d_accounting.py mc          -> data/p1d/b_mc_allowance.json
    python experiments/p1d_accounting.py components  -> data/p1d/component_reconciliation.json
    python experiments/p1d_accounting.py repcells    -> data/p1d/representative_cells.json
    python experiments/p1d_accounting.py chrono      -> data/p1d/backup_chronology.json (B3, spine mode)
    python experiments/p1d_accounting.py thresholds  -> data/p1d/variant_thresholds.json
    python experiments/p1d_accounting.py unitscheck  -> data/p1d/units_accounting_check.json

v0 is the true-law no-signal benchmark: 8 MILPs (configuration × level, Δ = 0) at MIP gap 1e-5, with dual bounds, each
polished and evaluated under every family's cleaned evaluation law. The only other re-solves are:
- the D-022a model export and optimal-face LPs (A1);
- the representative cells, re-solved deterministically to extract dispatch quantities (A4c). Their costs are checked
  against the frozen items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from model import evidence as ev  # noqa: E402
from model.bridge import BLOCKS, COMPONENTS, Model, polish, recompute_costs, backup_shortfall  # noqa: E402
from model.signal import clean_law, joint_law  # noqa: E402

OUT = ROOT / "data" / "p1d"
RUN = ROOT / "experiments" / "r2_runs" / "p1c_main"
D022A_KEY = "39cf540b499e5e769320"


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"written data/p1d/{name}")


def _plan_items():
    plan = json.loads((RUN / "plan.json").read_text(encoding="utf-8"))
    load = lambda k: json.loads((RUN / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    return plan, load


def eval_law(cell_thresholds, cfg, level, fam, shift=0):
    pt = core.truth(cfg, level, shift)
    return clean_law(joint_law(pt, core.true_kernel(tuple(cell_thresholds), fam, pt), core.TAU))


def marginal(law):
    return np.asarray(law["pre"]) + sum(np.asarray(q) for q in law["post"])


# ---- A2: the true-law no-signal benchmark v0 -------------------------------------------------------------------
def _v0_job(args):
    cfg, level = args
    P = core.params(cfg)
    pt = ev.clean(core.truth(cfg, level, 0))
    m = Model(P, gap=1e-5, time_limit=3600)
    st = m.solve_saa(pt, tiebreak=False)
    info = m.h.getInfo()
    pm, pst = polish(P, m.spine_values(), time_limit=3600)
    C = pm.C_values()
    return dict(cfg=cfg, level=level, ok=bool(st["ok"] and pst["ok"]), status=st["status"], obj=st["obj"],
                dual_bound=float(info.mip_dual_bound), gap=st["gap"], C=C.tolist(),
                J_train_law=float(pt @ C), bookkeeping=float(np.max(np.abs(C - recompute_costs(P, pm)))))


def cmd_v0():
    jobs = [(c, lv) for c in core.CONFIGS for lv in core.LEVELS]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = pool.map(_v0_job, jobs)
    _dump("v0.json", res)


# ---- A2: the decomposition ---------------------------------------------------------------------------------------
def cmd_decomp():
    plan, _ = _plan_items()
    rows = json.loads((ROOT / "data" / "p1c_rows.json").read_text(encoding="utf-8"))
    v0 = {(r["cfg"], r["level"]): r for r in json.loads((OUT / "v0.json").read_text(encoding="utf-8"))}
    cells = {(c["cfg"], c["level"], c["r"]): c for c in plan["cells"]}
    out, checks = {}, []
    for cfg in core.CONFIGS:
        for fam in core.FAMILIES:
            per = {}
            for lv in core.LEVELS:
                R = [r for r in rows if r["cfg"] == cfg and r["family"] == fam and r["shift"] == 0
                     and r["level"] == lv and "J_oracle" in r]
                if not R:
                    continue
                p = np.array([r["p"] for r in R])
                w = p / p.sum()
                # v0 evaluated under each cell's own (cleaned) evaluation law, matched to N and S
                C0 = np.array(v0[cfg, lv]["C"])
                J0 = np.array([marginal(eval_law(cells[cfg, lv, r["r"]]["thresholds"], cfg, lv, fam)) @ C0 for r in R])
                N, S = w @ [r["J_nosig"] for r in R], w @ [r["J_sig"] for r in R]
                vY, LY = w @ [r["J_oracle"] for r in R], w @ [r["L_oracle"] for r in R]
                v0p, L0 = float(w @ J0), v0[cfg, lv]["dual_bound"]
                per[lv] = dict(N=N, S=S, v0=v0p, L0=L0, vY=vY, LY=LY, d_info=N - S, G0=N - v0p, V_true=v0p - vY,
                               R_Y=S - vY, gap_to_oracle=N - vY, V_true_cert=(L0 - vY, v0p - LY))
                if fam == "uninf":
                    checks += [dict(cfg=cfg, level=lv, r=r["r"], J_oracle_minus_v0=r["J_oracle"] - j0,
                                    L_oracle_minus_L0=r["L_oracle"] - L0) for r, j0 in zip(R, J0)]
            if per:
                keys = ("N", "S", "v0", "vY", "d_info", "G0", "V_true", "R_Y", "gap_to_oracle")
                avg = {k: float(np.mean([per[lv][k] for lv in per])) for k in keys}
                avg["identity_residual"] = avg["d_info"] - (avg["V_true"] + avg["G0"] - avg["R_Y"])
                avg["share_of_gap_closed"] = avg["d_info"] / avg["gap_to_oracle"] if avg["gap_to_oracle"] else None
                out[f"{cfg}|{fam}"] = dict(mean=avg, by_level=per)
    _dump("oracle_decomposition.json", dict(decomposition=out, uninf_checks=checks,
                                            max_abs_uninf_primal=max(abs(c["J_oracle_minus_v0"]) for c in checks),
                                            max_abs_uninf_dual=max(abs(c["L_oracle_minus_L0"]) for c in checks)))


# ---- A1: D-022a verification ----------------------------------------------------------------------------------------
def _second_stage(spec, presolve_off, write_to=None):
    P = core.params(spec["cfg"])
    centre = np.array(spec["centre"])
    m = Model(P, time_limit=1800)
    h = m.h
    primary = sum(centre[i] * m.C[i + 1] for i in range(P.Q + 1) if centre[i] > 1e-12)
    h.minimize(primary)
    z = h.getInfo().objective_function_value
    tol = 1e-7 * abs(z) + 1e-6
    ints_before = [float(h.val(v)) for v in m._ints]
    m.fix_integers()
    h.addConstr(primary <= z + tol)
    tb = sum(m.C[T] for T in range(1, P.Q + 2))
    h.setObjective(tb)
    if write_to is not None:
        h.writeModel(str(write_to))
    if presolve_off:
        h.setOptionValue("presolve", "off")
    h.minimize(tb)
    return m, P, centre, primary, tb, z, tol, ints_before


def cmd_d022a():
    plan, load = _plan_items()
    spec = plan["jobs"][D022A_KEY]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="d022a_"))
    res = {}
    hashes = {}
    for label, off in (("registered", False), ("repaired", True)):
        f = tmp / f"{label}.mps"
        m, P, centre, primary, tb, z, tol, ints = _second_stage(spec, off, write_to=f)
        hashes[label] = hashlib.sha256(f.read_bytes()).hexdigest()
        st = m._status()
        info = m.h.getInfo()
        res[label] = dict(status=st["status"], z_primary=z, tol=tol, ints_hash=hashlib.sha256(
            json.dumps([round(x, 9) for x in ints]).encode()).hexdigest()[:16])
        if label == "repaired":
            C = m.C_values()
            prim_val = float(sum(centre[i] * C[i] for i in range(P.Q + 1) if centre[i] > 1e-12))
            res[label].update(tiebreak_obj=float(info.objective_function_value),
                              max_primal_infeasibility=float(info.max_primal_infeasibility),
                              primary_value=prim_val, primary_bound=z + tol, primary_ok=prim_val <= z + tol + 1e-9,
                              ints_fixed_max_frac=max(abs(m.h.val(v) - round(m.h.val(v))) for v in m._ints))
            # A1(c): the remaining optimal face (integers fixed, primary bound, tie-break objective at its optimum)
            w = float(info.objective_function_value)
            m.h.addConstr(tb <= w + 1e-7 * abs(w) + 1e-6)
            cell = next(c for c in plan["cells"] if c["jobs"]["nosig"] == D022A_KEY)
            law = marginal(eval_law(cell["thresholds"], cell["cfg"], cell["level"], "sym2"))
            JQ = sum(law[i] * m.C[i + 1] for i in range(P.Q + 1) if law[i] > 0)
            m.h.maximize(JQ)
            jmax = float(m.h.getInfo().objective_function_value)
            st_max = m._status()["status"]
            m.h.minimize(JQ)
            jmin = float(m.h.getInfo().objective_function_value)
            st_min = m._status()["status"]
            item = load(D022A_KEY)
            J_pol = float(law @ np.array(item["C"]))
            v0 = {(r["cfg"], r["level"]): r for r in json.loads((OUT / "v0.json").read_text(encoding="utf-8"))}
            L0 = v0[cell["cfg"], cell["level"]]["dual_bound"]
            p = cell["p"]
            res["face"] = dict(cell=dict(cfg=cell["cfg"], level=cell["level"], r=cell["r"], p=p),
                               max_unpolished_JQ=jmax, status_max=st_max, min_unpolished_JQ=jmin, status_min=st_min,
                               J_polished_item=J_pol, L0_true_law_no_signal=L0,
                               bound_on_policy_JQ=(L0, jmax),
                               max_change_in_grid_statistic=p / 2 * (jmax - L0),
                               note="any policy the rule could return has L0 <= J_Q <= max_unpolished_JQ; the grid "
                                    "statistic weighs this cell p/2")
    res["model_files_identical"] = hashes["registered"] == hashes["repaired"]
    res["model_hashes"] = hashes
    res["ints_identical"] = res["registered"]["ints_hash"] == res["repaired"]["ints_hash"]
    _dump("d022a_verification.json", res)


# ---- A3: B's paired Monte Carlo allowance ---------------------------------------------------------------------------
def cmd_mc():
    rows = json.loads((ROOT / "data" / "p1c_rows.json").read_text(encoding="utf-8"))
    out = {}
    for h, fam in (("H1a", "sym1"), ("H1b", "sym2")):
        per = {}
        for lv in core.LEVELS:
            R = [r for r in rows if r["cfg"] == "B" and r["family"] == fam and r["shift"] == 0 and r["level"] == lv]
            p = np.array([r["p"] for r in R])
            d = np.array([r["d_info"] for r in R])
            n = np.array([r["J_nosig"] for r in R])
            stats = {}
            for name, x in (("d_minus_0.5pct_N", d - 0.005 * n), ("d_minus_3", d - 3.0), ("d", d)):
                mu = float(p @ x)
                se = float(np.sqrt(max(p @ x ** 2 - mu ** 2, 0.0) / core.REPORT_MC))
                stats[name] = dict(mean=mu, se=se)
            per[lv] = dict(stats, p_sum=float(p.sum()), n_outcomes=len(R))
        grid = {}
        for name in ("d_minus_0.5pct_N", "d_minus_3", "d"):
            mu = float(np.mean([per[lv][name]["mean"] for lv in per]))
            se_indep = float(np.sqrt(sum(per[lv][name]["se"] ** 2 for lv in per)) / 2)
            se_corr = float(sum(per[lv][name]["se"] for lv in per) / 2)        # perfectly correlated levels (bound)
            grid[name] = dict(mean=mu, se_independent=se_indep, se_upper=se_corr,
                              margin_in_se_upper=(mu / se_corr if se_corr > 0 else None))
        out[h] = dict(by_level=per, grid=grid)
    _dump("b_mc_allowance.json", out)


# ---- A4: component reconciliation over all cells ------------------------------------------------------------------
def cmd_components():
    plan, load = _plan_items()
    S = json.loads((ROOT / "data" / "p1c_summary.json").read_text(encoding="utf-8"))
    out = {}
    for cfg in core.CONFIGS:
        for fam in core.FAMILIES:
            acc = {lv: {c: 0.0 for c in COMPONENTS} for lv in core.LEVELS}
            acc_cat = {lv: {c: [0.0] * core.NCAT for c in COMPONENTS} for lv in core.LEVELS}
            for cell in [c for c in plan["cells"] if c["cfg"] == cfg]:
                ns, sg = load(cell["jobs"]["nosig"]), load(cell["jobs"]["sig"])
                law = eval_law(cell["thresholds"], cfg, cell["level"], fam)
                pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
                for c in COMPONENTS:
                    cn = np.asarray(ns["components"][c])
                    cs = [np.asarray(sg["components"][y][c]) for y in range(core.NCAT)]
                    d_pre = pre @ (cn - cs[0])
                    d_cat = [post[y] @ (cn - cs[y]) for y in range(core.NCAT)]
                    acc[cell["level"]][c] += cell["p"] * (d_pre + sum(d_cat))
                    for y in range(core.NCAT):
                        acc_cat[cell["level"]][c][y] += cell["p"] * d_cat[y]
            comp = {c: float(np.mean([acc[lv][c] for lv in core.LEVELS])) for c in COMPONENTS}
            by_cat = {c: [float(np.mean([acc_cat[lv][c][y] for lv in core.LEVELS])) for y in range(core.NCAT)]
                      for c in COMPONENTS}
            total = sum(comp.values())
            ref = S["table"][f"{cfg}|{fam}|0"]["mean"]
            out[f"{cfg}|{fam}"] = dict(components=comp, by_category=by_cat, total=total, claim_table_mean=ref,
                                       residual=total - ref)
    _dump("component_reconciliation.json", out)


# ---- A4(b, c): representative cells, re-solved for quantities --------------------------------------------------------
DIESEL_MMBTU_PER_GAL = 5.772 / 42          # EIA distillate fuel oil heat content, HHV (5.772 MMBtu/bbl)
MMBTU_PER_MWH = 3.412


def _quantities(P, pm, T_law):
    """Expected physical quantities of a polished single-copy policy under a law over T (index i -> T = i + 1).
    Undiscounted totals over quarters 1..Q; HHV fuel basis. The service boundary (IT + other electricity,
    cooling) is identical across policies under must-energize service."""
    Q, H, val = P.Q, P.H, pm.val
    k = dict(GE_MWh_e=0.0, owned_DG_MWh_e=0.0, rental_MWh_e=0.0, grid_MWh=0.0, ABS_cooling_MWh_th=0.0,
             elec_chiller_cooling_MWh_th=0.0, service_elec_MWh=0.0, service_cooling_MWh_th=0.0)
    for i, pT in enumerate(T_law):
        if pT <= 0:
            continue
        T = i + 1
        for q in range(1, min(T, Q + 1)):                       # pre-connection quarters 1..T-1 (spine)
            for b in P.blocks:
                k["GE_MWh_e"] += pT * H * val(pm.v["gGE"][q, b])
                k["owned_DG_MWh_e"] += pT * H * val(pm.v["gDG"][q, b])
                k["rental_MWh_e"] += pT * H * (val(pm.v["gNR"][q, b]) + val(pm.v["gST"][q, b]))
                k["ABS_cooling_MWh_th"] += pT * H * val(pm.v["a"][q, b])
                k["elec_chiller_cooling_MWh_th"] += pT * H * val(pm.v["e"][q, b])
                k["service_elec_MWh"] += pT * H * (1 + P.a_oth) * val(pm.v["S"][q])
                k["service_cooling_MWh_th"] += pT * H * P.kappa * val(pm.v["S"][q])
        if T <= Q:
            for q in range(T, Q + 1):
                for b in P.blocks:
                    k["GE_MWh_e"] += pT * H * val(pm.v["bg"][T, q, b])
                    k["grid_MWh"] += pT * H * val(pm.v["m"][T, q, b])
                    k["ABS_cooling_MWh_th"] += pT * H * val(pm.v["ba"][T, q, b])
                    k["elec_chiller_cooling_MWh_th"] += pT * H * val(pm.v["be"][T, q, b])
                    k["service_elec_MWh"] += pT * H * (1 + P.a_oth) * P.L(q)
                    k["service_cooling_MWh_th"] += pT * H * P.kappa * P.L(q)
                    if (T, q, b) in pm.v.get("bgd", {}):
                        k["owned_DG_MWh_e"] += pT * H * val(pm.v["bgd"][T, q, b])
                        k["rental_MWh_e"] += pT * H * val(pm.v["bgs"][T, q, b])
    diesel_gal = (k["owned_DG_MWh_e"] + k["rental_MWh_e"]) * P.dg_gal_per_mwh
    k.update(gas_MMBtu=k["GE_MWh_e"] * P.ge_hr, diesel_gal=diesel_gal, owned_DG_MWh=k["owned_DG_MWh_e"],
             gas_fuel_MWh_HHV=k["GE_MWh_e"] * P.ge_hr / MMBTU_PER_MWH,
             diesel_fuel_MWh_HHV=diesel_gal * DIESEL_MMBTU_PER_GAL / MMBTU_PER_MWH,
             recovered_heat_used_MWh_th=k["ABS_cooling_MWh_th"] / P.cop_abs)
    k["onsite_fuel_MWh_HHV"] = k["gas_fuel_MWh_HHV"] + k["diesel_fuel_MWh_HHV"]
    return k


def _capacity(P, orders, q):
    """MW (MWth for ABS) delivered by quarter q from spine orders per node."""
    return {j: float(sum(orders[j][k] for k in range(len(orders[j])) if k <= q - P.lead[j])) for j in orders}


def _timing(orders):
    return {j: dict(root=float(v[0]), pre_signal=float(sum(v[1:core.TAU])), post_signal=float(sum(v[core.TAU:])))
            for j, v in orders.items()}


def _repcell_job(args):
    from model.signal import SignalModel
    cell, spec_ns, spec_sg, item_ns, item_sg = args
    cfg, lv = cell["cfg"], cell["level"]
    P = core.params(cfg)
    law = eval_law(cell["thresholds"], cfg, lv, "sym2")
    pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
    mT = pre + sum(post)
    # no-signal, re-solved exactly as registered
    m = Model(P, time_limit=1800)
    m.solve_saa(np.array(spec_ns["centre"]), tiebreak_mode="lp",
                tiebreak_presolve=not (item_ns.get("repair", {}).get("tiebreak_presolve") == "off"))
    pm, _ = polish(P, m.spine_values(), time_limit=1800)
    C = pm.C_values()
    ns = dict(cost_match=float(np.max(np.abs(C - np.array(item_ns["C"])))), all=_quantities(P, pm, mT),
              by_cat=[_quantities(P, pm, post[y] / post[y].sum()) if post[y].sum() > 0 else None
                      for y in range(core.NCAT)])
    # signal policy
    sm = SignalModel(P, core.NCAT, core.TAU, time_limit=1800)
    sm.solve_saa(spec_sg["law"], tiebreak=True, tiebreak_mode="lp")
    sg = dict(cost_match=0.0, by_cat=[], all={})
    for y, c in enumerate(sm.copies):
        pmy, _ = polish(P, c.spine_values(), time_limit=1800)
        Cy = pmy.C_values()
        sg["cost_match"] = max(sg["cost_match"], float(np.max(np.abs(Cy - np.array(item_sg["costs"][y])))))
        wy = post[y].sum()
        qy = _quantities(P, pmy, post[y] / wy) if wy > 0 else None
        sg["by_cat"].append(qy)
        if qy:
            for k in qy:
                sg["all"][k] = sg["all"].get(k, 0.0) + wy * qy[k]
        if y == 0 and pre.sum() > 0:
            q0 = _quantities(P, pmy, pre / pre.sum())
            for k in q0:
                sg["all"][k] = sg["all"].get(k, 0.0) + pre.sum() * q0[k]
    return dict(cfg=cfg, level=lv, r=cell["r"], p=cell["p"], reports=cell["reports"], thresholds=cell["thresholds"],
                P_cat=[float(q.sum()) for q in post], no_signal=ns, signal=sg)


def cmd_repcells():
    plan, load = _plan_items()
    jobs, info = [], []
    for cfg in core.CONFIGS:
        for lv in core.LEVELS:
            cs = [c for c in plan["cells"] if c["cfg"] == cfg and c["level"] == lv]
            c = max(cs, key=lambda x: x["p"])
            ins, isg = load(c["jobs"]["nosig"]), load(c["jobs"]["sig"])
            jobs.append((c, plan["jobs"][c["jobs"]["nosig"]], plan["jobs"][c["jobs"]["sig"]], ins, isg))
            P = core.params(cfg)
            law = eval_law(c["thresholds"], cfg, lv, "sym2")
            post = [np.asarray(q) for q in law["post"]]
            med = []
            for y in range(core.NCAT):
                cdf = np.cumsum(post[y]) / post[y].sum() if post[y].sum() > 0 else None
                med.append(int(np.searchsorted(cdf, 0.5) + 1) if cdf is not None else None)
            det = dict(no_signal=dict(timing=_timing(ins["detail"]["orders"]),
                                      installed_by_tau=_capacity(P, ins["detail"]["orders"], core.TAU)),
                       signal=[dict(timing=_timing(isg["detail"][y]["orders"]),
                                    installed_by_tau=_capacity(P, isg["detail"][y]["orders"], core.TAU),
                                    median_T=med[y],
                                    installed_by_median_T=(_capacity(P, isg["detail"][y]["orders"], med[y])
                                                           if med[y] else None),
                                    no_signal_installed_by_median_T=(_capacity(P, ins["detail"]["orders"], med[y])
                                                                     if med[y] else None))
                               for y in range(core.NCAT)])
            info.append(dict(cfg=cfg, level=lv, r=c["r"], decisions=det))
    with mp.get_context("spawn").Pool(min(8, len(jobs))) as pool:
        q = pool.map(_repcell_job, jobs)
    for d, qq in zip(info, q):
        d["quantities"] = qq
    _dump("representative_cells.json", dict(units="orders and capacity: MW electrical (GE, DG, BESS), MWth (ABS); "
                                                 "gas MMBtu HHV; diesel gallons; grid MWh; sym2, shift 0",
                                            cells=info))


# ---- B3: backup chronology under spine-ordered standby ---------------------------------------------------------------
def _chrono_policy(P, pm):
    """Per scenario T: the obligation start, the standby cohorts and purchases that cover it (order, delivery, MW),
    and the minimum quarterly residual (counted backup - B_need) over the obligation period."""
    out = []
    for T in range(1, P.Q + 1):
        q0 = T + P.bk_start
        qs = [q for q in range(q0, P.Q + 1) if P.B_need(q) > 0]
        if not qs:
            continue
        sb = [(k, k + P.lead["SB"], pm.val(pm.v["y"][T]["SB"][k])) for k in range(T) if pm.val(pm.v["y"][T]["SB"][k]) > 1e-6]
        bu = [(qo, qd, pm.val(v)) for (qo, qd, v) in pm.bu_orders[T] if pm.val(v) > 1e-6]
        dg = [(k, k + P.lead["DG"], pm.val(pm.v["y"][T]["DG"][k])) for k in range(T) if pm.val(pm.v["y"][T]["DG"][k]) > 1e-6]
        res = []
        for q in qs:
            have = sum(b for (_, qd, b) in bu if qd <= q) + sum(y for (_, qd, y) in sb if qd <= q) \
                + P.bk_ratio * sum(y for (_, qd, y) in dg if qd <= q)
            res.append(have - P.B_need(q))
        out.append(dict(T=T, obligation_start=q0, min_residual=min(res), first_quarter_residual=res[0],
                        sb_orders=[(k, d, round(y, 3)) for (k, d, y) in sb],
                        bu_orders=[(qo, qd, round(b, 3)) for (qo, qd, b) in bu],
                        sb_ordered_before_T=all(k <= T - 1 for (k, _, _) in sb),
                        bu_delivered_after_lead=all(qd - qo == P.lead["SB"] for (qo, qd, _) in bu)))
    return out


def _chrono_job(args):
    from experiments import p1d_core as p1d
    from model.signal import SignalModel
    cell, spec_ns, spec_sg = args
    P = p1d.params_for(spec_ns["cfg"], spec_ns["over"])
    m = Model(P, time_limit=1800)
    st = m.solve_saa(np.array(spec_ns["centre"]), tiebreak_mode="lp")
    if st.get("tiebreak_ok") is False:
        m = Model(P, time_limit=1800)
        m.solve_saa(np.array(spec_ns["centre"]), tiebreak_mode="lp", tiebreak_presolve=False)
    pm, _ = polish(P, m.spine_values(), time_limit=1800)
    ns = _chrono_policy(P, pm)
    sm = SignalModel(P, core.NCAT, core.TAU, time_limit=1800)
    st = sm.solve_saa(spec_sg["law"], tiebreak=True, tiebreak_mode="lp")
    if not st["ok"] or st.get("tiebreak_ok") is False:
        sm = SignalModel(P, core.NCAT, core.TAU, time_limit=1800)
        sm.solve_saa(spec_sg["law"], tiebreak=True, tiebreak_mode="lp", tiebreak_presolve=False)
    sg = [_chrono_policy(P, polish(P, c.spine_values(), time_limit=1800)[0]) for c in sm.copies]
    return dict(cfg=cell["cfg"], level=cell["level"], r=cell["r"], p=cell["p"], lead_SB=P.lead["SB"],
                bk_start=P.bk_start, no_signal=ns, signal=sg)


def cmd_chrono():
    d = ROOT / "experiments" / "r2_runs" / "p1d_spine"
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    jobs = []
    for name in sorted({c["cfg"] for c in plan["cells"]}):
        for lv in core.LEVELS:
            cs = [c for c in plan["cells"] if c["cfg"] == name and c["level"] == lv]
            c = max(cs, key=lambda x: x["p"])
            jobs.append((c, plan["jobs"][c["jobs"]["nosig"]], plan["jobs"][c["jobs"]["sig"]]))
    with mp.get_context("spawn").Pool(min(8, len(jobs))) as pool:
        res = pool.map(_chrono_job, jobs)
    # every spine job: the independent shortfall check stored in the items
    items = [json.loads(f.read_text(encoding="utf-8")) for f in (d / "items").glob("J_*.json")
             if not f.name.endswith("timing.json")]
    worst = max(r.get("backup_shortfall", 0.0) for r in items)
    summary = dict(max_backup_shortfall_all_spine_jobs=worst, n_jobs=len(items))
    for r in res:
        pols = [r["no_signal"]] + r["signal"]
        r["min_residual"] = min(x["min_residual"] for pol in pols for x in pol)
        r["all_sb_ordered_before_T"] = all(x["sb_ordered_before_T"] for pol in pols for x in pol)
        r["all_bu_after_lead"] = all(x["bu_delivered_after_lead"] for pol in pols for x in pol)
    _dump("backup_chronology.json", dict(summary=summary, cells=res))


# ---- thresholds of the audit variants ------------------------------------------------------------------------
def cmd_thresholds():
    """theta_a = max($3M, 0.5 % of the variant's grid-mean no-signal cost), for accurate reporting only: Phase 1d did not
    retest any hypothesis."""
    p1c = json.loads((ROOT / "data" / "p1c_summary.json").read_text(encoding="utf-8"))["table"]
    out = {}
    for base in core.CONFIGS:
        J = p1c[f"{base}|sym2|0"]["mean_J_nosig"]
        out[f"{base} (Phase 1c)"] = dict(mean_J_nosig=J, theta=max(3.0, 0.005 * J))
    for v in ("stage", "spine", "refurb", "permit", "units"):
        f = OUT / f"{v}_summary.json"
        if not f.exists():
            continue
        T = json.loads(f.read_text(encoding="utf-8"))["table"]
        for name in sorted({k.rsplit("|", 2)[0] for k in T}):
            J = T[f"{name}|sym2|0"]["mean_J_nosig"]
            out[name] = dict(mean_J_nosig=J, theta=max(3.0, 0.005 * J))
    _dump("variant_thresholds.json", out)


# ---- equipment accounting of the rounded policies ------------------------------------------------------------
def _units_check_policy(P, pm, units):
    from experiments.p1d_core import round_repair, UNITS, BU_UNIT, RENT_UNIT
    m2, st, _ = round_repair(P, pm)
    val = m2.val
    bad = []
    mult = lambda x, u: abs(x / u - round(x / u)) < 1e-6                           # noqa: E731
    for j in P.assets:
        for k in range(P.Q + 1):
            if not mult(val(m2.v["x"][j][k]), UNITS[j]):
                bad.append(("order not whole units", j, k))
    for T in range(1, P.Q + 1):
        for j in P.assets:
            for k in range(T):
                y, x = val(m2.v["y"][T][j][k]), val(m2.v["x"][j][k])
                if y > x + 1e-6:
                    bad.append(("retained > delivered", T, j, k))
                if not (mult(y, UNITS[j]) and mult(x - y, UNITS[j])):
                    bad.append(("retained/disposed not whole units", T, j, k))
        for (_, _, var) in m2.bu_orders[T]:
            if not mult(val(var), BU_UNIT):
                bad.append(("standby not whole units", T))
    for q in range(1, P.Q + 1):
        for key in ("RST", "RNR"):
            if not mult(val(m2.v[key][q]), RENT_UNIT):
                bad.append(("rental not whole units", key, q))
    spine = {n: v for n, v in m2.spine_values().items()}
    return dict(status=st["status"], violations=bad[:20], n_violations=len(bad),
                shortfall=float(backup_shortfall(P, m2)),
                bookkeeping=float(np.max(np.abs(m2.C_values() - recompute_costs(P, m2))))), spine


def _units_check_job(args):
    from model.signal import SignalModel, node_names
    cell, spec_ns, spec_sg = args
    P = core.params(cell["cfg"])
    m = Model(P, time_limit=1800)
    m.solve_saa(np.array(spec_ns["centre"]), tiebreak_mode="lp")
    ns, _ = _units_check_policy(P, polish(P, m.spine_values(), time_limit=1800)[0], None)
    sm = SignalModel(P, core.NCAT, core.TAU, time_limit=1800)
    sm.solve_saa(spec_sg["law"], tiebreak=True, tiebreak_mode="lp")
    res, spines = [], []
    for c in sm.copies:
        r, sp = _units_check_policy(P, polish(P, c.spine_values(), time_limit=1800)[0], None)
        res.append(r)
        spines.append(sp)
    # common histories: every rounded decision at nodes k < TAU must be identical across the signal copies
    shared = [n for k in range(core.TAU) for n in node_names(P, k)]
    diff = max((abs(spines[y][n] - spines[0][n]) for y in range(1, len(spines)) for n in shared if n in spines[0]),
               default=0.0)
    return dict(cfg=cell["cfg"], level=cell["level"], r=cell["r"], no_signal=ns, signal=res,
                n_shared_decisions=len(shared), max_rounded_difference_at_shared_nodes=diff)


def cmd_unitscheck():
    plan, _ = _plan_items()
    jobs = []
    for cfg in core.CONFIGS:
        for lv in core.LEVELS:
            cs = [c for c in plan["cells"] if c["cfg"] == cfg and c["level"] == lv]
            c = max(cs, key=lambda x: x["p"])
            jobs.append((c, plan["jobs"][c["jobs"]["nosig"]], plan["jobs"][c["jobs"]["sig"]]))
    with mp.get_context("spawn").Pool(min(8, len(jobs))) as pool:
        res = pool.map(_units_check_job, jobs)
    total = sum(r["no_signal"]["n_violations"] + sum(x["n_violations"] for x in r["signal"]) for r in res)
    _dump("units_accounting_check.json", dict(total_violations=total,
                                              max_shared_difference=max(r["max_rounded_difference_at_shared_nodes"]
                                                                        for r in res), cells=res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["v0", "decomp", "d022a", "mc", "components", "repcells", "chrono", "thresholds",
                                    "unitscheck"])
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"]()
