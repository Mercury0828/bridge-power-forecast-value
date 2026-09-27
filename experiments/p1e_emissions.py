"""Phase 1e E3: operational CO2 screening of the benchmark policies (docs/phase1e_plan.md v2). A screening, not an
objective or an LCA.

    python experiments/p1e_emissions.py   -> data/p1e/emissions_summary.json

Quantities come from the E2 jobs (`experiments/r2_runs/p1e_e2`). They are undiscounted per-scenario physical totals of
every signal copy, for the continuous benchmark policy ("quantities") and its batched whole-unit implementation
("quantities_rounded").

For each cell (configuration, level, report outcome) and forecast family (Δ = 0):
- expected quantities come from the cell's cleaned joint law: the no-signal policy under the marginal; the signal
  policy under pre (copy 0) plus post (copy y);
- they are aggregated like E[Δ_info]: the equal-weight mean over levels of Σ_outcomes p × effect.

Factors (pinned):
- EPA GHG Emission Factors Hub (2025): natural gas 53.06 kg CO2/MMBtu; distillate fuel oil No. 2 73.96 kg CO2/MMBtu.
  Both are HHV; diesel is 5.772 MMBtu/bbl (EIA).
- EPA eGRID2023 summary tables (rev. 2, June 2025), subregion CO2 output emission rates:
  - ERCT (context A): total output 733.9 lb/MWh, non-baseload 1,242.7 lb/MWh;
  - SRVC (context B): total output 593.4 lb/MWh, non-baseload 1,286.8 lb/MWh.
- A generic sweep: 0.2 / 0.4 / 0.6 / 0.8 t/MWh.
- No T&D loss gross-up is applied to purchased MWh. Recovered heat gets no credit: it is internal to the CHP and ABS
  service.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from model.signal import joint_law, clean_law  # noqa: E402

OUT = ROOT / "data" / "p1e"
RUN = ROOT / "experiments" / "r2_runs" / "p1e_e2"
LB_T = 0.000453592
EF_GAS = 53.06e-3            # t CO2 per MMBtu
EF_DIESEL = 73.96e-3         # t CO2 per MMBtu
DIESEL_MMBTU_GAL = 5.772 / 42
GRID = {"A": {"eGRID2023 ERCT total": 733.9 * LB_T, "eGRID2023 ERCT non-baseload": 1242.7 * LB_T},
        "B": {"eGRID2023 SRVC total": 593.4 * LB_T, "eGRID2023 SRVC non-baseload": 1286.8 * LB_T}}
SWEEP = (0.2, 0.4, 0.6, 0.8)
FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")


def _expected(qty_copies, law):
    """Expected per-quantity totals of a policy (list of copies, each {key: [per T]}) under a joint law."""
    pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
    keys = qty_copies[0].keys()
    if len(qty_copies) == 1:
        m = pre + sum(post)
        return {k: float(m @ np.asarray(qty_copies[0][k])) for k in keys}
    return {k: float(pre @ np.asarray(qty_copies[0][k]) + sum(post[y] @ np.asarray(qty_copies[y][k])
                                                              for y in range(len(post)))) for k in keys}


def co2_t(q, P, ef_grid):
    gas = q["GE_MWh"] * P.ge_hr
    diesel = (q["DG_MWh"] + q["rental_MWh"]) * P.dg_gal_per_mwh * DIESEL_MMBTU_GAL
    return gas * EF_GAS + diesel * EF_DIESEL + q["grid_MWh"] * ef_grid, gas, diesel


def main():
    plan = json.loads((RUN / "plan.json").read_text(encoding="utf-8"))
    load = lambda k: json.loads((RUN / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    acc = {}
    for cell in plan["cells"]:
        base = cell["base"]
        ctx = core.base_ctx(base)
        P = core.params(base)
        ns, sg = load(cell["jobs"]["nosig"]), load(cell["jobs"]["sig"])
        if not (ns.get("ok") and sg.get("ok")):
            continue
        for fam in FAMS:
            pt = core.truth(base, cell["level"], 0)
            law = clean_law(joint_law(pt, core.true_kernel(tuple(cell["thresholds"]), fam, pt), core.TAU))
            for impl, key in (("continuous", "quantities"), ("batched", "quantities_rounded")):
                qn, qs = _expected(ns[key], law), _expected(sg[key], law)
                efs = dict(GRID[ctx], **{f"sweep {e}": e for e in SWEEP})
                for name, ef in efs.items():
                    cn, gn, dn = co2_t(qn, P, ef)
                    cs, gs, ds = co2_t(qs, P, ef)
                    k = (base, fam, impl, name)
                    a = acc.setdefault(k, {"low": [0.0] * 6, "high": [0.0] * 6})
                    for i, v in enumerate((cs - cn, cn, gs - gn, ds - dn, qs["grid_MWh"] - qn["grid_MWh"],
                                           qn["grid_MWh"])):
                        a[cell["level"]][i] += cell["p"] * v
    out = {}
    for (base, fam, impl, name), a in acc.items():
        m = [float(np.mean([a["low"][i], a["high"][i]])) for i in range(6)]
        out[f"{base}|{fam}|{impl}|{name}"] = dict(delta_co2_kt=m[0] / 1e3, nosig_co2_kt=m[1] / 1e3,
                                                   delta_rel=m[0] / m[1], delta_gas_MMBtu=m[2],
                                                   delta_diesel_MMBtu=m[3], delta_grid_MWh=m[4])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "emissions_summary.json").write_text(json.dumps(dict(
        factors=dict(gas_t_per_MMBtu=EF_GAS, diesel_t_per_MMBtu=EF_DIESEL, diesel_MMBtu_per_gal=DIESEL_MMBTU_GAL,
                     grid_t_per_MWh=GRID, sweep=SWEEP, loss_gross_up=False,
                     sources="EPA GHG Emission Factors Hub 2025; eGRID2023 summary tables rev2 (June 2025)"),
        table=out), indent=1), encoding="utf-8")
    for k, v in out.items():
        if "|sym2|" in k and ("total" in k or "non-baseload" in k):
            print(f"{k:60s} dCO2 {v['delta_co2_kt']:+8.2f} kt ({100 * v['delta_rel']:+.2f} %)")


if __name__ == "__main__":
    main()
