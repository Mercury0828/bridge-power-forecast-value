"""Phase-1g report (docs/phase1g_plan.md; D-030): G1 rental carry-over at staged connection and G2 limited gas rentals,
against the frozen Phase-1c benchmark. Exploratory; nothing here is a registered claim.

    python experiments/make_p1g_report.py  ->  data/p1g_result_tables.md, data/p1g/summary.json

Every number is read from data/p1d/<variant>_summary.json (experiments/analyze_p1d.py), data/p1c_summary.json and the
run items (policy detail of the representative cells). Values are E[Delta_info] in $M at Delta = 0: the equal-weight
mean over the two dispersion levels of the report-weighted value.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")
CONFIGS = ("A", "A+staged", "B", "B+staged")
PRICES = (25, 35, 50)
_CLAIMS = json.loads((ROOT / "data" / "p1c_summary.json").read_text(encoding="utf-8"))["claims"]
REG_THRESHOLD = {"B": _CLAIMS["H1a"]["threshold"], "A+staged": _CLAIMS["H2a"]["threshold"]}   # full precision
REP = {("A", "low"): 7, ("A", "high"): 12, ("A+staged", "low"): 7, ("A+staged", "high"): 12,
       ("B", "low"): 2, ("B", "high"): 4, ("B+staged", "low"): 2, ("B+staged", "high"): 4}   # representative cells


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def _f(x, sign=True):
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "−")


def _cmp(S, key):
    """Matched-cell comparison with Phase 1c (experiments/analyze_p1d.py): the benchmark on the variant's usable cells,
    the variant's value and the change; n = matched cells."""
    v, t = S["vs_phase1c"][key], S["table"][key]
    return dict(benchmark=v["d_info_main"]["mean"], value=t["mean"], change=v["change"]["mean"], n=v["change"]["n"],
                low=t["by_level"]["low"], high=t["by_level"]["high"], own_ref=0.005 * t["mean_J_nosig"])


def _usability(S):
    return dict(complete=S["complete"], n_jobs=S["n_jobs"], n_usable=S["n_usable"], unusable=S["unusable"])


def _interaction(rows_v, name, fam, unstaged=None):
    """Staging interaction on matched cells, cell by cell: the variant's A+staged value minus the unstaged A value, and
    the Phase-1c interaction on the same cells (grid statistic, Delta = 0). `unstaged` names the variant's own A rows
    (G2: gas rentals in both configurations); None takes the Phase-1c A value (G1: carry-over changes staged service
    only, so unstaged A is the Phase-1c A)."""
    from experiments import p1c_core as core
    from experiments.analyze_p1c import _stat
    p1c = _j("data/p1c_rows.json")

    def ser(rows, cfg):
        return {(r["level"], r["r"]): (r["p"], r["d_info"], r["J_nosig"]) for r in rows
                if r["cfg"] == cfg and r["family"] == fam and r["shift"] == 0}
    v, s = ser(rows_v, name), ser(p1c, "A+staged")
    a = ser(p1c, "A") if unstaged is None else ser(rows_v, unstaged)
    a0 = ser(p1c, "A")
    common = sorted(set(v) & set(a) & set(s) & set(a0))
    ctx = core.base_ctx("A")
    var = _stat({k: (v[k][0], v[k][1] - a[k][1], v[k][2]) for k in common}, ctx)
    ben = _stat({k: (s[k][0], s[k][1] - a0[k][1], s[k][2]) for k in common}, ctx)
    return dict(variant=var["mean"], benchmark=ben["mean"], n=len(common), by_level=var["by_level"],
                registered_threshold=REG_THRESHOLD["A+staged"])


def g1():
    S = _j("data/p1d/carry_summary.json")
    rows_v = _j("data/p1d/carry_rows.json")
    rows, inter = [], {}
    for cfg in ("A+staged", "B+staged"):
        for fam in FAMS:
            c = _cmp(S, f"{cfg}|carry|{fam}|0")
            rows.append(dict(cfg=cfg, fam=fam, benchmark=c["benchmark"], carry=c["value"], change=c["change"],
                             n=c["n"], low=c["low"], high=c["high"], own_ref=c["own_ref"],
                             j_nosig_change=S["costs"][f"{cfg}|carry|J_nosig"]["change"]))
    for fam in ("sym1", "sym2"):
        inter[fam] = _interaction(rows_v, "A+staged|carry", fam)
        inter[fam]["own_ref"] = 0.005 * S["table"][f"A+staged|carry|{fam}|0"]["mean_J_nosig"]
    return rows, inter, _usability(S)


def g2():
    rows, reading, usab = [], {}, {}
    for p in PRICES:
        S = _j(f"data/p1d/gasrent_{p}_summary.json")
        usab[p] = _usability(S)
        for cfg in CONFIGS:
            for fam in FAMS:
                c = _cmp(S, f"{cfg}|gasrent_{p}|{fam}|0")
                rows.append(dict(price=p, cfg=cfg, fam=fam, benchmark=c["benchmark"], value=c["value"],
                                 change=c["change"], n=c["n"], own_ref=c["own_ref"],
                                 j_nosig_change=S["costs"][f"{cfg}|gasrent_{p}|J_nosig"]["change"],
                                 low=c["low"], high=c["high"]))
    for cfg in ("B", "A+staged"):
        for fam in ("sym1", "sym2"):
            above = {r["price"]: r["value"] >= r["own_ref"] for r in rows if r["cfg"] == cfg and r["fam"] == fam}
            reading[f"{cfg}|{fam}"] = above
    return rows, reading, usab


def _cell_items(run_name, cfg, lv, r):
    run = ROOT / "experiments" / "r2_runs" / run_name
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    cell = next(c for c in plan["cells"] if c.get("base", c["cfg"]) == cfg and c["level"] == lv and c["r"] == r)
    load = lambda k: json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8"))["detail"]  # noqa: E731
    return cell, load(cell["jobs"]["nosig"]), load(cell["jobs"]["sig"])


def _expect(cell, cfg, lv, per_T):
    """E[per_T(T)] under the cell's true law of T (Delta = 0) for the no-forecast plan, and under the sigma = 2 joint law
    for the forecast-aware plan. per_T(detail, T) is a scenario quantity of one plan copy; T = 1..Q+1."""
    import numpy as np
    from experiments import p1c_core as core
    from model.signal import joint_law, clean_law, evaluate
    pT = core.truth(cfg, lv, 0)
    law = clean_law(joint_law(pT, core.true_kernel(tuple(cell["thresholds"]), "sym2", pT), core.TAU))
    n = len(pT)

    def vec(detail):
        return np.array([per_T(detail, i + 1) for i in range(n)])

    def ns_value(ns):
        return float(np.dot(np.asarray(pT, dtype=float), vec(ns)))

    def sg_value(sg):
        return evaluate([vec(sg[y]) for y in range(len(sg))], law, core.TAU)
    return ns_value, sg_value


def _gr_before(detail, T):                     # gas rentals contracted for bridge quarters 1..T-1 (1..Q if T = Q + 1)
    r = detail.get("RGR", [])
    return float(sum(r[:T - 1]))


def _ge_before(detail, T):                     # gas CHP ordered at nodes 0..T-1 (0..Q if T = Q + 1)
    return float(sum(detail["orders"]["GE"][:T]))


def _staged_rental_T(detail, T):               # staged rental in quarter T of scenario T (0 without connection)
    b = detail["branches"].get(str(T), detail["branches"].get(T)) if T <= len(detail["orders"]["GE"]) - 1 else None
    return float(b["staged_rental"][0]) if b and b.get("staged_rental") else 0.0


def rep_cells():
    """Expected gas-rental capacity for bridge quarters (MW-quarters) and gas CHP ordered before connection (MW), no
    forecast and forecast-aware (sigma = 2 joint law), in the eight representative cells: the benchmark (Phase 1c) and
    each price level."""
    out = []
    for label, run_name in [("benchmark", "p1c_main")] + [(p, f"p1g_gasrent_{p}") for p in PRICES]:
        for (cfg, lv), r in REP.items():
            cell, ns, sg = _cell_items(run_name, cfg, lv, r)
            row = dict(price=label, cfg=cfg, level=lv, r=r)
            for name, fn in (("GR", _gr_before), ("GE", _ge_before)):
                nsv, sgv = _expect(cell, cfg, lv, fn)
                row[f"nosig_{name}"], row[f"sig_{name}"] = nsv(ns), sgv(sg)
            out.append(row)
    return out


def components(run_name, cfg, fam="sym2"):
    """Where the money moves: the value of the forecast split into additive cost components (no forecast minus
    forecast-aware; $M; positive: the forecast-aware plan spends less), over all cells of one configuration, with the
    Phase-1c weighting (equal-weight mean over the two dispersion levels of the report-weighted sum; Delta = 0). The
    method of experiments/p1d_accounting.py (components), applied to any run."""
    import numpy as np
    from experiments import p1c_core as core
    from model.bridge import COMPONENTS
    from model.signal import joint_law, clean_law
    run = ROOT / "experiments" / "r2_runs" / run_name
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    load = lambda k: json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    acc = {lv: {c: 0.0 for c in COMPONENTS} for lv in core.LEVELS}
    for cell in [c for c in plan["cells"] if c.get("base", c["cfg"]) == cfg]:
        ns, sg = load(cell["jobs"]["nosig"]), load(cell["jobs"]["sig"])
        pt = core.truth(cfg, cell["level"], 0)
        law = clean_law(joint_law(pt, core.true_kernel(tuple(cell["thresholds"]), fam, pt), core.TAU))
        pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
        for c in COMPONENTS:
            cn = np.asarray(ns["components"][c])
            cs = [np.asarray(sg["components"][y][c]) for y in range(core.NCAT)]
            acc[cell["level"]][c] += cell["p"] * (pre @ (cn - cs[0]) + sum(post[y] @ (cn - cs[y])
                                                                            for y in range(core.NCAT)))
    comp = {c: float(np.mean([acc[lv][c] for lv in core.LEVELS])) for c in COMPONENTS}
    return dict(components=comp, total=sum(comp.values()))


def money(configs=CONFIGS):
    out = {}
    for label, run_name in [("benchmark", "p1c_main")] + [(p, f"p1g_gasrent_{p}") for p in PRICES]:
        for cfg in configs:
            out[f"{label}|{cfg}"] = components(run_name, cfg)
    return out


def g2_interaction():
    return {p: {fam: _interaction(_j(f"data/p1d/gasrent_{p}_rows.json"), f"A+staged|gasrent_{p}", fam,
                                  unstaged=f"A|gasrent_{p}")
                for fam in ("sym1", "sym2")} for p in PRICES}


GROUPS = [("capital", ("capex",)), ("rentals", ("rental",)), ("bridge fuel and O&M", ("bridge_fuel_om",)),
          ("grid", ("grid",)), ("post-connection CHP", ("post_chp",)), ("standby", ("backup",)),
          ("resale and end values", ("retirement_value", "terminal_value")),
          ("filings and fixed O&M", ("filing", "fixed_om", "delay"))]


def expectations(r1, inter, r2, reps):
    """The pre-declared expectations of docs/phase1g_plan.md v1.1, checked."""
    def rel(r):
        return (r["change"] / abs(r["benchmark"])) if abs(r["benchmark"]) >= 0.5 else None
    out = []
    for cfg, lim in (("B+staged", 1.0), ("A+staged", 2.0)):
        for fam in ("sym1", "sym2"):
            r = next(x for x in r1 if x["cfg"] == cfg and x["fam"] == fam)
            out.append((f"G1: absolute change below ${lim:.0f}M, {cfg} {fam}", f"{_f(r['change'])}", abs(r["change"]) < lim))
            rr = rel(r)
            out.append((f"G1: material finding (relative change > 40 %), {cfg} {fam}",
                        _f(100 * rr) + " %" if rr is not None else "n/a", rr is not None and abs(rr) > 0.40))
    out.append(("G1: staging interaction at sym1 stays positive", _f(inter["sym1"]["variant"]),
                inter["sym1"]["variant"] > 0))
    for fam in ("sym1", "sym2"):
        r25 = next(x for x in r2 if x["price"] == 25 and x["cfg"] == "B" and x["fam"] == fam)
        r50 = next(x for x in r2 if x["price"] == 50 and x["cfg"] == "B" and x["fam"] == fam)
        out.append((f"G2: B {fam} at $25 falls by 10-50 %", _f(100 * rel(r25)) + " %", -0.50 <= rel(r25) <= -0.10))
        out.append((f"G2: B {fam} at $50 within ±10 %", _f(100 * rel(r50)) + " %", abs(rel(r50)) <= 0.10))
        for pr in PRICES:
            r = next(x for x in r2 if x["price"] == pr and x["cfg"] == "B" and x["fam"] == fam)
            out.append((f"G2: B {fam} positive at ${pr}", _f(r["value"]), r["value"] > 0))
    for lv in ("low", "high"):
        b = next(x for x in reps if x["price"] == "benchmark" and x["cfg"] == "B" and x["level"] == lv)
        g = next(x for x in reps if x["price"] == 25 and x["cfg"] == "B" and x["level"] == lv)
        out.append((f"G2: B {lv} at $25, no-forecast plan contracts gas rentals and orders less gas CHP before "
                    "connection", f"GR {g['nosig_GR']:.1f} MW-q; GE {g['nosig_GE']:.1f} vs {b['nosig_GE']:.1f} MW",
                    g["nosig_GR"] > 0 and g["nosig_GE"] < b["nosig_GE"]))
    return out


def carry_cells():
    """Expected quarter-T staged rental (MW) in the four representative staged cells: benchmark branch rental against
    the carried spine contract."""
    out = []
    for cfg in ("A+staged", "B+staged"):
        for lv in ("low", "high"):
            r = REP[cfg, lv]
            row = dict(cfg=cfg, level=lv, r=r)
            for label, run_name in (("benchmark", "p1c_main"), ("carry", "p1g_carry")):
                cell, ns, sg = _cell_items(run_name, cfg, lv, r)
                nsv, sgv = _expect(cell, cfg, lv, _staged_rental_T)
                row[f"{label}_nosig"], row[f"{label}_sig"] = nsv(ns), sgv(sg)
            out.append(row)
    return out


def main():
    r1, inter, usab1 = g1()
    r2, reading, usab2 = g2()
    reps = rep_cells()
    carry = carry_cells()
    inter2 = g2_interaction()
    mon = money()
    checks = expectations(r1, inter, r2, reps)
    out = ROOT / "data" / "p1g"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(g1=r1, g1_interaction=inter, g1_carry_cells=carry, g1_usability=usab1, g2=r2,
                                                      g2_reading=reading, g2_usability=usab2,
                                                      g2_interaction=inter2, g2_money=mon,
                                                      expectations=[dict(item=a, result=b, met=c)
                                                                    for a, b, c in checks],
                                                      representative=reps), indent=1),
                                      encoding="utf-8")
    md = ["# Phase 1g tables (exploratory; generated by experiments/make_p1g_report.py)", "",
          "## G1. Rental carry-over at staged connection (Δ = 0; $M)", "",
          f"Jobs usable: {usab1['n_usable']} of {usab1['n_jobs']}; comparisons on matched cells (n). ΔJ no-forecast: "
          "change in the grid-mean no-forecast cost (σ = 2 law, matched cells).", "",
          "| configuration | family | benchmark | carry-over | change | n | low | high | own reference | "
          "ΔJ no-forecast |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in r1:
        md.append(f"| {r['cfg']} | {r['fam']} | {_f(r['benchmark'])} | {_f(r['carry'])} | {_f(r['change'])} | "
                  f"{r['n']} | {_f(r['low'])} | {_f(r['high'])} | {_f(r['own_ref'], False)} | "
                  f"{_f(r['j_nosig_change'])} |")
    md += ["", "**Staging interaction under carry-over** (A+staged, carry − A, benchmark; registered threshold "
           f"${REG_THRESHOLD['A+staged']:.2f}M)", "",
           "| family | benchmark (matched cells) | carry-over | n | low | high | own reference |",
           "|---|---|---|---|---|---|---|"]
    for fam, d in inter.items():
        md.append(f"| {fam} | {_f(d['benchmark'])} | {_f(d['variant'])} | {d['n']} | {_f(d['by_level']['low'])} | "
                  f"{_f(d['by_level']['high'])} | {_f(d['own_ref'], False)} |")
    md += ["", "**Expected quarter-T staged rental** (MW; true law of T at Δ = 0; forecast-aware under the σ = 2 joint "
           "law): the benchmark's branch rental against the carried spine contract", "",
           "| cell | benchmark, no forecast | benchmark, forecast-aware | carry, no forecast | carry, forecast-aware |",
           "|---|---|---|---|---|"]
    for r in carry:
        md.append(f"| {r['cfg']} {r['level']} (r = {r['r']}) | {r['benchmark_nosig']:.2f} | {r['benchmark_sig']:.2f} | "
                  f"{r['carry_nosig']:.2f} | {r['carry_sig']:.2f} |")
    md += ["", "## G2. Limited gas rentals (50 MW inside the 100 MW rental cap; Δ = 0; $M)", "",
           "Jobs usable: " + "; ".join(f"${p}: {u['n_usable']} of {u['n_jobs']}" for p, u in usab2.items())
           + "; comparisons on matched cells (n).", "",
           "| price $/kW-month | configuration | family | benchmark | with gas rentals | change | n | low | high | "
           "own reference | ΔJ no-forecast |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in r2:
        md.append(f"| {r['price']} | {r['cfg']} | {r['fam']} | {_f(r['benchmark'])} | {_f(r['value'])} | "
                  f"{_f(r['change'])} | {r['n']} | {_f(r['low'])} | {_f(r['high'])} | {_f(r['own_ref'], False)} | "
                  f"{_f(r['j_nosig_change'])} |")
    md += ["", "**Pointwise materiality comparisons** (grid statistic at or above its own reference, 0.5 % of the variant's "
           "no-forecast cost; each price separately, no interpolation)", "",
           "| contrast | " + " | ".join(f"${p}" for p in PRICES) + " |", "|---|" + "---|" * len(PRICES)]
    for k, d in reading.items():
        md.append(f"| {k.replace('|', ', ')} | " + " | ".join("above" if d[p] else "below" for p in PRICES) + " |")
    md += ["", "## G2. Compact table (σ = 1 and σ = 2; Δ = 0; $M)", "",
           "| configuration | family | benchmark | $25 | $35 | $50 |", "|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        for fam in ("sym1", "sym2", "uninf"):
            vals = {r["price"]: r for r in r2 if r["cfg"] == cfg and r["fam"] == fam}
            md.append(f"| {cfg} | {fam} | {_f(vals[25]['benchmark'])} | "
                      + " | ".join(_f(vals[p]["value"]) for p in PRICES) + " |")
    md += ["", "**Staging interaction under gas rentals** (A+staged − A, both with gas rentals; matched cells)", "",
           "| family | benchmark | " + " | ".join(f"${p}" for p in PRICES) + " |", "|---|---|" + "---|" * len(PRICES)]
    for fam in ("sym1", "sym2"):
        md.append(f"| {fam} | {_f(inter2[PRICES[0]][fam]['benchmark'])} | "
                  + " | ".join(_f(inter2[p][fam]["variant"]) for p in PRICES) + " |")
    md += ["", "## G2. Where the money moves (σ = 2, Δ = 0; $M; positive: the forecast-aware plan spends less)", "",
           "| configuration | rent | " + " | ".join(g for g, _ in GROUPS) + " | total |",
           "|---|---|" + "---|" * len(GROUPS) + "---|"]
    for cfg in ("A+staged", "B", "B+staged"):
        for label in ["benchmark"] + list(PRICES):
            v = mon[f"{label}|{cfg}"]
            md.append(f"| {cfg} | {label if label == 'benchmark' else '$' + str(label)} | "
                      + " | ".join(_f(sum(v["components"][c] for c in cs)) for _, cs in GROUPS)
                      + f" | {_f(v['total'])} |")
    md += ["", "## Pre-declared expectations (docs/phase1g_plan.md v1.1)", "", "| expectation | result | met |",
           "|---|---|---|"]
    for a, b, c in checks:
        md.append(f"| {a} | {b} | {'yes' if c else 'no'} |")
    md += ["", "## G2. Representative cells: expected gas rentals for bridge quarters E[Σ_{q<T} R^GR_q] (MW-quarters) "
           "and gas CHP ordered before connection E[Σ_{k<T} x_GE,k] (MW)", "",
           "True law of T at Δ = 0; forecast-aware plan under the σ = 2 joint law.", "",
           "| price | cell | no forecast: GR / GE | forecast-aware: GR / GE |", "|---|---|---|---|"]
    for r in reps:
        md.append(f"| {r['price']} | {r['cfg']} {r['level']} (r = {r['r']}) | {r['nosig_GR']:.1f} / {r['nosig_GE']:.1f} | "
                  f"{r['sig_GR']:.1f} / {r['sig_GE']:.1f} |")
    (ROOT / "data" / "p1g_result_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written data/p1g_result_tables.md and data/p1g/summary.json")


if __name__ == "__main__":
    main()
