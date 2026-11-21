"""Decision map: descriptive re-tabulation of stored results.

Reads only committed frozen files: per-report rows (data/p1c_rows.json, data/p1d/<v>_rows.json), their summaries, and
the stored policies of the optimization runs (experiments/r2_runs/<run>/plan.json and items/J_<job>.json). No model is
built or solved.

Usage: .venv/Scripts/python.exe experiments/w3_decision_map.py
Writes data/w3/decision_map.json, data/w3/decision_map.md and data/w3/manifest.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "r2_runs"
OUT = ROOT / "data" / "w3"

LEVELS = ("low", "high")
OWNED_E = ("GE", "DG", "BESS")
UNIT = {"GE": 4.455, "DG": 2.1}
GAP = 1e-4  # policy solver relative gap; heuristic tolerance sensitivity only

# variant key -> (rows file, summary file, run dir or None, tau)
MAIN = {
    "base": ("data/p1c_rows.json", "data/p1c_summary.json", "p1c_main", 4),
    "tau2": ("data/p1d/tau2_rows.json", "data/p1d/tau2_summary.json", "p1f_tau2", 2),
    "tau6": ("data/p1d/tau6_rows.json", "data/p1d/tau6_summary.json", "p1f_tau6", 6),
    "spine": ("data/p1d/spine_rows.json", "data/p1d/spine_summary.json", "p1d_spine", 4),
    "e4b_spine": ("data/p1d/e4b_spine_rows.json", "data/p1d/e4b_spine_summary.json", None, 4),
    "e1": ("data/p1d/e1_rows.json", "data/p1d/e1_summary.json", "p1e_e1", 4),
    "gasrent_25": ("data/p1d/gasrent_25_rows.json", "data/p1d/gasrent_25_summary.json", "p1g_gasrent_25", 4),
    "gasrent_35": ("data/p1d/gasrent_35_rows.json", "data/p1d/gasrent_35_summary.json", "p1g_gasrent_35", 4),
    "gasrent_50": ("data/p1d/gasrent_50_rows.json", "data/p1d/gasrent_50_summary.json", "p1g_gasrent_50", 4),
    "naive": ("data/p1d/naive_rows.json", "data/p1d/naive_summary.json", "p1f_naive", 4),
}
OR_ONLY = ("units", "e2", "e4a", "e4a2", "e4b_rsb", "e1s", "stage", "carry", "blocks4", "nocchp", "permit", "refurb",
           "size075", "size300", "s_capex", "s_co2", "s_diesel", "s_disc", "s_eff", "s_gas", "s_grid", "s_life",
           "s_rent")


def _load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def _summary_table(rel):
    s = _load(rel)
    return s["table"] if "table" in s else s


def value_stats(rows):
    """Grid statistic of the main analysis per (full cfg key, family) at shift 0; M0 within-level normalized; N by level."""
    acc = defaultdict(lambda: {lv: dict(pd=0.0, pj=0.0, p=0.0, pneg=0.0, pneg_tol=0.0, n=0) for lv in LEVELS})
    for x in rows:
        if x.get("shift", 0) != 0:
            continue
        a = acc[(x["cfg"], x["family"])][x["level"]]
        p, d, j0 = x["p"], x["d_info"], x["J_nosig"]
        a["pd"] += p * d
        a["pj"] += p * j0
        a["p"] += p
        a["n"] += 1
        if d < 0:
            a["pneg"] += p
        if d < -GAP * j0:
            a["pneg_tol"] += p
    out = {}
    for (cfg, fam), by in acc.items():
        V = 0.5 * sum(by[lv]["pd"] for lv in LEVELS)
        M0 = 0.5 * sum(by[lv]["pj"] / by[lv]["p"] for lv in LEVELS)
        out[(cfg, fam)] = dict(
            V=V, M0=M0, rel=V / M0,
            by_level={lv: by[lv]["pd"] for lv in LEVELS},
            N={lv: by[lv]["pneg"] / by[lv]["p"] for lv in LEVELS},
            N_tol={lv: by[lv]["pneg_tol"] / by[lv]["p"] for lv in LEVELS},
            n={lv: by[lv]["n"] for lv in LEVELS},
            mass={lv: by[lv]["p"] for lv in LEVELS})
    return out


def _tail(arr, k):
    return float(sum(arr[k:]))


def policy_stats(run, tau):
    """Open capacity and technology switch per plan cell, from stored spine orders and planner laws (no solves)."""
    plan = json.loads((RUNS / run / "plan.json").read_text(encoding="utf-8"))
    cache = {}

    def item(job):
        if job not in cache:
            cache[job] = json.loads((RUNS / run / "items" / f"J_{job}.json").read_text(encoding="utf-8"))
        return cache[job]

    cells, excluded = {}, []
    for c in plan["cells"]:
        sig_job, ns_job = c["jobs"]["sig"], c["jobs"]["nosig"]
        post = plan["jobs"][sig_job]["law"]["post"]
        ny = len(post)
        surv_tau = sum(_tail(post[y], tau) for y in range(ny))  # Pr_pl(T > tau)
        wcat = [sum(post[y]) for y in range(ny)]
        wsum = sum(wcat)
        wcat = [w / wsum for w in wcat]
        det = item(sig_job)["detail"]
        pre_by_y, O_surv, O_cum = [], 0.0, 0.0
        ge_post, dg_post, abs_open = [], [], 0.0
        for y in range(ny):
            o = det[y]["orders"]
            pre_by_y.append(sum(sum(o[j][:tau]) for j in OWNED_E))
            # orders at node k >= tau are placed only if T > k: weight Pr_pl(T > k, Y = y) / Pr_pl(T > tau)
            wk = {k: _tail(post[y], k) / surv_tau for k in range(tau, len(o["GE"]))}
            O_surv += sum(o[j][k] * wk[k] for j in OWNED_E for k in wk)
            abs_open += sum(o["ABS"][k] * wk[k] for k in wk)
            O_cum += wcat[y] * sum(sum(o[j][tau:]) for j in OWNED_E)
            py = _tail(post[y], tau) / surv_tau  # Pr_pl(Y = y | T > tau)
            ge_post.append(sum(o["GE"][k] * wk[k] for k in wk) / py if py > 0 else 0.0)
            dg_post.append(sum(o["DG"][k] * wk[k] for k in wk) / py if py > 0 else 0.0)
        for j in det[0]["orders"]:  # every individual pre-estimate order equal across categories
            for k in range(tau):
                vals = [det[y]["orders"][j][k] for y in range(ny)]
                if max(vals) - min(vals) > 1e-6:
                    raise SystemExit(f"non-anticipativity check failed in {run} {c['cfg']} {c['level']} r={c['r']} "
                                     f"asset {j} node {k}")
        C_pre = pre_by_y[0]
        abs_pre = sum(det[0]["orders"]["ABS"][:tau])
        # no-forecast plan
        centre = plan["jobs"][ns_job]["centre"]
        ns = item(ns_job)["detail"]["orders"]
        s_tau = _tail(centre, tau)
        ns_pre = sum(sum(ns[j][:tau]) for j in OWNED_E)
        ns_post = sum(ns[j][k] * _tail(centre, k) / s_tau for j in OWNED_E for k in range(tau, len(ns["GE"]))) \
            if s_tau > 0 else 0.0
        S_ge, S_dg = max(ge_post) - min(ge_post), max(dg_post) - min(dg_post)
        key = (c["cfg"], c["level"], c["r"])
        if C_pre + O_surv <= 0:  # excluded from the O average only
            excluded.append(key)
        cells[key] = dict(
            p=c["p"], C_pre=C_pre, O_surv=O_surv,
            O=O_surv / (C_pre + O_surv) if C_pre + O_surv > 0 else None, O_cum=O_cum,
            ns_pre=ns_pre, ns_post=ns_post, ns_O=ns_post / (ns_pre + ns_post) if ns_pre + ns_post > 0 else None,
            S_ge=S_ge, S_dg=S_dg, sw_ge=float(S_ge >= UNIT["GE"]), sw_dg=float(S_dg >= UNIT["DG"]),
            ge_by_cat=ge_post, abs_pre=abs_pre, abs_open=abs_open, jobs=dict(sig=sig_job, nosig=ns_job))
    return cells, excluded


METRICS = ("C_pre", "O_surv", "O", "O_cum", "ns_pre", "ns_post", "ns_O", "S_ge", "S_dg", "sw_ge", "sw_dg",
           "abs_pre", "abs_open")


def aggregate_policy(cells):
    """Within-level normalized report weights, then the mean of the two levels (as M0)."""
    acc = defaultdict(lambda: {lv: defaultdict(float) for lv in LEVELS})
    for (cfg, lv, r), m in cells.items():
        a = acc[cfg][lv]
        for k in METRICS:
            if m[k] is None:  # undefined share: left out of that metric's average only
                continue
            a[k] += m["p"] * m[k]
            a["_p_" + k] += m["p"]
    out = {}
    for cfg, by in acc.items():
        out[cfg] = {k: 0.5 * sum(by[lv][k] / by[lv]["_p_" + k] for lv in LEVELS) for k in METRICS}
        out[cfg]["by_level"] = {lv: {k: by[lv][k] / by[lv]["_p_" + k] for k in ("O", "sw_ge", "S_ge")}
                                for lv in LEVELS}
    return out


def check_reproduction(vs, table):
    worst_v = worst_m = 0.0
    for (cfg, fam), s in vs.items():
        key = f"{cfg}|{fam}|0"
        if key not in table:
            raise SystemExit(f"summary key missing: {key}")
        dv = abs(s["V"] - table[key]["mean"])
        dm = abs(s["M0"] - table[key]["mean_J_nosig"])
        worst_v, worst_m = max(worst_v, dv), max(worst_m, dm)
        if dv > 1e-9:
            raise SystemExit(f"V reproduction failed for {key}: {s['V']} vs {table[key]['mean']}")
        if dm > 1e-6:
            raise SystemExit(f"M0 reproduction failed for {key}: {s['M0']} vs {table[key]['mean_J_nosig']}")
    return worst_v, worst_m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(main={}, online_resource={})
    manifest = {}
    lines = ["# Decision map (generated by experiments/w3_decision_map.py)", "",
             "V, levels in $M; rel = V / M0 (no-forecast cost); N = share of a level's report probability with "
             "d_info < 0; N_tol = heuristic tolerance version (d_info < -1e-4 J_nosig). Policy columns (forecast-aware "
             "plan, owned GE+DG+BESS in MW, planner law): C_pre = ordered before the estimate; O_surv = expected orders "
             "still to be placed given the campus waits at the estimate quarter; O = O_surv / (C_pre + O_surv); "
             "ns_O = same share for the no-forecast plan; S_GE = spread across categories of expected GE still to be "
             "ordered (MW); sw_GE = share of cells with S_GE >= one 4.455 MW unit; O_cum = secondary metric "
             "(category-weighted cumulative post-estimate spine orders). Averages: within-level normalized report "
             "weights, then the mean of the two levels.", ""]
    for name, (rows_f, summ_f, run, tau) in MAIN.items():
        rows = _load(rows_f)
        vs = value_stats(rows)
        wv, wm = check_reproduction(vs, _summary_table(summ_f))
        if run:
            cells, excl = policy_stats(run, tau)
            pol = aggregate_policy(cells)
        else:
            cells, excl, pol = {}, [], {}
        manifest[name] = dict(rows=rows_f, summary=summ_f, run=run, tau=tau,
                              keys=sorted({f"{c}|{f}" for (c, f) in vs}),
                              n_rows_shift0=sum(1 for x in rows if x.get("shift", 0) == 0),
                              n_policy_cells=len(cells), excluded_policy_cells=[list(k) for k in excl],
                              jobs={f"{k[0]}|{k[1]}|{k[2]}": v["jobs"] for k, v in cells.items()})
        result["main"][name] = dict(tau=tau, reproduction=dict(V=wv, M0=wm),
                                    value={f"{c}|{f}": v for (c, f), v in vs.items()},
                                    policy=pol,
                                    policy_cells={f"{k[0]}|{k[1]}|{k[2]}": {m: v[m] for m in METRICS + ("ge_by_cat",)}
                                                  for k, v in cells.items()})
        lines += [f"## {name} (tau = {tau}; reproduction max |dV| = {wv:.1e}, |dM0| = {wm:.1e}; "
                  f"policy cells {len(cells)}, excluded {len(excl)})", "",
                  "| cfg | family | V | rel % | low | high | N low | N high | N_tol low | N_tol high | n low/high |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (c, f), v in sorted(vs.items()):
            lines.append(f"| {c.replace('|', ' / ')} | {f} | {v['V']:.2f} | {100 * v['rel']:.2f} | {v['by_level']['low']:.2f} | "
                         f"{v['by_level']['high']:.2f} | {v['N']['low']:.3f} | {v['N']['high']:.3f} | "
                         f"{v['N_tol']['low']:.3f} | {v['N_tol']['high']:.3f} | {v['n']['low']}/{v['n']['high']} |")
        if pol:
            lines += ["", "| cfg | C_pre | O_surv | O | O low/high | ns_O | S_GE | sw_GE | sw_GE low/high | S_DG | sw_DG | "
                          "ABS pre | ABS open | O_cum |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c, m in sorted(pol.items()):
                bl = m["by_level"]
                lines.append(f"| {c.replace('|', ' / ')} | {m['C_pre']:.1f} | {m['O_surv']:.1f} | {m['O']:.3f} | "
                             f"{bl['low']['O']:.3f}/{bl['high']['O']:.3f} | {m['ns_O']:.3f} | {m['S_ge']:.1f} | "
                             f"{m['sw_ge']:.3f} | {bl['low']['sw_ge']:.3f}/{bl['high']['sw_ge']:.3f} | "
                             f"{m['S_dg']:.1f} | {m['sw_dg']:.3f} | {m['abs_pre']:.1f} | {m['abs_open']:.1f} | "
                             f"{m['O_cum']:.1f} |")
        else:
            lines += ["", "Mechanism columns: not available (no stored order detail for this variant)."]
        lines.append("")
    lines += ["# Online Resource: remaining frozen variants (value statistics only)", ""]
    for name in OR_ONLY:
        rows_f, summ_f = f"data/p1d/{name}_rows.json", f"data/p1d/{name}_summary.json"
        rows = _load(rows_f)
        vs = value_stats(rows)
        wv, wm = check_reproduction(vs, _summary_table(summ_f))
        manifest[name] = dict(rows=rows_f, summary=summ_f, keys=sorted({f"{c}|{f}" for (c, f) in vs}),
                              n_rows_shift0=sum(1 for x in rows if x.get("shift", 0) == 0))
        result["online_resource"][name] = dict(reproduction=dict(V=wv, M0=wm),
                                               value={f"{c}|{f}": v for (c, f), v in vs.items()})
        note = (" Three unusable jobs serve four cells; the rows file leaves out the report outcomes they serve (see"
                " `unusable` in data/p1d/blocks4_summary.json)." if name == "blocks4" else "")
        lines += [f"## {name} (reproduction max |dV| = {wv:.1e}, |dM0| = {wm:.1e})", note,
                  "| cfg | family | V | rel % | low | high | N low | N high |", "|---|---|---|---|---|---|---|---|"]
        for (c, f), v in sorted(vs.items()):
            lines.append(f"| {c.replace('|', ' / ')} | {f} | {v['V']:.2f} | {100 * v['rel']:.2f} | {v['by_level']['low']:.2f} | "
                         f"{v['by_level']['high']:.2f} | {v['N']['low']:.3f} | {v['N']['high']:.3f} |")
        lines.append("")
    (OUT / "decision_map.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "decision_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote data/w3/decision_map.md")


if __name__ == "__main__":
    main()
