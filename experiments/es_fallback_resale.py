"""Face-value fallback (variant naive_eps) and resale value (variant s_salv): the tables behind Sections 5.4 and 5.5 of
the paper and Section S7 of Online Resource 1. Re-tabulates the rows and summaries written by analyze_p1d.py and the
stored policies of the runs; no solve. Writes data/es_fallback_resale.json and data/es_fallback_resale.md.

Run after: harness.py --kind p1d --variant {naive_eps,s_salv} --run es_{variant}; p1d_repair.py; analyze_p1d.py
(see experiments/run_es_fallback_resale.sh), or unzip runs_es.zip into experiments/.

Rules: values and policy metrics use the same usable cells; a (configuration, family) without both dispersion levels
is reported as INCOMPLETE and gets no materiality verdict, as does any key with an omitted cell; repaired jobs and
omitted cells are listed; the planner-support audit lists waiting histories with zero planner probability and the
evaluation probability they carry.
"""
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import w3_decision_map as dmap  # noqa: E402
from experiments import analyze_p1d as ap  # noqa: E402

FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")
LEVELS = dmap.LEVELS


def run_status(run):
    """-> usable job set, unusable jobs, repaired jobs (id -> rule), omitted cells (cfg, level, r, p)."""
    plan, res = ap.load(run)
    usable = {k for k, spec in plan["jobs"].items() if ap.usable(spec, res.get(k))}
    unusable = sorted(set(plan["jobs"]) - usable)
    repaired = {k: r["repair"]["rule"] for k, r in res.items() if isinstance(r.get("repair"), dict)}
    omitted = [(c["cfg"], c["level"], c["r"], c["p"]) for c in plan["cells"]
               if c["jobs"]["sig"] not in usable or c["jobs"]["nosig"] not in usable]
    return plan, usable, unusable, repaired, omitted


def values(variant, omitted, plan):
    rows = dmap._load(f"data/p1d/{variant}_rows.json")
    summ = dmap._load(f"data/p1d/{variant}_summary.json")
    mass = defaultdict(float)
    for x in rows:
        if x.get("shift", 0) == 0:
            mass[(x["cfg"], x["family"], x["level"])] += x["p"]
    keys = {(c["cfg"], f) for c in plan["cells"] for f in FAMS}      # every planned key, also one with no usable row
    complete = {k for k in keys if all(mass[(k[0], k[1], lv)] > 0 for lv in LEVELS)}
    vs = dmap.value_stats([x for x in rows if (x["cfg"], x["family"]) in complete])
    dmap.check_reproduction(vs, summ["table"])
    hit = {c for (c, _, _, _) in omitted}
    out = {}
    for (cfg, fam) in sorted(keys):
        if (cfg, fam) not in complete:
            out[f"{cfg}|{fam}"] = dict(status="INCOMPLETE (a dispersion level has no usable cell)", material=None)
            continue
        v = vs[(cfg, fam)]
        s = summ["table"][f"{cfg}|{fam}|0"]
        thr = max(3.0, 0.005 * v["M0"])
        inc = cfg in hit
        out[f"{cfg}|{fam}"] = dict(V=v["V"], low=v["by_level"]["low"], high=v["by_level"]["high"], M0=v["M0"],
                                  lower=s["lower"], threshold=thr,
                                  status="INCOMPLETE (omitted cell)" if inc else "complete",
                                  material=None if inc else bool(s["lower"] >= thr))
    return out


def support_audit(run):
    """Waiting histories (category y, node k >= tau) with zero planner probability Pr_pl(T > k, Y = y) after cleaning,
    and the report-weighted evaluation probability Pr_Q(T > k, Y = y) they carry, per configuration, family and level.
    Each history is counted at its first node, so the masses are not double counted along a path."""
    plan = json.loads((dmap.RUNS / run / "plan.json").read_text(encoding="utf-8"))
    acc = defaultdict(float)
    count = defaultdict(int)
    for c in plan["cells"]:
        tau = ap._tau(c)
        post = plan["jobs"][c["jobs"]["sig"]]["law"]["post"]
        for fam in FAMS:
            ev_post = ap._law(c, fam, 0)["post"]
            for y in range(len(post)):
                if dmap._tail(post[y], tau) <= 0:          # category never announced under the planner law
                    continue
                first = next((k for k in range(tau, len(post[y])) if dmap._tail(post[y], k) <= 0
                              and dmap._tail(post[y], k - 1) > 0), None)
                if first is None:
                    continue
                m = dmap._tail(ev_post[y], first)
                acc[(c["cfg"], fam, c["level"])] += c["p"] * m
                count[(c["cfg"], fam, c["level"])] += 1
    return {f"{k[0]}|{k[1]}|{k[2]}": dict(eval_mass=acc[k], histories=count[k]) for k in sorted(acc)}


def safe_aggregate(cells, planned=()):
    """dmap.aggregate_policy over the configurations that have usable cells at both levels; the rest are INCOMPLETE."""
    have = defaultdict(set)
    for (cfg, lv, _r) in cells:
        have[cfg].add(lv)
    ok = {cfg for cfg, lvs in have.items() if set(LEVELS) <= lvs}
    agg = dmap.aggregate_policy({k: v for k, v in cells.items() if k[0] in ok}) if ok else {}
    return agg, sorted((set(have) | set(planned)) - ok)       # wholly absent planned configurations included


def base_values(rel):
    t = dmap._summary_table(rel)
    return {k[:-2]: dict(V=v["mean"], low=v["by_level"]["low"], high=v["by_level"]["high"])
            for k, v in t.items() if k.endswith("|0")}


def _md(x):
    """Escape the '|' of configuration keys inside a Markdown table cell."""
    return str(x).replace("|", "\\|")


def _fmt(d, key="V"):
    return f"{d[key]:.2f}" if key in d else d["status"]


def main():
    res = {}
    lines = ["# Face-value fallback and resale value (generated by experiments/es_fallback_resale.py)", "",
             "V in $M: report-weighted value of the estimate at shift 0, mean of the two dispersion levels "
             "(M0 and policy averages within-level normalized).", ""]
    base = base_values("data/p1c_summary.json")
    naive = base_values("data/p1d/naive_summary.json")

    # Face-value fallback
    planF, _, unus, rep, omit = run_status("es_naive_eps")
    F = values("naive_eps", omit, planF)
    sup_eps, sup_naive = support_audit("es_naive_eps"), support_audit("p1f_naive")
    res["naive_eps"] = dict(values=F, unusable=unus, repaired=rep, omitted_cells=omit, support=sup_eps,
                            support_face_value=sup_naive)
    lines += ["## F. Face-value planner with weight eps on the broad planner kernel", "",
              f"Solver: unusable jobs {len(unus)}, repaired jobs {len(rep)}, omitted cells {len(omit)}.", "",
              "| configuration | family | forecast-aware planner | face value (eps 0) | eps 0.01 | eps 0.05 | "
              "eps 0.01 low/high | eps 0.05 low/high |", "|---|---|---|---|---|---|---|---|"]
    for cfg in ("A", "A+staged"):
        for fam in FAMS:
            miss = dict(status="INCOMPLETE (no usable cell)")
            a, b = F.get(f"{cfg}|naive_eps01|{fam}", miss), F.get(f"{cfg}|naive_eps05|{fam}", miss)
            lh = (lambda d: f"{d['low']:.2f}/{d['high']:.2f}" if "low" in d else d["status"])
            lines.append(f"| {cfg} | {fam} | {base[f'{cfg}|{fam}']['V']:.2f} | {naive[f'{cfg}|naive|{fam}']['V']:.2f} | "
                         f"{_fmt(a)} | {_fmt(b)} | {lh(a)} | {lh(b)} |")
    lines += ["", "Planner-support audit: report-weighted evaluation probability of waiting histories that have zero "
              "planner probability (first node of each such history).", "",
              "| planner | configuration | family | level | histories | evaluation probability |", "|---|---|---|---|---|---|"]
    for name, sup in (("face value", sup_naive), ("eps", sup_eps)):
        for k, v in sup.items():
            cfg, fam, lv = k.rsplit("|", 2)
            if cfg.split("|")[0] in ("A", "A+staged") and fam in ("sym1", "sym2"):
                lines.append(f"| {name} | {_md(cfg)} | {fam} | {lv} | {v['histories']} | {v['eval_mass']:.3e} |")

    # Resale value
    planR, usable, unus, rep, omit = run_status("es_s_salv")
    R = values("s_salv", omit, planR)
    skip = frozenset(set(json.loads((dmap.RUNS / "es_s_salv" / "plan.json").read_text(encoding="utf-8"))["jobs"])
                     - usable)
    pol, excl = dmap.policy_stats("es_s_salv", 4, skip_jobs=skip)
    agg, pol_incomplete = safe_aggregate(pol, planned={c["cfg"] for c in planR["cells"]})
    base_pol = dmap.aggregate_policy(dmap.policy_stats("p1c_main", 4)[0])
    res["s_salv"] = dict(values=R, unusable=unus, repaired=rep, omitted_cells=omit, policy=agg,
                         policy_incomplete=pol_incomplete,
                         policy_excluded=[list(k) for k in excl], base_policy=base_pol)
    lines += ["", "## R. Resale fraction of new equipment x0.5 (lo) and x1.5 (hi)", "",
              f"Solver: unusable jobs {len(unus)}, repaired jobs {len(rep)}, omitted cells {len(omit)}; "
              f"policy cells excluded from O {len(excl)}.", "",
              "| configuration | family | base | lo | hi | lo after allowance / threshold | hi after allowance / threshold |",
              "|---|---|---|---|---|---|---|"]
    for cfg in ("A", "A+staged", "B", "B+staged"):
        for fam in FAMS:
            miss = dict(status="INCOMPLETE (no usable cell)")
            lo, hi = R.get(f"{cfg}|salv_lo|{fam}", miss), R.get(f"{cfg}|salv_hi|{fam}", miss)
            at = (lambda d: f"{d['lower']:.2f} / {d['threshold']:.2f} ({d['status']})" if "lower" in d else d["status"])
            lines.append(f"| {cfg} | {fam} | {base[f'{cfg}|{fam}']['V']:.2f} | {_fmt(lo)} | {_fmt(hi)} | {at(lo)} | "
                         f"{at(hi)} |")
    lines += ["", "Policy metrics of the forecast-aware plans (decision-map definitions, usable cells): C_pre, O, "
              "S_GE (MW), sw_GE.", "", "| configuration | C_pre | O | S_GE | sw_GE |", "|---|---|---|---|---|"]
    for cfg, m in sorted(list(base_pol.items()) + list(agg.items())):
        lines.append(f"| {_md(cfg)} | {m['C_pre']:.1f} | {m['O']:.3f} | {m['S_ge']:.1f} | {m['sw_ge']:.3f} |")
    for cfg in pol_incomplete:
        lines.append(f"| {_md(cfg)} | INCOMPLETE (a dispersion level has no usable cell) | | | |")
    if rep or omit:
        lines += ["", "Repaired jobs: " + ", ".join(f"{k} ({v})" for k, v in sorted(rep.items())),
                  "Omitted cells: " + ", ".join(f"{c} {lv} r={r} p={p:.3g}" for c, lv, r, p in omit)]
    (ROOT / "data" / "es_fallback_resale.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (ROOT / "data" / "es_fallback_resale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
