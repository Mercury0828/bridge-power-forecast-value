"""Phase-1f tables (docs/phase1f_plan.md): exploratory, no claims.

    python experiments/make_p1f_report.py   -> data/p1f_tables.md

**Inputs.**
- `data/p1d/<variant>_summary.json` for the Phase-1f variants (E-S s_*, nocchp, naive, tau2, tau6, size075, size300,
  blocks4), from `experiments/analyze_p1d.py`.
- The Phase-1c summary, and the Phase-1c run items (`experiments/r2_runs/p1c_main`) for the realized quantities.
- `data/p1e/e0_ext_summary.json` and `data/p1e/tpit_errors_ext.json` (E0-ext).

**Values.** E[Δ_info] (Δ = 0, $M): the equal-weight mean over the two dispersion levels, on the matched Phase-1c
cells. Each variant is compared with the Phase-1c value and its own threshold (0.5 % of its no-signal cost).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
D1, DE = ROOT / "data" / "p1d", ROOT / "data" / "p1e"
CONFIGS = ("A", "A+staged", "B", "B+staged")
ROWS = [("B", "sym1"), ("B", "sym2"), ("A+staged", "sym1"), ("A+staged", "sym2"), ("B+staged", "sym2"),
        ("A", "sym2"), ("A+staged", "uninf"), ("B", "uninf")]
E_S = [("s_disc", ("disc_lo", "disc_hi")), ("s_gas", ("gas_lo", "gas_hi")), ("s_diesel", ("diesel_lo", "diesel_hi")),
       ("s_grid", ("grid_lo", "grid_hi")), ("s_capex", ("capex_lo", "capex_hi")), ("s_life", ("life_lo", "life_hi")),
       ("s_rent", ("rent_lo", "rent_hi")), ("s_eff", ("eff",)), ("s_co2", ("co2_lo", "co2_hi"))]
SINGLE = [("nocchp", "nocchp", "E-C CCHP ablation (no absorption chillers)"),
          ("naive", "naive", "E-N naive planner (estimate category taken as exact)"),
          ("tau2", "tau2", "E-T estimate at node 2"), ("tau6", "tau6", "E-T estimate at node 6"),
          ("size075", "size075", "E-Z campus × 0.5 (75 MW-IT)"), ("size300", "size300", "E-Z campus × 2 (300 MW-IT)"),
          ("blocks4", "blocks4", "E-R four operating blocks per quarter")]


def _j(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd=2):
    return "—" if x is None else f"{x:+.{nd}f}"


def val(S, key):
    t = (S or {}).get("table", {}).get(key)
    return None if t is None else t["mean"]


def thr(S, key):
    t = (S or {}).get("table", {}).get(key)
    return None if t is None else 0.005 * t["mean_J_nosig"]


def h3(S, name, fam):
    a, b = val(S, f"A+staged|{name}|{fam}|0"), val(S, f"A|{name}|{fam}|0")
    return None if a is None or b is None else a - b


# selfcheck/expected.md, E-S: the predicted direction of E[Δ_info] as the parameter rises (B; A+staged) and the falsifier
# on B sym2. Directions: "+" rises, "-" falls, "?" uncertain; "(w)" weak. Falsifier kinds: ("rise", 15) = B sym2 rises by
# more than 15 % at the high level; ("fall", 15) = falls by more than 15 % at the high level; ("change", 25) = changes by
# more than 25 %; ("sign",) = changes sign at any level.
ES_EXPECT = {"s_disc": ("- (w)", "- (w)", ("rise", 15)), "s_gas": ("-", "-", ("rise", 15)),
             "s_diesel": ("+", "+", ("fall", 15)), "s_grid": ("?", "+ (w)", ("sign",)),
             "s_capex": ("- (w)", "?", ("sign",)), "s_life": ("+ (w)", "?", ("sign",)),
             "s_rent": ("+", "+", ("fall", 15)), "s_eff": ("+ (w)", "?", ("change", 25)),
             "s_co2": ("+ (w)", "?", ("sign",))}


def _trend(xs):
    """The observed direction along the ordered values (base inserted by the caller in parameter order)."""
    d = [b - a for a, b in zip(xs, xs[1:])]
    if all(x >= -0.005 for x in d):
        return "+"
    if all(x <= 0.005 for x in d):
        return "-"
    return "non-monotone"


def es_reconciliation(p1c):
    """Rows: parameter · predicted and observed B sym2 / A+staged sym2 along the parameter (low, main, high; for eff and
    co2 the main case is the lowest value) · the falsifier and whether it is met."""
    rows = []
    b0, a0 = val(p1c, "B|sym2|0"), val(p1c, "A+staged|sym2|0")
    for v, levels in E_S:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        bs = [val(S, f"B|{lv}|sym2|0") for lv in levels]
        as_ = [val(S, f"A+staged|{lv}|sym2|0") for lv in levels]
        if v in ("s_eff", "s_co2"):                       # the main case is the lowest tested value
            bx, ax_ = [b0] + bs, [a0] + as_
        else:
            bx, ax_ = [bs[0], b0, bs[1]], [as_[0], a0, as_[1]]
        pb, pa, fal = ES_EXPECT[v]
        hi = bs[-1]
        if fal[0] == "rise":
            met, rule = (hi - b0) / b0 * 100 > fal[1], f"B sym2 rises > {fal[1]} % at the high level"
        elif fal[0] == "fall":
            met, rule = (b0 - hi) / b0 * 100 > fal[1], f"B sym2 falls > {fal[1]} % at the high level"
        elif fal[0] == "change":
            met, rule = max(abs(x - b0) for x in bs) / b0 * 100 > fal[1], f"B sym2 changes > {fal[1]} %"
        else:
            met, rule = any(x * b0 < 0 for x in bs), "B sym2 changes sign"
        rows.append((v, pb, " → ".join(f"{x:.2f}" for x in bx), _trend(bx), pa,
                     " → ".join(f"{x:.2f}" for x in ax_), _trend(ax_), rule, "**met**" if met else "not met"))
    return rows


def es_globals(p1c):
    """The global E-S expectations: B sym1 and sym2 above $4.01M at every level; A+staged sym2 without a sign change
    (and within ±50 % of the main value, stated); the uninformative control never positive (mean over levels)."""
    out = []
    lows, a_rng, un_pos = [], [], []
    a0 = val(p1c, "A+staged|sym2|0")
    for v, levels in E_S:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        for lv in levels:
            for fam in ("sym1", "sym2"):
                lows.append((val(S, f"B|{lv}|{fam}|0"), f"{lv} {fam}"))
            a_rng.append((val(S, f"A+staged|{lv}|sym2|0"), lv))
            for c in CONFIGS:
                u = val(S, f"{c}|{lv}|uninf|0")
                if u is not None and u > 0:
                    un_pos.append(f"{c} {lv} {u:+.2f}")
    if lows:
        m = min(lows)
        out.append(f"- B sym1/sym2 minimum over the E-S levels: {m[0]:.2f} ({m[1]}); expected > 4.01: "
                   f"{'as expected' if m[0] > 4.01 else '**falsified**'}.")
    if a_rng:
        lo, hi = min(a_rng), max(a_rng)
        sign = any(x * a0 < 0 for x, _ in a_rng)
        within = all(abs(x - a0) <= 0.5 * abs(a0) for x, _ in a_rng)
        out.append(f"- A+staged sym2 range: {lo[0]:.2f} ({lo[1]}) to {hi[0]:.2f} ({hi[1]}), main {a0:.2f}; within ±50 %: "
                   f"{'yes' if within else 'no'}; sign change (the falsifier): {'**yes**' if sign else 'no'}.")
    out.append("- Uninformative control positive (the falsifier): " + (", ".join(un_pos) if un_pos else "none") + ".")
    # the registered decision reference: 0.5 % of each level's own expected no-signal cost
    below = []
    for v, levels in E_S:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        for lv in levels:
            for c in ("B", "A+staged"):
                for fam in ("sym1", "sym2"):
                    x, t = val(S, f"{c}|{lv}|{fam}|0"), thr(S, f"{c}|{lv}|{fam}|0")
                    if x is not None and t is not None and x < t:
                        below.append(f"{c} {fam} at {lv} ({x:.2f} < {t:.2f})")
    out.append("- Below the level's own 0.5 % reference (B and A+staged, sym1 and sym2): "
               + ("; ".join(below) if below else "none") + ".")
    return out


def p1c_realized():
    """Expected realized orders and rentals of the Phase-1c policies (sym2, Δ = 0), aggregated as in
    `analyze_p1d.summarize`; cached in data/p1f/p1c_realized.json."""
    cache = ROOT / "data" / "p1f" / "p1c_realized.json"
    if cache.exists():
        return _j(cache)
    import sys
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from experiments import analyze_p1d as an
    plan, res = an.load("p1c_main")
    rows = []
    for cell in plan["cells"]:
        cell = dict(cell, base=cell["cfg"])
        law = an._law(cell, "sym2", 0)
        ns, sg = res[cell["jobs"]["nosig"]], res[cell["jobs"]["sig"]]
        rows.append(dict(cfg=cell["cfg"], level=cell["level"], p=cell["p"],
                         realized_nosig=an._realized(ns["detail"], law, False, an._tau(cell)),
                         realized_sig=an._realized(sg["detail"], law, True, an._tau(cell))))
    out = {}
    for c in CONFIGS:
        by = {}
        for r in rows:
            if r["cfg"] == c:
                by.setdefault(r["level"], []).append(r)
        out[c] = {pol: {k: float(np.mean([sum(r["p"] * r[pol][k] for r in v) / sum(r["p"] for r in v)
                                          for v in by.values()])) for k in rows[0][pol]}
                  for pol in ("realized_nosig", "realized_sig")}
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    p1c = _j(ROOT / "data" / "p1c_summary.json")
    out = ["# Phase 1f — tables (generated by `experiments/make_p1f_report.py`; exploratory, no claims)", ""]
    base = {f"{c}|{fam}": val(p1c, f"{c}|{fam}|0") for c, fam in ROWS}
    base_h3 = {fam: val(p1c, f"A+staged|{fam}|0") - val(p1c, f"A|{fam}|0") for fam in ("sym1", "sym2")}
    # ---- runs
    out += ["## Runs", "", "| variant | jobs | usable | complete |", "|---|---|---|---|"]
    for v in [v for v, _ in E_S] + [v for v, _, _ in SINGLE]:
        S = _j(D1 / f"{v}_summary.json")
        if S:
            out.append(f"| {v} | {S['n_jobs']} | {S['n_usable']} | {S['complete']} |")
    out.append("")
    # ---- E-S
    out += ["## E-S. Techno-economic one-at-a-time sensitivity (E[Δ_info], $M, Δ = 0)", "",
            "| level | " + " | ".join(f"{c} {fam}" for c, fam in ROWS) + " | H3 sym1 | H3 sym2 |",
            "|---|" + "---|" * (len(ROWS) + 2),
            "| Phase 1c (main) | " + " | ".join(f(base[f"{c}|{fam}"]) for c, fam in ROWS)
            + f" | {f(base_h3['sym1'])} | {f(base_h3['sym2'])} |"]
    for v, levels in E_S:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        for lv in levels:
            cells = [f(val(S, f"{c}|{lv}|{fam}|0")) for c, fam in ROWS]
            out.append(f"| {lv} | " + " | ".join(cells) + f" | {f(h3(S, lv, 'sym1'))} | {f(h3(S, lv, 'sym2'))} |")
    out += ["", "Own thresholds (0.5 % of each level's no-signal cost; B sym2 / A+staged sym2):", ""]
    for v, levels in E_S:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        out.append("- " + "; ".join(f"{lv}: {f(thr(S, f'B|{lv}|sym2|0'))} / {f(thr(S, f'A+staged|{lv}|sym2|0'))}"
                                    for lv in levels))
    out.append("")
    rec = es_reconciliation(p1c)
    if rec:
        out += ["Expected against observed (`selfcheck/expected.md`, E-S; sym2, Δ = 0; values along the parameter: "
                "low → main → high, or main → tested for heat rates and carbon):", "",
                "| parameter | B predicted | B observed | B trend | A+staged predicted | A+staged observed | "
                "A+staged trend | falsifier (B sym2) | falsifier met? |", "|---|---|---|---|---|---|---|---|---|"]
        out += ["| " + " | ".join(r) + " |" for r in rec]
        out += ["", "Global expectations:", ""] + es_globals(p1c)
    out.append("")
    # ---- single-change variants
    out += ["## E-C, E-N, E-T, E-Z, E-R (E[Δ_info], $M, Δ = 0)", "",
            "| variant | " + " | ".join(f"{c} {fam}" for c, fam in ROWS) + " | H3 sym1 | H3 sym2 |",
            "|---|" + "---|" * (len(ROWS) + 2),
            "| Phase 1c (main) | " + " | ".join(f(base[f"{c}|{fam}"]) for c, fam in ROWS)
            + f" | {f(base_h3['sym1'])} | {f(base_h3['sym2'])} |"]
    for v, name, label in SINGLE:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        cells = [f(val(S, f"{c}|{name}|{fam}|0")) for c, fam in ROWS]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {f(h3(S, name, 'sym1'))} | {f(h3(S, name, 'sym2'))} |")
    for v, name, label in SINGLE:
        S = _j(D1 / f"{v}_summary.json")
        if S and not S["complete"]:
            short = {c: S["table"].get(f"{c}|{name}|sym2|0", {}).get("n") for c in CONFIGS}
            full = {c: (p1c or {}).get("table", {}).get(f"{c}|sym2|0", {}).get("n") for c in CONFIGS}
            out.append(f"\nIncomplete: {label}, {S['n_usable']} of {S['n_jobs']} jobs usable; cells used: "
                       + ", ".join(f"{c} {short[c]} of {full[c]}" for c in CONFIGS if short[c] != full[c])
                       + ". Its row sums over the usable cells only; the matched comparison is in its section.")
    out += ["", "Expected cost (sym2, Δ = 0; $M), no-signal / signal, and the change against Phase 1c:", "",
            "| variant | configuration | J no-signal | J signal | ΔJ no-signal | ΔJ signal | own threshold |",
            "|---|---|---|---|---|---|---|"]
    for v, name, label in SINGLE + [(v, lv, lv) for v, lvs in E_S for lv in lvs]:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        for c in CONFIGS:
            n, s_ = S["costs"].get(f"{c}|{name}|J_nosig"), S["costs"].get(f"{c}|{name}|J_sig")
            if n and s_:
                out.append(f"| {label} | {c} | {n['variant']:.2f} | {s_['variant']:.2f} | {n['change']:+.2f} | "
                           f"{s_['change']:+.2f} | {0.005 * n['variant']:.2f} |")
    out.append("")
    # ---- physical shift: expected realized orders and rentals (sym2, Δ = 0), against Phase 1c
    base_rz = p1c_realized()
    out += ["Expected realized orders (MW; ABS in MWth) and stationary bridge rentals (RST, pre-connection "
            "MW-quarters; not the Phase-1e rented standby, which is off in these variants), sym2, Δ = 0; "
            "no-signal / signal, variant (Phase 1c):", "",
            "| variant | configuration | GE | DG | BESS | ABS | stationary rental |", "|---|---|---|---|---|---|---|"]
    for v, name, label in SINGLE + [(v, lv, f"E-S {lv}") for v, lvs in E_S for lv in lvs]:
        S = _j(D1 / f"{v}_summary.json")
        if not S:
            continue
        for c in CONFIGS:
            rz, b = S.get("realized", {}).get(f"{c}|{name}"), base_rz.get(c)
            if not rz:
                continue
            cells = [" / ".join(f"{rz[p][k]:.1f} ({b[p][k]:.1f})" for p in ("realized_nosig", "realized_sig"))
                     for k in ("GE", "DG", "BESS", "ABS", "ST_MWq")]
            out.append(f"| {label} | {c} | " + " | ".join(cells) + " |")
    out.append("")
    # ---- the value under the planner's own law (data/p1f/planner_law.json; p1f_checks plannerlaw)
    pl = _j(ROOT / "data" / "p1f" / "planner_law.json")
    if pl:
        out += ["Value under the planner's own law ($M; J_nosig − J_sig with both costs under the planner's law) and "
                "the embedding check (✓: signal cost ≤ no-signal cost within the 1e-4 gap in every cell):", "",
                "| run · level | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
        for run, acc in pl["runs"].items():
            levels = sorted({k.split("|")[1] if "|" in k else "main" for k in acc})
            for lv in levels:
                cells = []
                for c in CONFIGS:
                    a = acc.get(c if lv == "main" else f"{c}|{lv}")
                    cells.append("—" if a is None else f"{a['value']:.2f} {'✓' if a['embedding_ok'] else '✗'}")
                out.append(f"| {run} · {lv} | " + " | ".join(cells) + " |")
        out.append("")
    # ---- derived contrasts
    nv = _j(D1 / "naive_summary.json")
    if nv:
        out += ["## E-N. The value of modelling forecast error (Phase 1c signal-aware − naive, $M; Δ = 0)", "",
                "| configuration | family | Δ_info registered (J_nosig − J_sig) | Δ_info naive (J_nosig − J_naive; "
                "< 0: naive costs more than no signal) | value of modelling error (J_naive − J_sig) |",
                "|---|---|---|---|---|"]
        for c in CONFIGS:
            for fam in ("sym1", "sym2", "sym4", "opt2", "uninf"):
                a, b = val(p1c, f"{c}|{fam}|0"), val(nv, f"{c}|naive|{fam}|0")
                if a is not None and b is not None:
                    out.append(f"| {c} | {fam} | {f(a)} | {f(b)} | {f(a - b)} |")
        out.append("")
    sz = {v: _j(D1 / f"{v}_summary.json") for v in ("size075", "size300")}
    if any(sz.values()):
        out += ["## E-Z. Size-normalized information value (sym2, Δ = 0)", "",
                "| configuration | size | peak MW-IT | E[Δ_info] ($M) | $M per 100 MW-IT | % of no-signal cost |",
                "|---|---|---|---|---|---|"]
        for c in CONFIGS:
            t = (p1c or {}).get("table", {}).get(f"{c}|sym2|0")
            if t:
                out.append(f"| {c} | × 1 | 150 | {f(t['mean'])} | {f(100 * t['mean'] / 150)} | "
                           f"{100 * t['mean'] / t['mean_J_nosig']:+.2f} |")
            for v, mw in (("size075", 75), ("size300", 300)):
                t = (sz[v] or {}).get("table", {}).get(f"{c}|{v}|sym2|0")
                if t:
                    out.append(f"| {c} | × {mw / 150:g} | {mw} | {f(t['mean'])} | {f(100 * t['mean'] / mw)} | "
                               f"{100 * t['mean'] / t['mean_J_nosig']:+.2f} |")
        out.append("")
        # selfcheck/expected.md E-Z: B sym2 at × 2 within [1.5, 2.5] × main and at × 0.5 within [0.3, 0.7] × main; the
        # ratio E[Δ_info] / J_nosig within ± 25 % of the main ratio. Reported for every configuration; the falsifier is
        # stated on B sym2.
        out += ["Expected against observed (`selfcheck/expected.md`, E-Z; sym2, Δ = 0):", "",
                "| configuration | size | value / main value | expected (B) | (value/J) / main (value/J) | expected |",
                "|---|---|---|---|---|---|"]
        for c in CONFIGS:
            t0 = (p1c or {}).get("table", {}).get(f"{c}|sym2|0")
            for v, mult, rng in (("size075", 0.5, (0.3, 0.7)), ("size300", 2.0, (1.5, 2.5))):
                t = (sz[v] or {}).get("table", {}).get(f"{c}|{v}|sym2|0")
                if not (t and t0):
                    continue
                r = t["mean"] / t0["mean"] if abs(t0["mean"]) > 1e-9 else float("nan")
                q = (t["mean"] / t["mean_J_nosig"]) / (t0["mean"] / t0["mean_J_nosig"]) if abs(t0["mean"]) > 1e-9 \
                    else float("nan")
                ok_r = rng[0] <= r <= rng[1]
                ok_q = 0.75 <= q <= 1.25
                out.append(f"| {c} | × {mult:g} | {r:.2f} | [{rng[0]}, {rng[1]}]"
                           f"{(' ' + ('as expected' if ok_r else '**falsified**')) if c == 'B' else ''} | {q:.2f} | "
                           f"[0.75, 1.25]{(' ' + ('as expected' if ok_q else '**falsified**')) if c == 'B' else ''} |")
        out.append("")
    if _j(D1 / "blocks4_summary.json"):
        import sys
        sys.path.insert(0, str(ROOT))
        from experiments import p1d_core as p1d
        out += ["## E-R. Peak-demand consequences of the four-block profile (static)", "",
                "| configuration | B_req 2-block (MW) | B_req 4-block | G_part 2-block (MW) | G_part 4-block |",
                "|---|---|---|---|---|"]
        for c in CONFIGS:
            p2 = p1d.params_for(c, {})
            p4 = p1d.params_for(c, p1d.VARIANTS["blocks4"][f"{c}|blocks4"][1])
            out.append(f"| {c} | {p2.B_req:.2f} | {p4.B_req:.2f} | {p2.G_part:.2f} | {p4.G_part:.2f} |")
        out.append("")
        # selfcheck/expected.md E-R: J_nosig rises and E[Δ_info] changes by less than 20 %; falsified if it changes by
        # more than 40 % in B or A+staged (sym1, sym2), or if J_nosig falls.
        S4 = _j(D1 / "blocks4_summary.json")
        out += ["Expected against observed (`selfcheck/expected.md`, E-R; Δ = 0; matched cells: both sides on the cells "
                "whose two jobs are usable in the variant):", "",
                "| configuration | family | cells | E[Δ_info] main → 4 blocks | change | ΔJ no-signal (sym2) | reading |",
                "|---|---|---|---|---|---|---|"]
        for c in ("B", "A+staged", "B+staged", "A"):
            dj = S4["costs"].get(f"{c}|blocks4|J_nosig", {}).get("change")
            for fam in ("sym1", "sym2"):
                vs = S4["vs_phase1c"].get(f"{c}|blocks4|{fam}|0")
                if vs is None:
                    continue
                a, b = vs["d_info_main"]["mean"], vs["d_info_main"]["mean"] + vs["change"]["mean"]
                n_all = (p1c or {}).get("table", {}).get(f"{c}|{fam}|0", {}).get("n")
                ch = 100 * (b - a) / abs(a) if abs(a) > 1e-9 else float("nan")
                if c in ("B", "A+staged"):
                    rd = ("**falsified**" if abs(ch) > 40 or (dj is not None and dj < 0) else
                          "as expected" if abs(ch) < 20 else "between 20 and 40 % (not as expected, not falsified)")
                else:
                    rd = "(no prediction)"
                out.append(f"| {c} | {fam} | {vs['change']['n']} of {n_all} | {a:.2f} → {b:.2f} | {ch:+.1f} % | "
                           f"{'—' if dj is None else f'{dj:+.2f}'} | {rd} |")
        out.append("")
    # ---- E0-ext
    ext, err = _j(DE / "e0_ext_summary.json"), _j(DE / "tpit_errors_ext.json")
    if err:
        pv = err["provenance"]
        out += ["## E0-ext. TPIT with the pre-2015 workbooks (pre-declared sensitivity; the primary results stay of record)",
                "", f"{pv['ext_snapshots']} pre-2015 snapshots ({pv['ext_dates'][0]} … {pv['ext_dates'][-1]}) pooled "
                f"with {pv['primary_snapshots']} primary snapshots; failed to parse: {pv['ext_failed_to_parse'] or 'none'}.",
                "", "| horizon · subset | n | not observed | mean | SD | late | on time | early |",
                "|---|---|---|---|---|---|---|---|"]
        for k, v in err["summary"].items():
            out.append(f"| {k.replace('|', ' · ')} | {v['n']} | {v['n_not_observed']} | {v['mean']:+.2f} | "
                       f"{v['sd']:.2f} | {v['share_late']:.2f} | {v['share_on_time']:.2f} | {v['share_early']:.2f} |")
        out += ["", "Era split (completion before 2018 / 2018 on):", "",
                "| horizon · subset | n (<2018) | mean (<2018) | n (≥2018) | mean (≥2018) |", "|---|---|---|---|---|"]
        a, b = err["era"]["completed_before_2018"], err["era"]["completed_2018_on"]
        for k in err["summary"]:
            x, y = a.get(k), b.get(k)
            out.append(f"| {k.replace('|', ' · ')} | {x['n'] if x else 0} | {f(x['mean']) if x else '—'} | "
                       f"{y['n'] if y else 0} | {f(y['mean']) if y else '—'} |")
        out.append("")
    if ext:
        out += ["E[Δ_info] of the six policy sets under the extended families (Δ = 0):", "",
                "| run · configuration | ext h4 | ext h8 | ext h12 | ext h8 load | ext h8 uninf |", "|---|---|---|---|---|---|"]
        keys = sorted({"|".join(k.split("|")[:-2]) for k in ext["table"]})
        for rk in keys:
            cells = [f(ext["table"].get(f"{rk}|{fm}|0", {}).get("mean")) for fm in
                     ("tpit_ext_h4", "tpit_ext_h8", "tpit_ext_h12", "tpit_ext_h8_load", "tpit_ext_h8-uninf")]
            out.append(f"| {rk.replace('|', ' · ')} | " + " | ".join(cells) + " |")
        out.append("")
    (ROOT / "data" / "p1f_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("written data/p1f_tables.md")


if __name__ == "__main__":
    main()
