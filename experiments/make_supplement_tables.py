"""Four tables of Online Resource 1 computed from the released JSON files: S5 (value by cost component), S6 (orders in
the representative cells), S7 (fuel and grid use in the representative cells) and S34 (certificates).

    python experiments/make_supplement_tables.py    -> data/supplement_tables.md

No solve. The other result tables of the supplement come from the generators named in README.md.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINUS = chr(0x2212)


def load(rel):
    return json.loads((ROOT / "data" / rel).read_text(encoding="utf-8"))


def signed(x, places):
    return f"{x:+.{places}f}".replace("-", MINUS)


def plain(x, places=1):
    return "0" if abs(x) < 0.5 * 10 ** -places else f"{x:.{places}f}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def money():
    d = load("p1d/component_reconciliation.json")
    cols = ["B|sym1", "B|sym2", "A+staged|sym1", "A+staged|sym2"]
    groups = (("bridge fuel & VOM", ("bridge_fuel_om",)), ("capex", ("capex",)), ("grid purchases", ("grid",)),
              ("post-connection CHP (fuel + O&M)", ("post_chp",)), ("backup", ("backup",)),
              ("retirement value", ("retirement_value",)), ("terminal value", ("terminal_value",)),
              ("rental, filing, fixed O&M", ("rental", "filing", "fixed_om", "delay")))
    rows = [[name] + [signed(sum(d[c]["components"][p] for p in parts), 2) for c in cols] for name, parts in groups]
    rows.append(["**total**"] + [f"**{signed(d[c]['total'], 2)}**" for c in cols])
    rows.append(["by estimate category: early / middle / late"] +
                [" / ".join(f"{sum(v[y] for v in d[c]['by_category'].values()):.2f}" for y in range(3)) for c in cols])
    return table(["Component ($M)"] + [c.replace("|", " ") for c in cols], rows)


def cells():
    return {(c["cfg"], c["level"]): c for c in load("p1d/representative_cells.json")["cells"]}


def timing(C):
    rows = []
    for cfg in ("A+staged", "B"):
        d = C[cfg, "high"]["decisions"]
        ns = d["no_signal"]["timing"]
        ge = ns["GE"]
        common = [f"GE {ge['root']:.1f} + {ge['pre_signal']:.1f} MW"]
        for asset, unit in (("DG", "MW"), ("ABS", "MWth")):
            t = d["signal"][0]["timing"][asset]               # decided before the forecast, so the same in every copy
            if asset == "DG" and cfg != "B":
                continue                                       # the table lists DG where it is the bridge commitment
            total = t["root"] + t["pre_signal"]
            if total >= 0.05:
                common.append(f"{asset} {total:.1f} {unit}")
        post = [ge["post_signal"]] + [s["timing"]["GE"]["post_signal"] for s in d["signal"]]
        rows.append([cfg, "; ".join(common), " / ".join(plain(x) for x in post) + " MW"])
    return table(["Configuration", "Common orders (root + pre-signal)",
                  "Post-signal GE orders: no-signal / early / middle / late"], rows)


def fuel(C):
    keys = (("gas_MMBtu", 0), ("diesel_gal", 0), ("grid_MWh", 0), ("onsite_fuel_MWh_HHV", None))
    rows = []
    for cfg in ("B", "A+staged", "B+staged"):
        row = [cfg]
        for key, places in keys:
            v = []
            for level in ("low", "high"):
                q = C[cfg, level]["quantities"]
                a, b = q["no_signal"]["all"][key], q["signal"]["all"][key]
                v.append(100 * (b - a) / a)
            same_sign = v[0] * v[1] > 0
            v.sort(key=abs if same_sign else None)
            p = places if places is not None else (0 if min(abs(x) for x in v) >= 5 else 1)
            cell = f"{signed(v[0], p)} % to {signed(v[1], p)} %"
            row.append(f"**{cell}**" if (cfg, key) == ("A+staged", "onsite_fuel_MWh_HHV") else cell)
        rows.append(row)
    return table(["Configuration", "Gas", "Diesel", "Grid MWh", "On-site fuel energy"], rows)


def certificates():
    d = load("gate2_vertex_certificates_A_A-staged_B.json")["results"]

    def m(x, plus=False):
        sign = MINUS if x < 0 else "+" if plus else ""
        return f"{sign}${abs(x):.3f}M"

    rows = []
    for key, name in (("A|one", "A, one type"), ("A|two", "A, two types"), ("A+staged|one", "A + staged, one type"),
                      ("A+staged|two", "A + staged, two types"), ("B|one", "B, one type"), ("B|two", "B, two types")):
        r = d[key]
        lo, hi = min(r["robust_minus_predictive"]), max(r["robust_minus_predictive"])  # one value per vertex
        first = [name, m(r["cert_oracle_gap_predictive"]), m(r["cert_info_value_predictive"], True)]
        if key.startswith("B"):
            first = [f"**{x}**" for x in first]
        rows.append(first + [f"{m(r['cert_oracle_gap_robust'])} / {m(r['cert_info_value_robust'], True)}",
                             f"{signed(lo, 2)} … {signed(hi, 2)}"])
    return table(["Configuration", "certified room over predictive ≤",
                  "certified implemented info value of predictive ≥", "robust: room ≤ / info ≥",
                  "robust − predictive at vertices"], rows)


def main():
    C = cells()
    out = ["# Supplement tables computed from the released JSON files (generated by "
           "`experiments/make_supplement_tables.py`)", ""]
    for head, body in (("S5. Value by cost component", money()),
                       ("S6. Orders in the representative cells", timing(C)),
                       ("S7. Fuel and grid use in the representative cells", fuel(C)),
                       ("S34. Certificates over the reliability boxes", certificates())):
        out += [f"## {head}", ""] + body + [""]
    (ROOT / "data" / "supplement_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("written data/supplement_tables.md")


if __name__ == "__main__":
    main()
