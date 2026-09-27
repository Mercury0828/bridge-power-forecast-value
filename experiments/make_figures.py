"""Result figures from the frozen artefacts (Phase 1f S8; *Energy* #1 lens 16: visual quality only).

    python experiments/make_figures.py [name ...]   -> figures/<name>.pdf and .png (300 dpi)
    python experiments/make_figures.py --paper [name ...]   -> paper/figs/<name>.pdf and .png: the manuscript versions,
        140 mm wide (the preprint text width is 137 mm), four-panel figures as 2 x 2, the tornado stacked, and the
        manuscript's terms (forecast, no forecast, face value). The frozen figures/ versions are unchanged.

**Style.** Elsevier double-column page: 90 mm (single) and 190 mm (double) widths; ≥ 7 pt text; greyscale-safe (marker
and line-style redundancy, no colour-only encoding); ticks inward so tick marks never cross tick labels. Every number is
read from a frozen summary. Nothing is typed in.

**Figures:**
- fig_main_values: E[Δ_info] by configuration and error family (Phase 1c registered).
- fig_implementation: the registered contrasts under each implementation variant (Phase 1c/1d/1e).
- fig_oracle_decomp: Δ_info = V_true + G0 − R_Y (Proposition 5).
- fig_staging_2x2: cap, backup-delay and interaction effects under three procurement models (Phase 1d B1; Phase 1e
  E4a/E4a').
- fig_policy_illustration: cumulative post-root GE orders by estimate category in the representative B cell.
- fig_tpit_errors: TPIT schedule-error pmfs vs the declared families and txUS.
- fig_co2_grid: ΔCO2 (signal − no-signal) vs the grid emission factor.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402

PAPER = "--paper" in sys.argv
FIG = ROOT / ("paper/figs" if PAPER else "figures")
# Lint aid: tools/figure_lint.py reads the tiles of hatch patterns as page strokes (their coordinates lie outside the
# page), which inflates its content width and its line-through-text count. MF_NOHATCH=1 renders the same figures
# without hatches into MF_OUT, so the mechanical checks can be run on them; the printed figures keep their hatches.
if os.environ.get("MF_NOHATCH") == "1":
    import matplotlib.patches as _mpatches
    _mpatches.Patch.set_hatch = lambda self, hatch: setattr(self, "_hatch", None)
    FIG = pathlib.Path(os.environ["MF_OUT"])
MM = 1 / 25.4
W1, W2 = (88 if PAPER else 90) * MM, (137 if PAPER else 190) * MM   # paper: tight bbox adds ~2 mm
# axis labels and legend terms; the manuscript says "forecast" where the internal records say "estimate"
TXT = dict(value="E[Δ_info] ($M)", arrives="estimate\narrives", none="no estimate", cat="estimate: ",
           tau_axis="Quarter in which the utility estimate arrives (τ)", aware="estimate-aware",
           naive="naive (category taken as exact)", naive_title="Naive use of the estimate", ref="0.5 % of expected lifecycle cost")
if PAPER:                                       # plain text only: mathtext subscripts print below 7 pt
    TXT.update(value="Value of the forecast ($M)", arrives="forecast\narrives", none="no forecast", cat="forecast: ",
               tau_axis="Quarter in which the forecast arrives, τ", aware="forecast-aware",
               naive="face value (category taken as exact)", naive_title="Face-value use of the forecast",
               ref="0.5% of main-case no-forecast cost")
FAM_SHORT = {"sym1": "σ = 1", "sym2": "σ = 2", "sym4": "σ = 4", "opt2": "biased", "uninf": "uninf."}
GREYS = ["0.0", "0.35", "0.6", "0.8", "1.0"]
HATCH = ["", "////", "", "xxxx", "...."]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")
FAM_LABEL = {"sym1": "σ = 1 q", "sym2": "σ = 2 q", "sym4": "σ = 4 q", "opt2": "biased (−2 q)", "uninf": "uninformative"}
CFG_LABEL = {"A": "ERCOT-like (A)", "A+staged": "ERCOT-like, staged", "B": "Dominion-like (B)",
             "B+staged": "Dominion-like, staged"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---- manuscript design system (2026-09-27). Okabe-Ito colours, CIE L* in brackets: blue (46.0), sky (69.8),
# amber (70.6), vermilion (54.2), teal (57.7), purple (61.0), grey (68.1), ink (13.2), slate (43.2). Every colour
# distinction is also carried by a hatch, a marker or a line style, so a greyscale print keeps every series apart.
PC = dict(blue="#0072B2", sky="#56B4E9", amber="#E69F00", verm="#D55E00", teal="#009E73", purple="#CC79A7",
          grey="#A6A6A6", ink="#222222", slate="#666666")
DASH, DOT, DASHDOT, REFDASH = (0, (4, 2)), (0, (1, 1.8)), (0, (5, 2, 1, 2)), (0, (4, 2.5))
FAM_P = {"sym1": (PC["blue"], "", "-", "o"), "sym2": (PC["sky"], "//", DASH, "s"),       # colour, bar hatch,
         "sym4": (PC["amber"], "", DASHDOT, "^"), "opt2": (PC["purple"], "x", DOT, "D"),    # line style, marker
         "uninf": (PC["grey"], "..", "-", "v")}
CFG_P = {"A": (PC["blue"], "o", "-"), "A+staged": (PC["blue"], "s", DASH), "B": (PC["verm"], "^", DASHDOT),
         "B+staged": (PC["verm"], "D", DOT)}                                                  # colour, marker, line
IMPL_P = [(PC["ink"], "o", True), (PC["blue"], "s", False), (PC["blue"], "^", False), (PC["verm"], "D", False),
          (PC["teal"], "v", False), (PC["teal"], "P", False), (PC["verm"], "X", False), (PC["purple"], "*", False),
          (PC["slate"], "h", False)]                                                          # colour, marker, filled
if PAPER:
    plt.rcParams.update({
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": PC["ink"], "axes.labelcolor": PC["ink"], "text.color": PC["ink"],
        "xtick.color": PC["ink"], "ytick.color": PC["ink"], "lines.linewidth": 1.15, "lines.markersize": 4.2,
        "lines.markeredgewidth": 0.8, "lines.scale_dashes": False, "lines.dash_capstyle": "round",
        "hatch.linewidth": 0.35, "hatch.color": PC["ink"], "patch.linewidth": 0.55, "legend.handlelength": 3.0,
        "legend.handletextpad": 0.7, "legend.columnspacing": 1.6, "legend.labelspacing": 0.8, "axes.titlepad": 5.7})


def _j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


# ---- manuscript canvases: exact size in mm, axes placed in mm from the top-left corner, no tight bounding box -------
def _fig_mm(w, h):
    return plt.figure(figsize=(w * MM, h * MM))


def _ax_mm(fig, w, h, left, top, width, height, **kw):
    return fig.add_axes((left / w, 1 - (top + height) / h, width / w, height / h), **kw)


def _legend_mm(fig, w, h, handles, labels, x, y, ncol=1, **kw):
    return fig.legend(handles, labels, loc="center", bbox_to_anchor=(x / w, 1 - y / h), ncol=ncol, frameon=False,
                      **kw)


def _save_exact(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})   # reproducible bytes
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    where = FIG.relative_to(ROOT).as_posix() if FIG.is_relative_to(ROOT) else str(FIG)
    print(f"written {where}/{name}.pdf/.png")


def _clear(ax, values, span_mm, axis="y", pad_mm=3.0):
    """Axis limits that leave pad_mm (marker radius, stroke and 2 mm) between the outermost element and the frame."""
    lo, hi = min(values), max(values)
    pad = (hi - lo) * pad_mm / max(span_mm - 2 * pad_mm, 1e-9)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo - pad, hi + pad)


def _refline(ax, v, vertical=False):
    kw = dict(color=PC["ink"], linestyle=REFDASH, linewidth=0.8, zorder=2)
    return ax.axvline(v, **kw) if vertical else ax.axhline(v, **kw)


def _refhandle():
    return plt.Line2D([], [], color=PC["ink"], linestyle=REFDASH, linewidth=0.8)


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    pad = dict(pad_inches=0.14) if PAPER else {}          # a stroke at the axes edge otherwise overhangs the page
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", **pad)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", **pad)
    plt.close(fig)
    where = FIG.relative_to(ROOT).as_posix() if FIG.is_relative_to(ROOT) else str(FIG)
    print(f"written {where}/{name}.pdf/.png")


def _ref(cfg):
    """The Phase-1c reference: 0.5 % of the configuration's expected no-signal cost (sym1 row, Δ = 0)."""
    return 0.005 * _j("data/p1c_summary.json")["table"][f"{cfg}|sym1|0"]["mean_J_nosig"]


# ---- 1. main values ------------------------------------------------------------------------------------------------
def fig_main_values():
    if PAPER:
        return _p_main_values()
    T = _j("data/p1c_summary.json")["table"]
    if PAPER:
        fig, axes = plt.subplots(2, 2, figsize=(W2, 104 * MM))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(W2, 58 * MM))
    x = np.arange(len(FAMS))
    for ax, cfg in zip(axes, core.CONFIGS):
        m = [T[f"{cfg}|{f}|0"]["mean"] for f in FAMS]
        lo = [T[f"{cfg}|{f}|0"]["by_level"]["low"] for f in FAMS]
        hi = [T[f"{cfg}|{f}|0"]["by_level"]["high"] for f in FAMS]
        for i in range(len(FAMS)):
            ax.bar(x[i], m[i], 0.62, color=GREYS[min(i, 3)], edgecolor="0", linewidth=0.5, hatch=HATCH[i])
        ax.scatter(x, lo, marker="v", s=12, facecolor="white", edgecolor="0", linewidth=0.6, zorder=3,
                   label="low dispersion")
        ax.scatter(x, hi, marker="^", s=12, facecolor="0", edgecolor="0", linewidth=0.6, zorder=3,
                   label="high dispersion")
        ax.axhline(_ref(cfg), color="0", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="0", linewidth=0.5)
        ax.set_title(CFG_LABEL[cfg])
        ax.set_xticks(x)
        if PAPER:                               # horizontal short labels; the caption spells them out
            ax.set_xticklabels([FAM_SHORT[f] for f in FAMS])
        else:
            ax.set_xticklabels([FAM_LABEL[f] for f in FAMS], rotation=40, ha="right", rotation_mode="anchor")
    for ax in (axes[::2] if PAPER else axes[:1]):
        ax.set_ylabel(TXT["value"])
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="0", linestyle="--", linewidth=0.7))
    l.append(TXT["ref"])
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3)
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0.05 if PAPER else 0.08, 1, 1))
    _save(fig, "fig_main_values")


# ---- 2. implementation robustness ----------------------------------------------------------------------------------
IMPL = [  # label, summary file, key pattern
    ("benchmark (registered)", "data/p1c_summary.json", "{b}"),
    ("lead-consistent standby", "data/p1d/spine_summary.json", "{b}|spine"),
    ("+ rented standby", "data/p1d/e1_summary.json", "{b}|spine+rsb"),
    ("whole units, batch", "data/p1d/e2_summary.json", "{b}|batch"),
    ("lead-consistent + batch", "data/p1d/e4b_spine_summary.json", "{b}|spine|batch"),
    ("rented standby + batch", "data/p1d/e4b_rsb_summary.json", "{b}|spine+rsb|batch"),
    ("whole units, round-up", "data/p1d/units_summary.json", "{b}|units"),
    ("recommissioning (moderate)", "data/p1d/refurb_summary.json", "{b}|refurb_mod"),
    ("residence rule off", "data/p1d/permit_summary.json", "{b}|permit"),
]
if PAPER:                                       # the manuscript's names for the same variants (same files and keys)
    IMPL = [(lab, f, pat) for lab, (_, f, pat) in zip(
        ["zero-lead benchmark (registered)", "lead-consistent reference", "lead-consistent + rented standby",
         "whole units, batched", "lead-consistent + batched", "rented standby + batched", "whole units, round-up",
         "recommissioning (moderate)", "residence rule removed"], IMPL)]
IMPL_MAIN = (0, 1, 2, 3, 4)                     # the manuscript's Fig. 5; fig_implementation_all shows all nine
ROWS = [("B, σ = 1 q", "B", "sym1", False), ("B, σ = 2 q", "B", "sym2", False),
        ("ERCOT-like staged, σ = 1 q", "A+staged", "sym1", False), ("ERCOT-like staged, σ = 2 q", "A+staged", "sym2", False),
        ("staged − unstaged, σ = 1 q", "A", "sym1", True), ("staged − unstaged, σ = 2 q", "A", "sym2", True)]


def fig_implementation(show=None, name="fig_implementation"):
    show = tuple(range(len(IMPL))) if show is None else show
    if PAPER:
        return _p_implementation(show, name)
    S = {lab: _j(f)["table"] for lab, f, _ in IMPL}
    n = len(show)
    panels = [("Dominion-like (B)", [r for r in ROWS if r[1] == "B"]),
              ("ERCOT-like (A)", [r for r in ROWS if r[1] != "B"])]
    height = (78 if n > 6 else 66) if PAPER else 62
    fig, axes = plt.subplots(1, 2, figsize=(W2 - (6 if PAPER else 0) * MM, height * MM),
                             gridspec_kw=dict(width_ratios=[1, 1.6]))
    for ax, (title, rows) in zip(axes, panels):
        ys = np.arange(len(rows))[::-1]
        for (rlab, b, fam, inter), y in zip(rows, ys):
            ref = _ref("B") if b == "B" else _ref("A+staged")
            ax.plot([ref, ref], [y - 0.42, y + 0.42], color="0", linestyle="--", linewidth=0.7)
            for pos, i in enumerate(show):
                lab, _, pat = IMPL[i]
                T = S[lab]
                try:
                    if inter:
                        v = T[f"{pat.format(b='A+staged')}|{fam}|0"]["mean"] - T[f"{pat.format(b='A')}|{fam}|0"]["mean"]
                    else:
                        v = T[f"{pat.format(b=b)}|{fam}|0"]["mean"]
                except KeyError:
                    continue
                ax.scatter(v, y + ((n - 1) / 2 - pos) * 0.64 / (n - 1), marker=MARK[i], s=16,
                           facecolor="0" if i == 0 else "white",
                           edgecolor="0", linewidth=0.6, zorder=3, label=lab)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0].replace("ERCOT-like staged", "staged") for r in rows])
        ax.set_title(title)
        if PAPER:                               # the row labels already say "staged − unstaged"
            ax.set_xlabel(TXT["value"])
        else:
            ax.set_xlabel(TXT["value"] if b == "B" else TXT["value"] + "; bottom rows: staged − unstaged")
        if b != "B":
            ax.axvline(0, color="0", linewidth=0.5)
    h, l = axes[1].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    handles = list(seen.values()) + [plt.Line2D([], [], color="0", linestyle="--", linewidth=0.7)]
    labels = list(seen.keys()) + [TXT["ref"]]
    fig.legend(handles, labels, loc="lower center", ncol=2 if PAPER else 5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.2, columnspacing=1.2 if PAPER else 0.9)
    fig.tight_layout(w_pad=1.5, rect=(0, (0.29 if n > 6 else 0.22) if PAPER else 0.15, 1, 1))
    _save(fig, name)


def fig_implementation_main():
    fig_implementation(IMPL_MAIN if PAPER else None)


def fig_implementation_all():
    fig_implementation(None, "fig_implementation_all")


# ---- 3. oracle decomposition ---------------------------------------------------------------------------------------
def fig_oracle_decomp():
    if PAPER:
        return _p_oracle_decomp()
    D = _j("data/p1d/oracle_decomposition.json")["decomposition"]
    keys = [("B", "sym1"), ("B", "sym2"), ("A+staged", "sym1"), ("A+staged", "sym2")]
    labels = ["B\nσ = 1 q", "B\nσ = 2 q", "A staged\nσ = 1 q", "A staged\nσ = 2 q"]
    fig, ax = plt.subplots(figsize=(W1, 60 * MM))
    x = np.arange(len(keys))
    vt = [D[f"{c}|{f}"]["mean"]["V_true"] for c, f in keys]
    g0 = [D[f"{c}|{f}"]["mean"]["G0"] for c, f in keys]
    ry = [D[f"{c}|{f}"]["mean"]["R_Y"] for c, f in keys]
    di = [D[f"{c}|{f}"]["mean"]["d_info"] for c, f in keys]
    names = (("V_true", "G_0", "−G_Y", "value = V_true + G_0 − G_Y") if PAPER
             else ("V_true", "G0", "−R_Y", "Δ_info = V_true + G0 − R_Y"))
    ax.bar(x - 0.25, vt, 0.25, color="0.35", edgecolor="0", linewidth=0.5, label=names[0])
    ax.bar(x, g0, 0.25, color="white", edgecolor="0", linewidth=0.5, hatch="////", label=names[1])
    ax.bar(x + 0.25, [-r for r in ry], 0.25, color="0.8", edgecolor="0", linewidth=0.5, label=names[2])
    ax.scatter(x, di, marker="D", s=18, facecolor="0", edgecolor="0", zorder=3, label=names[3])
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$M (expected, Δ = 0)")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "fig_oracle_decomp")


# ---- 4. staging 2x2 ----------------------------------------------------------------------------------------------------
def _twobytwo(fam, S00, k00, k11, S10, k10, k01):
    i00, i11 = S00[f"{k00}|{fam}|0"]["mean"], S00[f"{k11}|{fam}|0"]["mean"]
    i10, i01 = S10[f"{k10}|{fam}|0"]["mean"], S10[f"{k01}|{fam}|0"]["mean"]
    return i10 - i00, i01 - i00, i11 - i10 - i01 + i00


def fig_staging_2x2():
    if PAPER:
        return _p_staging_2x2()
    p1c, stage = _j("data/p1c_summary.json")["table"], _j("data/p1d/stage_summary.json")["table"]
    spine, e4a = _j("data/p1d/spine_summary.json")["table"], _j("data/p1d/e4a_summary.json")["table"]
    e2, e4a2 = _j("data/p1d/e2_summary.json")["table"], _j("data/p1d/e4a2_summary.json")["table"]
    models = [("benchmark", lambda f: _twobytwo(f, p1c, "A", "A+staged", stage, "A|cap", "A|bkdelay")),
              ("lead-consistent", lambda f: _twobytwo(f, spine, "A|spine", "A+staged|spine", e4a, "A|cap|spine",
                                                       "A|bkdelay|spine")),
              ("whole units (batch)", lambda f: _twobytwo(f, e2, "A|batch", "A+staged|batch", e4a2, "A|cap|batch",
                                                          "A|bkdelay|batch"))]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 55 * MM), sharey=True)
    parts = ["import cap", "backup delay", "interaction"]
    for ax, fam in zip(axes, ("sym1", "sym2")):
        x = np.arange(len(models))
        vals = np.array([m[1](fam) for m in models])
        for j in range(3):
            ax.bar(x + (j - 1) * 0.25, vals[:, j], 0.25, color=GREYS[j + 1], edgecolor="0", linewidth=0.5,
                   hatch=["", "////", "...."][j], label=parts[j])
            for xi, v in zip(x, vals[:, j]):
                if abs(v) < 0.005:
                    ax.text(xi + (j - 1) * 0.25, 0.08, "0", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="0", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["zero-lead\nbenchmark", "lead-\nconsistent", "whole units\n(batched)"] if PAPER
                           else [m[0] for m in models])
        ax.set_title(f"ERCOT-like, σ = {fam[-1]} q")
    axes[0].set_ylabel("Effect on the value of the forecast ($M)" if PAPER else "Effect on E[Δ_info] ($M)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig_staging_2x2")


# ---- 5. policy illustration -------------------------------------------------------------------------------------------
def fig_policy_illustration():
    if PAPER:
        return _p_policy_illustration()
    run = ROOT / "experiments" / "r2_runs" / "p1c_main"
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    cells = [c for c in plan["cells"] if c["cfg"] == "B" and c["level"] == "high"]
    c = max(cells, key=lambda x: x["p"])
    load = lambda k: json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    ns, sg = load(c["jobs"]["nosig"]), load(c["jobs"]["sig"])
    if PAPER:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W2 - 2 * MM, 60 * MM))
    else:
        fig, ax = plt.subplots(figsize=(W1, 58 * MM))
    nodes = np.arange(len(ns["detail"]["orders"]["GE"]))
    shown = nodes <= 20 if PAPER else nodes >= 0   # the paper version draws only the visible nodes (no clipped paths)
    styles = [(TXT["none"], ns["detail"]["orders"]["GE"], "-", "o", "0")]
    for y, cat in enumerate(("early", "middle", "late")):
        styles.append((TXT["cat"] + cat, sg["detail"][y]["orders"]["GE"], ["--", "-.", ":"][y], MARK[y + 1], "0.35"))
    for lab, orders, ls, mk, col in styles:
        ax.plot(nodes[shown], np.cumsum(orders)[shown], drawstyle="steps-post", linestyle=ls, color=col, marker=mk,
                markersize=3, markevery=4, markerfacecolor="white", label=lab)
    ax.axvline(core.TAU, color="0", linewidth=0.5)
    ax.text(core.TAU - 0.3, ax.get_ylim()[1] * 0.95, TXT["arrives"], fontsize=7, va="top", ha="right")
    ax.set_xlabel("Planning node (quarter)")
    ax.set_ylabel("Cumulative gas-CHP orders (MW)")
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(range(0, 21, 4))
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    if PAPER:                                   # capacity in service while the grid has not arrived: F_q = sum_{k<=q-l}
        P = core.params("B")
        lead = P.lead["GE"]
        qs = np.arange(0, 25)
        for lab, orders, ls, mk, col in styles:
            cum = np.cumsum(orders)
            inservice = [cum[q - lead] if q - lead >= 0 else 0.0 for q in qs]
            ax2.plot(qs, inservice, drawstyle="steps-post", linestyle=ls, color=col, marker=mk, markersize=3,
                     markevery=4, markerfacecolor="white", label=lab)
        ax2.plot(qs, [(1 + P.a_oth) * P.L(q) for q in qs], drawstyle="steps-post", color="0.6", linewidth=1.8,
                 zorder=0, label="IT and other electricity")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Gas-CHP capacity in service (MW)")
        ax2.set_xlim(-0.6, 24.6)
        ax2.set_xticks(range(0, 25, 4))
        h, l = ax2.get_legend_handles_labels()
        ax2.legend(h[-1:], l[-1:], loc="lower right", frameon=False, handlelength=2.2)
        ax.set_title("(a) Orders")
        ax2.set_title("(b) In service before connection")
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig_policy_illustration")


# ---- 6. TPIT error distributions -----------------------------------------------------------------------------------
def fig_tpit_errors():
    if PAPER:
        return _p_tpit_errors()
    tp = _j("data/p1e/tpit_errors.json")["summary"]
    tx = _j("data/p1e/txus_summary.json")["family"]["pmf"]
    e = np.arange(-16, 17)
    vis = (e >= -12) & (e <= 6) if PAPER else np.ones_like(e, dtype=bool)   # paper: only the visible range
    fig, ax = plt.subplots(figsize=(W1, 58 * MM))
    for i, h in enumerate(("h4|all", "h8|all", "h12|all")):
        pm = tp[h]["pmf"]
        y = np.array([pm.get(str(k), 0.0) for k in e])
        ax.plot(e[vis], y[vis], linestyle=["-", "--", ":"][i], marker=MARK[i], markersize=2.5, color="0",
                markerfacecolor="white", label=f"TPIT, {h.split('|')[0][1:]} q ahead (n = {tp[h]['n']})")
    ax.plot(e[vis], np.array([tx.get(str(k), 0.0) for k in e])[vis], linestyle="-.", color="0.5", label="txUS proxy")
    ax.plot(e[vis], np.asarray(core.error_pmf(2.0))[vis], linestyle="-", color="0.6", linewidth=1.6, alpha=0.8,
            label="declared σ = 2 q")
    ax.set_xlim(-12, 6)
    ax.set_xlabel("Projected − actual (quarters; < 0: late)" if PAPER
                  else "Estimate − actual (quarters; < 0: later than estimated)")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_tpit_errors")


# ---- 7. CO2 vs grid factor ---------------------------------------------------------------------------------------------
def fig_co2_grid():
    if PAPER:
        return _p_co2_grid()
    em = _j("data/p1e/emissions_summary.json")["table"]
    fs = (0.2, 0.4, 0.6, 0.8)
    fig, ax = plt.subplots(figsize=(W1, 92 * MM))
    for i, (b, impl) in enumerate([("A+staged", "continuous"), ("A+staged", "batched"), ("B", "continuous"),
                                   ("B", "batched")]):
        y = [em[f"{b}|sym2|{impl}|sweep {f}"]["delta_co2_kt"] for f in fs]
        ax.plot(fs, y, linestyle="-" if impl == "continuous" else "--", marker=MARK[0 if b == "B" else 1],
                markersize=3.5, color="0" if b == "B" else "0.45", markerfacecolor="white",
                label=f"{'Dominion-like' if b == 'B' else 'ERCOT-like staged'}, {impl}")
    for b, name, mk, lab in (("A+staged", "eGRID2023 ERCT total", "^", "eGRID2023 ERCT (ERCOT-like)"),
                             ("B", "eGRID2023 SRVC total", "v", "eGRID2023 SRVC (Dominion-like)")):
        ef = 733.9 * 0.000453592 if b == "A+staged" else 593.4 * 0.000453592
        ax.scatter([ef], [em[f"{b}|sym2|continuous|{name}"]["delta_co2_kt"]], marker=mk, s=22, facecolor="0",
                   zorder=4, label=lab)
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xlabel("Grid emission factor (t CO2/MWh)")
    ax.set_ylabel("ΔCO2 (kt), forecast-aware − no forecast" if PAPER else "ΔCO2 (kt), estimate-aware − none")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    _save(fig, "fig_co2_grid")


# ---- Phase 1f figures -----------------------------------------------------------------------------------------------
def _p1f(v):
    p = ROOT / "data" / "p1d" / f"{v}_summary.json"
    return json.loads(p.read_text(encoding="utf-8"))["table"] if p.exists() else None


def fig_value_vs_tau():
    """Every family, so that the calibration effect is shown (Phase 1f E-T: in B the value falls with τ only for
    families at least as accurate as the planner's kernel)."""
    if PAPER:
        return _p_value_vs_tau()
    base = _j("data/p1c_summary.json")["table"]
    t2, t6 = _p1f("tau2"), _p1f("tau6")
    if PAPER:
        fig, axes = plt.subplots(2, 2, figsize=(W2, 88 * MM))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(W2, 60 * MM))
    taus = (2, 4, 6)
    style = {"sym1": ("-", "o", "0"), "sym2": ("--", "s", "0"), "sym4": ("-.", "^", "0"), "opt2": (":", "D", "0"),
             "uninf": ("-", "v", "0.55")}
    for ax, c in zip(axes, core.CONFIGS):
        for fam in FAMS:
            ys = [t2[f"{c}|tau2|{fam}|0"]["mean"], base[f"{c}|{fam}|0"]["mean"], t6[f"{c}|tau6|{fam}|0"]["mean"]]
            ls, mk, col = style[fam]
            ax.plot(taus, ys, linestyle=ls, marker=mk, color=col, markerfacecolor="white", markersize=4,
                    label=FAM_LABEL[fam])
        ax.axhline(0, color="0", linewidth=0.5)
        ax.set_xticks(taus)
        ax.set_xlim(1.5, 6.5)
        ax.set_title(CFG_LABEL[c])
    for ax in (axes[::2] if PAPER else axes[:1]):
        ax.set_ylabel(TXT["value"])
    if PAPER:
        fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=(0, 0.12, 1, 1))
        fig.text(0.5, 0.105, TXT["tau_axis"], ha="center", va="top", fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 0.06), handletextpad=0.3, columnspacing=1.0)
    else:                                       # the frozen figures/ layout, unchanged
        fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
        fig.text(0.5, 0.115, TXT["tau_axis"], ha="center", va="top", fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 0.06))
    _save(fig, "fig_value_vs_tau")


ES_ORDER = ["s_disc", "s_gas", "s_diesel", "s_grid", "s_capex", "s_life", "s_rent", "s_co2", "s_eff"]
ES_LEVELS = {"s_disc": ("disc_lo", "disc_hi"), "s_gas": ("gas_lo", "gas_hi"), "s_diesel": ("diesel_lo", "diesel_hi"),
             "s_grid": ("grid_lo", "grid_hi"), "s_capex": ("capex_lo", "capex_hi"), "s_life": ("life_lo", "life_hi"),
             "s_rent": ("rent_lo", "rent_hi"), "s_co2": ("co2_lo", "co2_hi"), "s_eff": ("eff", "eff")}
# tested values / main case (docs/phase1f_sources.md; experiments/p1d_core.py ES_*)
ES_LABEL = {"s_disc": {"A": "discount rate 5 / 11 % (8)", "B": "discount rate 5 / 11 % (8)"},
            "s_gas": {"A": "gas $2.5 / 6.5 per MMBtu (3.5)", "B": "gas $4 / 8 per MMBtu (5)"},
            "s_diesel": {"A": "diesel $2.5 / 4.7 per gal (3.6)", "B": "diesel $2.7 / 4.9 per gal (3.7)"},
            "s_grid": {"A": "grid $35 / 70 per MWh (50)", "B": "grid $55 / 100 per MWh (75)"},
            "s_capex": {"A": "capital costs low / high", "B": "capital costs low / high"},
            "s_life": {"A": "asset lives short / long", "B": "asset lives short / long"},
            "s_rent": {"A": "rental $17 / 35 per kW-month (25)", "B": "rental $17 / 35 per kW-month (25)"},
            "s_co2": {"A": "carbon price $25 / 286 per t (0)", "B": "carbon price $25 / 286 per t (0)"},
            "s_eff": {"A": "heat rates +10 %", "B": "heat rates +10 %"}}


def fig_tornado():
    if PAPER:
        return _p_tornado()
    """σ = 2 q (the lower of the two registered families); sym1 and H3 are in data/p1f_tables.md."""
    base = _j("data/p1c_summary.json")["table"]
    if PAPER:                                   # stacked: the long level labels need the width
        fig, axes = plt.subplots(2, 1, figsize=(W2 - 3 * MM, 124 * MM))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM))
    for ax, (c, fam, title) in zip(axes, (("B", "sym2", "Dominion-like (B), σ = 2 q"),
                                          ("A+staged", "sym2", "ERCOT-like, staged, σ = 2 q"))):
        b0 = base[f"{c}|{fam}|0"]["mean"]
        rows = []
        for v in ES_ORDER:
            T = _p1f(v)
            if T is None:
                continue
            lo, hi = ES_LEVELS[v]
            a, b = T[f"{c}|{lo}|{fam}|0"]["mean"], T[f"{c}|{hi}|{fam}|0"]["mean"]
            rows.append((ES_LABEL[v][c[0]], a, b, lo == hi))
        rows.sort(key=lambda r: max(abs(r[1] - b0), abs(r[2] - b0)))
        for y, (lab, a, b, single) in enumerate(rows):
            if single:                                      # one tested value, drawn as the higher one
                ax.barh(y, b - b0, left=b0, height=0.6, color="0.45", edgecolor="0", linewidth=0.6)
                continue
            ax.barh(y + 0.16, a - b0, left=b0, height=0.32, color="white", edgecolor="0", linewidth=0.6,
                    hatch="////")
            ax.barh(y - 0.16, b - b0, left=b0, height=0.32, color="0.45", edgecolor="0", linewidth=0.6)
        ax.axvline(b0, color="0", linewidth=0.8)
        ax.axvline(_ref(c), color="0", linestyle="--", linewidth=0.7)
        xs = [b0, _ref(c)] + [r[1] for r in rows] + [r[2] for r in rows]
        pad = 0.06 * (max(xs) - min(xs))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel(TXT["value"])
        ax.set_title(title)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0", hatch="////"),
         plt.Rectangle((0, 0), 1, 1, facecolor="0.45", edgecolor="0"),
         plt.Line2D([], [], color="0", linewidth=0.8), plt.Line2D([], [], color="0", linestyle="--", linewidth=0.7)]
    fig.legend(h, ["lower tested value", "higher tested value", "main case", TXT["ref"]], loc="lower center",
               ncol=2 if PAPER else 4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if PAPER:
        fig.tight_layout(h_pad=1.5, rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    _save(fig, "fig_tornado")


def fig_naive_and_size():
    base = _j("data/p1c_summary.json")["table"]
    nv, s0, s3 = _p1f("naive"), _p1f("size075"), _p1f("size300")
    fig, axes = plt.subplots(1, 2, figsize=(W2, (86 if PAPER else 72) * MM))
    ax = axes[0]
    x = np.arange(len(core.CONFIGS))
    for j, fam in enumerate(("sym1", "sym2")):
        reg = [base[f"{c}|{fam}|0"]["mean"] for c in core.CONFIGS]
        nai = [nv[f"{c}|naive|{fam}|0"]["mean"] for c in core.CONFIGS]
        ax.bar(x + (j - 0.5) * 0.38 - 0.09, reg, 0.18, color="0.35", edgecolor="0", linewidth=0.5,
               hatch=["", "////"][j], label=f"{TXT['aware']}, σ = {fam[-1]} q")
        ax.bar(x + (j - 0.5) * 0.38 + 0.09, nai, 0.18, color="white", edgecolor="0", linewidth=0.5,
               hatch=["", "////"][j], label=f"{TXT['naive']}, σ = {fam[-1]} q")
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "A staged", "B", "B staged"])
    ax.set_ylabel(TXT["value"])
    ax.set_title(TXT["naive_title"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1 if PAPER else 2, frameon=False, fontsize=7,
              handletextpad=0.4, columnspacing=1.0)
    ax = axes[1]
    for i, c in enumerate(core.CONFIGS):
        pts = [(75, s0[f"{c}|size075|sym2|0"]["mean"]), (150, base[f"{c}|sym2|0"]["mean"]),
               (300, s3[f"{c}|size300|sym2|0"]["mean"])]
        ax.plot([p[0] for p in pts], [100 * p[1] / p[0] for p in pts], marker=MARK[i],
                linestyle=["-", "--", "-.", ":"][i], color="0", markerfacecolor="white", label=CFG_LABEL[c])
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xticks((75, 150, 300))
    ax.set_xlabel("Campus peak IT load (MW)")
    ax.set_ylabel("$M per 100 MW of IT load (σ = 2 q)" if PAPER else "$M per 100 MW-IT (σ = 2 q)")
    ax.set_title("Campus size")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1 if PAPER else 2, frameon=False, fontsize=7,
              handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig_naive_and_size")


def fig_naive():
    """The manuscript's Fig. 7: the face-value panel of fig_naive_and_size, alone."""
    if PAPER:
        return _p_naive()
    base = _j("data/p1c_summary.json")["table"]
    nv = _p1f("naive")
    fig, ax = plt.subplots(figsize=(W1 + 12 * MM, 74 * MM))
    x = np.arange(len(core.CONFIGS))
    for j, fam in enumerate(("sym1", "sym2")):
        reg = [base[f"{c}|{fam}|0"]["mean"] for c in core.CONFIGS]
        nai = [nv[f"{c}|naive|{fam}|0"]["mean"] for c in core.CONFIGS]
        ax.bar(x + (j - 0.5) * 0.38 - 0.09, reg, 0.18, color="0.35", edgecolor="0", linewidth=0.5,
               hatch=["", "////"][j], label=f"{TXT['aware']}, σ = {fam[-1]} q")
        ax.bar(x + (j - 0.5) * 0.38 + 0.09, nai, 0.18, color="white", edgecolor="0", linewidth=0.5,
               hatch=["", "////"][j], label=f"{TXT['naive']}, σ = {fam[-1]} q")
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "A staged", "B", "B staged"])
    ax.set_ylabel(TXT["value"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize=7, handletextpad=0.4)
    fig.tight_layout()
    _save(fig, "fig_naive")


def fig_size():
    """Supplementary figure: the campus-size panel of fig_naive_and_size, alone."""
    if PAPER:
        return _p_size()
    base = _j("data/p1c_summary.json")["table"]
    s0, s3 = _p1f("size075"), _p1f("size300")
    fig, ax = plt.subplots(figsize=(W1 + 12 * MM, 70 * MM))
    for i, c in enumerate(core.CONFIGS):
        pts = [(75, s0[f"{c}|size075|sym2|0"]["mean"]), (150, base[f"{c}|sym2|0"]["mean"]),
               (300, s3[f"{c}|size300|sym2|0"]["mean"])]
        ax.plot([p[0] for p in pts], [100 * p[1] / p[0] for p in pts], marker=MARK[i],
                linestyle=["-", "--", "-.", ":"][i], color="0", markerfacecolor="white", label=CFG_LABEL[c])
    ax.axhline(0, color="0", linewidth=0.5)
    ax.set_xticks((75, 150, 300))
    ax.set_xlabel("Campus peak IT load (MW)")
    ax.set_ylabel("$M per 100 MW of IT load (σ = 2 q)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=7,
              handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig_size")


# ==== manuscript versions (--paper): exact canvases in mm and the colour design system above =========================
def _p_main_values():
    """Fig. 3, 137 x 116 mm: family bars (colour + hatch), dispersion triangles in ink, the 0.5 % reference dashed."""
    T = _j("data/p1c_summary.json")["table"]
    W, H = 137, 116
    fig = _fig_mm(W, H)
    x = np.arange(len(FAMS))
    for k, (cfg, rect) in enumerate(zip(core.CONFIGS, [(16, 10, 48, 34), (85, 10, 48, 34), (16, 59, 48, 34),
                                                         (85, 59, 48, 34)])):
        ax = _ax_mm(fig, W, H, *rect)
        for i, f in enumerate(FAMS):
            col, hatch, _, _ = FAM_P[f]
            ax.bar(x[i], T[f"{cfg}|{f}|0"]["mean"], 0.62, color=col, edgecolor=PC["ink"], linewidth=0.55, hatch=hatch,
                   zorder=1)
        _refline(ax, _ref(cfg))
        ax.axhline(0, color=PC["ink"], linewidth=0.5, zorder=2)
        lo = [T[f"{cfg}|{f}|0"]["by_level"]["low"] for f in FAMS]
        hi = [T[f"{cfg}|{f}|0"]["by_level"]["high"] for f in FAMS]
        ax.scatter(x - 0.14, lo, marker="v", s=4.2 ** 2, facecolor="white", edgecolor=PC["ink"], linewidth=0.8,
                   zorder=3, label="low dispersion")         # the two levels sit side by side within the bar
        ax.scatter(x + 0.14, hi, marker="^", s=4.2 ** 2, facecolor=PC["ink"], edgecolor=PC["ink"], linewidth=0.8,
                   zorder=3, label="high dispersion")
        _clear(ax, lo + hi + [T[f"{cfg}|{f}|0"]["mean"] for f in FAMS] + [0.0, _ref(cfg)], 34)
        ax.set_title(CFG_LABEL[cfg])
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_SHORT[f] for f in FAMS])
        if k % 2 == 0:
            ax.set_ylabel(TXT["value"])
    h, l = ax.get_legend_handles_labels()
    _legend_mm(fig, W, H, h + [_refhandle()], l + [TXT["ref"]], 68.5, 109, ncol=3)
    _save_exact(fig, "fig_main_values")


def _p_policy_illustration():
    """Fig. 4, 137 x 80 mm: no forecast (ink) and the early, middle and late categories (blue, teal, vermilion)."""
    run = ROOT / "experiments" / "r2_runs" / "p1c_main"
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    cells = [c for c in plan["cells"] if c["cfg"] == "B" and c["level"] == "high"]
    c = max(cells, key=lambda x: x["p"])
    load = lambda k: json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    ns, sg = load(c["jobs"]["nosig"]), load(c["jobs"]["sig"])
    W, H = 137, 80
    fig = _fig_mm(W, H)
    ax, ax2 = _ax_mm(fig, W, H, 17, 10, 47, 43), _ax_mm(fig, W, H, 86, 10, 47, 43)
    styles = [(TXT["none"], ns["detail"]["orders"]["GE"], "-", "o", PC["ink"])]
    for y, (cat, col, ls, mk) in enumerate((("early", PC["blue"], DASH, "s"), ("middle", PC["teal"], DASHDOT, "^"),
                                            ("late", PC["verm"], DOT, "D"))):
        styles.append((TXT["cat"] + cat, sg["detail"][y]["orders"]["GE"], ls, mk, col))
    nodes = np.arange(len(ns["detail"]["orders"]["GE"]))
    shown = nodes <= 20
    for lab, orders, ls, mk, col in styles:
        ax.plot(nodes[shown], np.cumsum(orders)[shown], drawstyle="steps-post", linestyle=ls, color=col, marker=mk,
                markevery=4, markerfacecolor="white", markeredgecolor=col, label=lab)
    ax.axvline(core.TAU, color=PC["ink"], linewidth=0.5)
    ax.text(core.TAU + 0.4, 80, TXT["arrives"], fontsize=7.5, va="center", ha="left")   # empty region below the plans
    ax.set_xlabel("Planning node (quarter)")
    ax.set_ylabel("Cumulative gas-CHP\norders (MW)")
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(range(0, 21, 4))
    P = core.params("B")
    lead, qs = P.lead["GE"], np.arange(0, 25)
    ax2.plot(qs, [(1 + P.a_oth) * P.L(q) for q in qs], drawstyle="steps-post", color=PC["grey"], linewidth=1.8,
             zorder=0, label="IT and other\nelectricity")
    for lab, orders, ls, mk, col in styles:
        cum = np.cumsum(orders)
        ax2.plot(qs, [cum[q - lead] if q - lead >= 0 else 0.0 for q in qs], drawstyle="steps-post", linestyle=ls,
                 color=col, marker=mk, markevery=4, markerfacecolor="white", markeredgecolor=col, label=lab)
    ax2.set_xlabel("Quarter")
    ax2.set_ylabel("Gas-CHP capacity\nin service (MW)")
    ax2.set_xlim(-0.6, 24.6)
    ax2.set_xticks(range(0, 25, 4))
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2[:1], l2[:1], loc="lower right", bbox_to_anchor=(1.0, 0.14), frameon=False)
    ax.set_title("(a) Orders")
    ax2.set_title("(b) In service before connection")
    h, l = ax.get_legend_handles_labels()                 # two columns: [none, early] | [middle, late]
    _legend_mm(fig, W, H, h, l, 68.5, 71, ncol=2)
    _save_exact(fig, "fig_policy_illustration")


def _impl_value(S, i, b, fam, inter):
    lab, _, pat = IMPL[i]
    T = S[lab]
    if inter:
        return T[f"{pat.format(b='A+staged')}|{fam}|0"]["mean"] - T[f"{pat.format(b='A')}|{fam}|0"]["mean"]
    return T[f"{pat.format(b=b)}|{fam}|0"]["mean"]


def _p_implementation(show, name):
    """Fig. 5 (five variants, 137 x 103 mm) and Supplementary S6 (nine, 137 x 134 mm): one colour and marker per
    implementation; rows in figure millimetres, variants top to bottom in the established order."""
    S = {lab: _j(f)["table"] for lab, f, _ in IMPL}
    n = len(show)
    if n <= 5:
        W, H, height, rows_l, rows_r, step, leg_y = 137, 103, 56, (26, 54), (18, 32, 46, 60), 2.2, 93
    else:
        W, H, height, rows_l, rows_r, step, leg_y = 137, 134, 87, (36, 78), (20, 42, 64, 86), 1.8, 121
    fig = _fig_mm(W, H)
    top = 12 if n <= 5 else 9.5
    axes = [_ax_mm(fig, W, H, 19, top, 29, height), _ax_mm(fig, W, H, 96, top, 38, height)]
    panels = [("Dominion-like (B)", [r for r in ROWS if r[1] == "B"], rows_l),
              ("ERCOT-like (A)", [r for r in ROWS if r[1] != "B"], rows_r)]
    half = (n - 1) / 2 * step + 2.0
    for ax, (title, rows, ys) in zip(axes, panels):
        ax.set_ylim(top + height, top)                   # data y = figure millimetres, downward
        xs = []
        for (rlab, b, fam, inter), yc in zip(rows, ys):
            ref = _ref("B") if b == "B" else _ref("A+staged")
            ax.plot([ref, ref], [yc - half, yc + half], color=PC["ink"], linestyle=REFDASH, linewidth=0.8, zorder=2)
            for pos, i in enumerate(show):
                try:
                    v = _impl_value(S, i, b, fam, inter)
                except KeyError:
                    continue
                col, mk, filled = IMPL_P[i]
                size = 5.0 if mk in ("P", "X", "*") else 4.5
                xs.append(v)
                ax.scatter(v, yc - (n - 1) / 2 * step + pos * step, marker=mk, s=size ** 2,
                           facecolor=col if filled else "white", edgecolor=col, linewidth=0.8, zorder=3,
                           label=IMPL[i][0])
        xs += [_ref("B") if rows[0][1] == "B" else _ref("A+staged")] + ([] if rows[0][1] == "B" else [0.0])
        _clear(ax, xs, 29 if rows[0][1] == "B" else 38, axis="x")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0].replace("ERCOT-like staged", "staged") for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.set_title(title)
        ax.set_xlabel(TXT["value"])
        if rows[0][1] != "B":
            ax.axvline(0, color=PC["ink"], linewidth=0.5, zorder=1)
    seen = {}
    for hh, ll in zip(*axes[1].get_legend_handles_labels()):
        seen.setdefault(ll, hh)
    handles, labels = list(seen.values()) + [_refhandle()], list(seen.keys()) + [TXT["ref"]]
    _legend_mm(fig, W, H, handles, labels, 68.5, leg_y, ncol=2)
    _save_exact(fig, name)


def _p_tornado():
    """Fig. 6, 137 x 134 mm: lower tested value sky with a hatch, higher tested value blue; main case solid ink."""
    base = _j("data/p1c_summary.json")["table"]
    W, H = 137, 134
    fig = _fig_mm(W, H)
    for rect, (c, fam, title) in zip([(50, 12, 84, 40), (50, 74, 84, 40)],
                                     (("B", "sym2", "Dominion-like (B), σ = 2 q"),
                                      ("A+staged", "sym2", "ERCOT-like, staged, σ = 2 q"))):
        ax = _ax_mm(fig, W, H, *rect)
        b0 = base[f"{c}|{fam}|0"]["mean"]
        rows = []
        for v in ES_ORDER:
            T = _p1f(v)
            if T is None:
                continue
            lo, hi = ES_LEVELS[v]
            rows.append((ES_LABEL[v][c[0]], T[f"{c}|{lo}|{fam}|0"]["mean"], T[f"{c}|{hi}|{fam}|0"]["mean"], lo == hi))
        rows.sort(key=lambda r: max(abs(r[1] - b0), abs(r[2] - b0)))
        for y, (lab, a, b, single) in enumerate(rows):
            if single:
                ax.barh(y, b - b0, left=b0, height=0.6, color=PC["blue"], edgecolor=PC["ink"], linewidth=0.55)
                continue
            ax.barh(y + 0.16, a - b0, left=b0, height=0.32, color=PC["sky"], edgecolor=PC["ink"], linewidth=0.55,
                    hatch="//")
            ax.barh(y - 0.16, b - b0, left=b0, height=0.32, color=PC["blue"], edgecolor=PC["ink"], linewidth=0.55)
        ax.axvline(b0, color=PC["ink"], linewidth=1.0, zorder=3)
        _refline(ax, _ref(c), vertical=True)
        xs = [b0, _ref(c)] + [r[1] for r in rows] + [r[2] for r in rows]
        pad = 0.06 * (max(xs) - min(xs))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(TXT["value"])
        ax.set_title(title)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=PC["sky"], edgecolor=PC["ink"], linewidth=0.55, hatch="//"),
         plt.Rectangle((0, 0), 1, 1, facecolor=PC["blue"], edgecolor=PC["ink"], linewidth=0.55),
         plt.Line2D([], [], color=PC["ink"], linewidth=1.0), _refhandle()]
    _legend_mm(fig, W, H, h, ["lower tested value", "higher tested value", "main case", TXT["ref"]], 68.5, 128, ncol=2)
    _save_exact(fig, "fig_tornado")


def _p_naive():
    """Fig. 7, 101 x 83 mm: filled bars forecast-aware, white bars face value; plain sigma = 1, hatched sigma = 2."""
    base = _j("data/p1c_summary.json")["table"]
    nv = _p1f("naive")
    W, H = 101, 83
    fig = _fig_mm(W, H)
    ax = _ax_mm(fig, W, H, 15, 6, 83, 44)
    x = np.arange(len(core.CONFIGS))
    for j, (fam, col, hatch) in enumerate((("sym1", PC["blue"], ""), ("sym2", PC["sky"], "//"))):
        reg = [base[f"{c}|{fam}|0"]["mean"] for c in core.CONFIGS]
        nai = [nv[f"{c}|naive|{fam}|0"]["mean"] for c in core.CONFIGS]
        ax.bar(x + (j - 0.5) * 0.38 - 0.09, reg, 0.18, color=col, edgecolor=PC["ink"], linewidth=0.55, hatch=hatch,
               label=f"{TXT['aware']}, σ = {fam[-1]} q")
        ax.bar(x + (j - 0.5) * 0.38 + 0.09, nai, 0.18, color="white", edgecolor=col, linewidth=0.8, hatch=hatch,
               hatchcolor=PC["ink"], label=f"{TXT['naive']}, σ = {fam[-1]} q")
    ax.axhline(0, color=PC["ink"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "A staged", "B", "B staged"])
    ax.set_ylabel(TXT["value"])
    h, l = ax.get_legend_handles_labels()
    _legend_mm(fig, W, H, h, l, 50.5, 70.5, ncol=1)
    _save_exact(fig, "fig_naive")


def _p_value_vs_tau():
    """Fig. 8, 137 x 116 mm: one colour, line style and open marker per forecast-error family."""
    base = _j("data/p1c_summary.json")["table"]
    t2, t6 = _p1f("tau2"), _p1f("tau6")
    W, H = 137, 116
    fig = _fig_mm(W, H)
    taus = (2, 4, 6)
    for k, (c, rect) in enumerate(zip(core.CONFIGS, [(16, 10, 48, 31), (85, 10, 48, 31), (16, 58, 48, 31),
                                                       (85, 58, 48, 31)])):
        ax = _ax_mm(fig, W, H, *rect)
        for fam in FAMS:
            ys = [t2[f"{c}|tau2|{fam}|0"]["mean"], base[f"{c}|{fam}|0"]["mean"], t6[f"{c}|tau6|{fam}|0"]["mean"]]
            col, _, ls, mk = FAM_P[fam]
            ax.plot(taus, ys, linestyle=ls, marker=mk, color=col, markerfacecolor="white", markeredgecolor=col,
                    label=FAM_LABEL[fam])
        ax.axhline(0, color=PC["ink"], linewidth=0.5)
        ax.set_xticks(taus)
        ax.set_xlim(1.5, 6.5)
        ax.set_title(CFG_LABEL[c])
        if k % 2 == 0:
            ax.set_ylabel(TXT["value"])
    fig.text(68.5 / W, 1 - 99 / H, TXT["tau_axis"], ha="center", va="center", fontsize=8)
    h, l = ax.get_legend_handles_labels()               # rows: sigma = 1, 2, 4 | biased, uninformative
    order = [0, 3, 1, 4, 2]
    _legend_mm(fig, W, H, [h[i] for i in order], [l[i] for i in order], 68.5, 110, ncol=3)
    _save_exact(fig, "fig_value_vs_tau")


def _p_staging_2x2():
    """Supplement, 137 x 74 mm: import cap blue, backup delay amber with a hatch, interaction purple with dots."""
    p1c, stage = _j("data/p1c_summary.json")["table"], _j("data/p1d/stage_summary.json")["table"]
    spine, e4a = _j("data/p1d/spine_summary.json")["table"], _j("data/p1d/e4a_summary.json")["table"]
    e2, e4a2 = _j("data/p1d/e2_summary.json")["table"], _j("data/p1d/e4a2_summary.json")["table"]
    models = [lambda f: _twobytwo(f, p1c, "A", "A+staged", stage, "A|cap", "A|bkdelay"),
              lambda f: _twobytwo(f, spine, "A|spine", "A+staged|spine", e4a, "A|cap|spine", "A|bkdelay|spine"),
              lambda f: _twobytwo(f, e2, "A|batch", "A+staged|batch", e4a2, "A|cap|batch", "A|bkdelay|batch")]
    W, H = 137, 74
    fig = _fig_mm(W, H)
    ax0 = _ax_mm(fig, W, H, 18, 9, 51, 41)
    ax1 = _ax_mm(fig, W, H, 82, 9, 51, 41, sharey=ax0)
    parts = [("import cap", PC["blue"], ""), ("backup delay", PC["amber"], "//"), ("interaction", PC["purple"], "..")]
    for ax, fam in zip((ax0, ax1), ("sym1", "sym2")):
        x = np.arange(len(models))
        vals = np.array([m(fam) for m in models])
        for j, (lab, col, hatch) in enumerate(parts):
            ax.bar(x + (j - 1) * 0.25, vals[:, j], 0.25, color=col, edgecolor=PC["ink"], linewidth=0.55, hatch=hatch,
                   label=lab)
            for xi, v in zip(x, vals[:, j]):
                if abs(v) < 0.005:
                    ax.text(xi + (j - 1) * 0.25, 0.08, "0", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0, color=PC["ink"], linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["zero-lead\nbenchmark", "lead-\nconsistent", "whole units\n(batched)"])
        ax.set_title(f"ERCOT-like, σ = {fam[-1]} q")
    ax1.tick_params(labelleft=False)
    ax0.set_ylabel("Effect on the value of\nthe forecast ($M)")
    h, l = ax0.get_legend_handles_labels()
    _legend_mm(fig, W, H, h, l, 68.5, 68, ncol=3)
    _save_exact(fig, "fig_staging_2x2")


def _p_size():
    """Supplement, 101 x 76 mm: the four configurations (blue ERCOT-like, vermilion Dominion-like)."""
    base = _j("data/p1c_summary.json")["table"]
    s0, s3 = _p1f("size075"), _p1f("size300")
    W, H = 101, 76
    fig = _fig_mm(W, H)
    ax = _ax_mm(fig, W, H, 18, 7, 80, 44)
    for c in core.CONFIGS:
        pts = [(75, s0[f"{c}|size075|sym2|0"]["mean"]), (150, base[f"{c}|sym2|0"]["mean"]),
               (300, s3[f"{c}|size300|sym2|0"]["mean"])]
        col, mk, ls = CFG_P[c]
        ax.plot([p[0] for p in pts], [100 * p[1] / p[0] for p in pts], marker=mk, linestyle=ls, color=col,
                markerfacecolor="white", markeredgecolor=col, label=CFG_LABEL[c])
    ax.axhline(0, color=PC["ink"], linewidth=0.5)
    ax.set_xticks((75, 150, 300))
    ax.set_xlabel("Campus peak IT load (MW)")
    ax.set_ylabel("$M per 100 MW\nof IT load (σ = 2 q)")
    h, l = ax.get_legend_handles_labels()
    _legend_mm(fig, W, H, h, l, 50.5, 68, ncol=2)
    _save_exact(fig, "fig_size")


def _p_oracle_decomp():
    """Supplement, 88 x 82 mm: V_true blue, G_0 sky with a hatch, -G_Y amber; the total value an ink diamond."""
    D = _j("data/p1d/oracle_decomposition.json")["decomposition"]
    keys = [("B", "sym1"), ("B", "sym2"), ("A+staged", "sym1"), ("A+staged", "sym2")]
    W, H = 88, 82
    fig = _fig_mm(W, H)
    ax = _ax_mm(fig, W, H, 15, 5, 70, 44)
    x = np.arange(len(keys))
    vt = [D[f"{c}|{f}"]["mean"]["V_true"] for c, f in keys]
    g0 = [D[f"{c}|{f}"]["mean"]["G0"] for c, f in keys]
    ry = [D[f"{c}|{f}"]["mean"]["R_Y"] for c, f in keys]
    di = [D[f"{c}|{f}"]["mean"]["d_info"] for c, f in keys]
    ax.bar(x - 0.25, vt, 0.25, color=PC["blue"], edgecolor=PC["ink"], linewidth=0.55, label="V_true")
    ax.bar(x, g0, 0.25, color=PC["sky"], edgecolor=PC["ink"], linewidth=0.55, hatch="//", label="G_0")
    ax.bar(x + 0.25, [-r for r in ry], 0.25, color=PC["amber"], edgecolor=PC["ink"], linewidth=0.55, label="−G_Y")
    ax.scatter(x, di, marker="D", s=4.5 ** 2, facecolor=PC["ink"], edgecolor=PC["ink"], zorder=3,
               label="value = V_true + G_0 − G_Y")
    ax.axhline(0, color=PC["ink"], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["B\nσ = 1 q", "B\nσ = 2 q", "A staged\nσ = 1 q", "A staged\nσ = 2 q"])
    ax.set_ylabel("$M (expected, Δ = 0)")
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]                                # the bars in plotting order, then the total
    _legend_mm(fig, W, H, [h[i] for i in order], [l[i] for i in order], 44, 69.5, ncol=1)
    _save_exact(fig, "fig_oracle_decomp")


def _p_co2_grid():
    """Supplement, 88 x 102 mm: configuration colour and marker, implementation line style; eGRID points filled."""
    em = _j("data/p1e/emissions_summary.json")["table"]
    fs = (0.2, 0.4, 0.6, 0.8)
    W, H = 88, 102
    fig = _fig_mm(W, H)
    ax = _ax_mm(fig, W, H, 17, 6, 68, 49)
    for b, impl, col, mk in (("A+staged", "continuous", PC["blue"], "s"), ("A+staged", "batched", PC["blue"], "s"),
                             ("B", "continuous", PC["verm"], "^"), ("B", "batched", PC["verm"], "^")):
        y = [em[f"{b}|sym2|{impl}|sweep {f}"]["delta_co2_kt"] for f in fs]
        ax.plot(fs, y, linestyle="-" if impl == "continuous" else DASH, marker=mk, color=col, markerfacecolor="white",
                markeredgecolor=col, label=f"{'Dominion-like' if b == 'B' else 'ERCOT-like staged'}, {impl}")
    for b, name, mk, col, lab in (("A+staged", "eGRID2023 ERCT total", "^", PC["blue"], "eGRID2023 ERCT (ERCOT-like)"),
                                  ("B", "eGRID2023 SRVC total", "v", PC["verm"], "eGRID2023 SRVC (Dominion-like)")):
        ef = 733.9 * 0.000453592 if b == "A+staged" else 593.4 * 0.000453592
        ax.scatter([ef], [em[f"{b}|sym2|continuous|{name}"]["delta_co2_kt"]], marker=mk, s=5.0 ** 2, facecolor=col,
                   edgecolor=PC["ink"], linewidth=0.6, zorder=4, label=lab)
    ax.axhline(0, color=PC["ink"], linewidth=0.5)
    ax.set_xlabel("Grid emission factor (t CO2/MWh)")
    ax.set_ylabel("ΔCO2 (kt), forecast-aware − no forecast")
    h, l = ax.get_legend_handles_labels()
    _legend_mm(fig, W, H, h, l, 44, 84.5, ncol=1)
    _save_exact(fig, "fig_co2_grid")


def _p_tpit_errors():
    """Supplement, 88 x 86 mm: TPIT horizons blue, teal and vermilion; txUS purple; the declared law sky, drawn first."""
    tp = _j("data/p1e/tpit_errors.json")["summary"]
    tx = _j("data/p1e/txus_summary.json")["family"]["pmf"]
    e = np.arange(-16, 17)
    vis = (e >= -12) & (e <= 6)
    W, H = 88, 86
    fig = _fig_mm(W, H)
    ax = _ax_mm(fig, W, H, 15, 6, 70, 43)
    ax.plot(e[vis], np.asarray(core.error_pmf(2.0))[vis], linestyle="-", color=PC["sky"], linewidth=1.8, zorder=1,
            label="declared σ = 2 q")
    for h, col, ls, mk in (("h4|all", PC["blue"], "-", "o"), ("h8|all", PC["teal"], DASH, "s"),
                           ("h12|all", PC["verm"], DOT, "^")):
        pm = tp[h]["pmf"]
        y = np.array([pm.get(str(k), 0.0) for k in e])
        ax.plot(e[vis], y[vis], linestyle=ls, marker=mk, markersize=3.5, color=col, markerfacecolor="white",
                markeredgecolor=col, zorder=3, clip_on=False,
                label=f"TPIT, {h.split('|')[0][1:]} q ahead (n = {tp[h]['n']})")
    ax.plot(e[vis], np.array([tx.get(str(k), 0.0) for k in e])[vis], linestyle=DASHDOT, color=PC["purple"], zorder=2,
            label="txUS proxy")
    ax.set_xlim(-12, 6)
    ax.set_xticks(range(-12, 7, 2))
    ax.set_xlabel("Projected − actual (quarters; < 0: late)")
    ax.set_ylabel("Probability")
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]                             # empirical horizons first, then the two comparison laws
    _legend_mm(fig, W, H, [h[i] for i in order], [l[i] for i in order], 44, 72, ncol=1)
    _save_exact(fig, "fig_tpit_errors")


ALL = dict(fig_main_values=fig_main_values, fig_implementation=fig_implementation_main,
           fig_implementation_all=fig_implementation_all, fig_naive=fig_naive, fig_size=fig_size,
           fig_oracle_decomp=fig_oracle_decomp,
           fig_staging_2x2=fig_staging_2x2, fig_policy_illustration=fig_policy_illustration,
           fig_tpit_errors=fig_tpit_errors, fig_co2_grid=fig_co2_grid, fig_value_vs_tau=fig_value_vs_tau,
           fig_tornado=fig_tornado, fig_naive_and_size=fig_naive_and_size)

PAPER_SET = ("fig_main_values", "fig_policy_illustration", "fig_implementation", "fig_tornado", "fig_value_vs_tau",
             "fig_naive")                  # the manuscript's Figs. 3-8; the others are supplementary
SUPP_SET = ("fig_implementation_all", "fig_size", "fig_staging_2x2", "fig_oracle_decomp", "fig_co2_grid",
            "fig_tpit_errors")

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    default = (PAPER_SET + SUPP_SET) if PAPER else [k for k in ALL if k not in ("fig_implementation_all", "fig_naive",
                                                                                    "fig_size")]
    for name in (names or default):
        ALL[name]()
