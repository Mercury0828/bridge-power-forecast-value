"""Energy Systems versions of the result figures (layout and labels only; the data are the frozen summaries).

    .venv/Scripts/python.exe experiments/make_figures_es.py   -> paper_es/figs/<name>.pdf and .png

Same drawing code and design system as the *Energy* manuscript figures (experiments/make_figures.py --paper), with
two changes: the canvases are 119 mm wide (the printed width of an Energy Systems page) instead of 137 mm, so that
every label keeps its point size at the printed width; and the labels use the paper's reader terms (long wait,
short wait, partial supply, value of using the estimate). No number is computed or typed here.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "--paper"]                       # the manuscript design system of make_figures.py
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import make_figures as mf  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402
from experiments import p1c_core as core  # noqa: E402

PRINT_W = 119.0
# MF_NOHATCH=1 MF_OUT=<dir>: the lint copies without hatch tiles (see make_figures.py); otherwise the paper figures
mf.FIG = pathlib.Path(mf.os.environ["MF_OUT"]) if mf.os.environ.get("MF_NOHATCH") == "1" else ROOT / "paper_es" / "figs"
_orig_fig_mm = mf._fig_mm


def _fig_es(w, h):
    """Scale the canvas width to the printed width; axes and legends are placed in fractions, fonts stay in points."""
    return _orig_fig_mm(min(w, PRINT_W), h)


mf._fig_mm = _fig_es
_orig_save_exact = mf._save_exact


def _clipped_hatches(fig):
    """Replace each hatch tile pattern by the same strokes and dots, cut to the bar. A tile extends far outside the bar
    and is only clipped by the viewer, so tools that read the page see it crossing margins and labels. The geometry is
    computed while drawing, because legend boxes move with the text metrics of each output backend."""
    items = [(p, ax) for ax in fig.axes for p in ax.patches]
    items += [(p, None) for leg in fig.legends for p in leg.get_patches()]
    for patch, ax in items:
        hatch = patch.get_hatch()
        if hatch:
            colour = patch.get_hatchcolor()
            patch.set_hatch(None)
            (fig if ax is None else ax).add_artist(_ClippedHatch(patch, ax, hatch, colour))


class _ClippedHatch(Artist):
    def __init__(self, patch, ax, hatch, colour):
        super().__init__()
        self._patch, self._ax, self._pattern, self._colour = patch, ax, hatch, colour
        self.set_zorder(10 if ax is None else patch.get_zorder() + 0.1)

    def get_window_extent(self, renderer=None):
        return self._patch.get_window_extent()

    def draw(self, renderer):
        fig, hatch = self.get_figure(root=True), self._pattern
        per_pt = fig.dpi / 72.0
        box = self._patch.get_window_extent()
        x0, x1 = sorted((box.x0, box.x1))
        y0, y1 = sorted((box.y0, box.y1))
        if self._ax is not None:
            x0, x1 = max(x0, self._ax.bbox.x0), min(x1, self._ax.bbox.x1)
            y0, y1 = max(y0, self._ax.bbox.y0), min(y1, self._ax.bbox.y1)
        inset = 0.25 * per_pt                             # stay inside the outline
        x0, x1, y0, y1 = x0 + inset, x1 - inset, y0 + inset, y1 - inset
        if x1 <= x0 or y1 <= y0:
            return
        shapes = []
        strokes = []
        for sign, n in ((1, hatch.count("/") + hatch.count("x")), (-1, hatch.count("x"))):
            if not n:
                continue
            step = 12.0 / n * per_pt                      # matplotlib's spacing: 6 n strokes per 72 pt
            # the strokes are y = sign * (x - c) with c on one grid for the whole page
            ends = (x0 - sign * y0, x0 - sign * y1, x1 - sign * y0, x1 - sign * y1)
            for c in np.arange(np.floor(min(ends) / step), np.ceil(max(ends) / step) + 1) * step:
                lo, hi = sorted((c + sign * y0, c + sign * y1))
                a, b = max(lo, x0), min(hi, x1)
                if b - a > 1e-6:
                    strokes.append([(a, sign * (a - c)), (b, sign * (b - c))])
        if strokes:
            shapes.append(LineCollection(strokes, colors=[self._colour],
                                         linewidths=plt.rcParams["hatch.linewidth"], transform=IdentityTransform()))
        if "." in hatch:
            step = 12.0 / hatch.count(".") * per_pt
            r = 0.1 * step
            turn = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ring = r * np.c_[np.cos(turn), np.sin(turn)]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2         # centred, so that a small legend box keeps its dots
            rows, cols = int((y1 - y0) / step) + 1, int((x1 - x0) / step) + 1
            dots = []
            for row in range(-rows, rows + 1):
                cy = my + row * step
                for col in range(-cols, cols + 1):
                    cx = mx + (col + 0.5 * (row % 2)) * step
                    if x0 + r <= cx <= x1 - r and y0 + r <= cy <= y1 - r:
                        dots.append(ring + (cx, cy))
            if dots:
                shapes.append(PolyCollection(dots, facecolors=[self._colour], edgecolors="none",
                                             transform=IdentityTransform()))
        for shape in shapes:
            shape.set_figure(fig)
            shape.draw(renderer)
        self.stale = False


def _save_clipped(fig, name):
    _clipped_hatches(fig)
    _orig_save_exact(fig, name)


mf._save_exact = _save_clipped
mf.CFG_LABEL.update({"A": "Short wait", "A+staged": "Short wait, partial supply", "B": "Long wait",
                     "B+staged": "Long wait, partial supply"})
mf.TXT.update(value="Value of using the estimate ($M)", arrives="estimate\narrives", none="no estimate",
              cat="estimate: ", tau_axis="Quarter in which the estimate arrives, τ", aware="forecast-aware",
              naive="face-value planner", ref="0.5% of cost without an estimate")
mf.FAM_SHORT.update({"sym1": "σ = 1", "sym2": "σ = 2", "sym4": "σ = 4", "opt2": "biased", "uninf": "uninformative"})
mf.FAM_LABEL.update({"sym1": "σ = 1 quarter", "sym2": "σ = 2 quarters", "sym4": "σ = 4 quarters", "opt2": "biased (−2 quarters)", "uninf": "uninformative"})
mf.TXT["aware"] = "forecast-aware plan"


def main_values():
    """make_figures._p_main_values at 119 mm; the families are named in the legend (bar fill and hatch)."""
    T = mf._j("data/p1c_summary.json")["table"]
    W, H = 119, 122
    fig = mf._fig_mm(W, H)
    x = np.arange(len(mf.FAMS))
    for k, (cfg, rect) in enumerate(zip(core.CONFIGS, [(15, 8, 42, 34), (74, 8, 42, 34), (15, 54, 42, 34),
                                                         (74, 54, 42, 34)])):
        ax = mf._ax_mm(fig, W, H, *rect)
        for i, f in enumerate(mf.FAMS):
            col, hatch, _, _ = mf.FAM_P[f]
            ax.bar(x[i], T[f"{cfg}|{f}|0"]["mean"], 0.62, color=col, edgecolor=mf.PC["ink"], linewidth=0.55,
                   hatch=hatch, zorder=1)
        mf._refline(ax, mf._ref(cfg))
        ax.axhline(0, color=mf.PC["ink"], linewidth=0.5, zorder=2)
        lo = [T[f"{cfg}|{f}|0"]["by_level"]["low"] for f in mf.FAMS]
        hi = [T[f"{cfg}|{f}|0"]["by_level"]["high"] for f in mf.FAMS]
        ax.scatter(x - 0.14, lo, marker="v", s=4.2 ** 2, facecolor="white", edgecolor=mf.PC["ink"],
                   linewidth=0.8, zorder=3, label="low dispersion")
        ax.scatter(x + 0.14, hi, marker="^", s=4.2 ** 2, facecolor=mf.PC["ink"], edgecolor=mf.PC["ink"],
                   linewidth=0.8, zorder=3, label="high dispersion")
        mf._clear(ax, lo + hi + [T[f"{cfg}|{f}|0"]["mean"] for f in mf.FAMS] + [0.0, mf._ref(cfg)], 34)
        ax.set_title(mf.CFG_LABEL[cfg])
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    fig.text(3.5 / W, 1 - 48 / H, mf.TXT["value"], rotation=90, ha="center", va="center", fontsize=8)
    h, l = ax.get_legend_handles_labels()
    fam_h = [plt.Rectangle((0, 0), 1, 1, facecolor=mf.FAM_P[f][0], edgecolor=mf.PC["ink"], linewidth=0.55,
                           hatch=mf.FAM_P[f][1]) for f in mf.FAMS]
    fam_l = [mf.FAM_LABEL[f] for f in mf.FAMS]
    mf._legend_mm(fig, W, H, fam_h + h + [mf._refhandle()], fam_l + l + [mf.TXT["ref"]], 59.5, 105, ncol=2)
    mf._save_exact(fig, "fig_main_values")



def policy_illustration():
    """The running example (representative long-wait cell): make_figures._p_policy_illustration at 119 mm."""
    run = ROOT / "experiments" / "r2_runs" / "p1c_main"
    plan = mf.json.loads((run / "plan.json").read_text(encoding="utf-8"))
    cells = [c for c in plan["cells"] if c["cfg"] == "B" and c["level"] == "high"]
    c = max(cells, key=lambda x: x["p"])
    load = lambda k: mf.json.loads((run / "items" / f"J_{k}.json").read_text(encoding="utf-8"))  # noqa: E731
    ns, sg = load(c["jobs"]["nosig"]), load(c["jobs"]["sig"])
    W, H = 119, 80
    fig = mf._fig_mm(W, H)
    ax, ax2 = mf._ax_mm(fig, W, H, 17, 10, 38, 43), mf._ax_mm(fig, W, H, 77, 10, 38, 43)
    styles = [(mf.TXT["none"], ns["detail"]["orders"]["GE"], "-", "o", mf.PC["ink"])]
    for y, (cat, col, ls, mk) in enumerate((("early", mf.PC["blue"], mf.DASH, "s"),
                                            ("middle", mf.PC["teal"], mf.DASHDOT, "^"),
                                            ("late", mf.PC["verm"], mf.DOT, "D"))):
        styles.append((mf.TXT["cat"] + cat, sg["detail"][y]["orders"]["GE"], ls, mk, col))
    nodes = np.arange(len(ns["detail"]["orders"]["GE"]))
    shown = nodes <= 20
    for lab, orders, ls, mk, col in styles:
        ax.plot(nodes[shown], np.cumsum(orders)[shown], drawstyle="steps-post", linestyle=ls, color=col, marker=mk,
                markevery=4, markerfacecolor="white", markeredgecolor=col, label=lab)
    ax.axvline(core.TAU, color=mf.PC["ink"], linewidth=0.5)
    ax.text(core.TAU + 0.4, 80, mf.TXT["arrives"], fontsize=7.5, va="center", ha="left")
    ax.set_xlabel("Planning node (quarter)")
    ax.set_ylabel("Cumulative gas-CHP\norders (MW)")
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(range(0, 21, 4))
    P = core.params("B")
    lead, qs = P.lead["GE"], np.arange(0, 25)
    ax2.plot(qs, [(1 + P.a_oth) * P.L(q) for q in qs], drawstyle="steps-post", color="#8C8C8C", linewidth=1.8,
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
    ax2.set_title("(b) In service, still waiting")
    h, l = ax.get_legend_handles_labels()
    mf._legend_mm(fig, W, H, h, l, 59.5, 71, ncol=2)
    mf._save_exact(fig, "fig_policy_illustration")


def tornado():
    base = mf._j("data/p1c_summary.json")["table"]
    W, H = 137, 134
    fig = mf._fig_mm(W, H)
    for rect, (c, fam, title) in zip([(58, 12, 76, 40), (58, 74, 76, 40)],
                                     (("B", "sym2", "(a) Long wait, σ = 2 quarters"),
                                      ("A+staged", "sym2", "(b) Short wait, partial supply, σ = 2 quarters"))):
        ax = mf._ax_mm(fig, W, H, *rect)
        b0 = base[f"{c}|{fam}|0"]["mean"]
        rows = []
        for v in mf.ES_ORDER:
            T = mf._p1f(v)
            if T is None:
                continue
            lo, hi = mf.ES_LEVELS[v]
            rows.append((mf.ES_LABEL[v][c[0]], T[f"{c}|{lo}|{fam}|0"]["mean"], T[f"{c}|{hi}|{fam}|0"]["mean"],
                         lo == hi))
        rows.sort(key=lambda r: max(abs(r[1] - b0), abs(r[2] - b0)))
        for y, (lab, a, b, single) in enumerate(rows):
            if single:
                ax.barh(y, b - b0, left=b0, height=0.6, color=mf.PC["blue"], edgecolor=mf.PC["ink"], linewidth=0.55)
                continue
            ax.barh(y + 0.16, a - b0, left=b0, height=0.32, color=mf.PC["sky"], edgecolor=mf.PC["ink"],
                    linewidth=0.55, hatch="//")
            ax.barh(y - 0.16, b - b0, left=b0, height=0.32, color=mf.PC["blue"], edgecolor=mf.PC["ink"],
                    linewidth=0.55)
        ax.axvline(b0, color=mf.PC["ink"], linewidth=1.0, zorder=3)
        mf._refline(ax, mf._ref(c), vertical=True)
        xs = [b0, mf._ref(c)] + [r[1] for r in rows] + [r[2] for r in rows]
        pad = 0.06 * (max(xs) - min(xs))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(mf.TXT["value"])
        ax.set_title(title)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=mf.PC["sky"], edgecolor=mf.PC["ink"], linewidth=0.55, hatch="//"),
         plt.Rectangle((0, 0), 1, 1, facecolor=mf.PC["blue"], edgecolor=mf.PC["ink"], linewidth=0.55),
         plt.Line2D([], [], color=mf.PC["ink"], linewidth=1.0), mf._refhandle()]
    mf._legend_mm(fig, W, H, h, ["lower tested value", "higher tested value", "main case", mf.TXT["ref"]], 68.5, 126,
                  ncol=2)
    fig.text(68.5 / W, 1 - 132.5 / H, "Parentheses: main-case values", ha="center", va="center", fontsize=7.5)
    mf._save_exact(fig, "fig_tornado")


def naive():
    base = mf._j("data/p1c_summary.json")["table"]
    nv = mf._p1f("naive")
    W, H = 119, 78
    fig = mf._fig_mm(W, H)
    ax = mf._ax_mm(fig, W, H, 16, 6, 99, 46)
    x = np.arange(len(core.CONFIGS))
    for j, (fam, col, hatch) in enumerate((("sym1", mf.PC["blue"], ""), ("sym2", mf.PC["sky"], "//"))):
        reg = [base[f"{c}|{fam}|0"]["mean"] for c in core.CONFIGS]
        nai = [nv[f"{c}|naive|{fam}|0"]["mean"] for c in core.CONFIGS]
        ax.bar(x + (j - 0.5) * 0.38 - 0.09, reg, 0.18, color=col, edgecolor=mf.PC["ink"], linewidth=0.55,
               hatch=hatch, label=f"{mf.TXT['aware']}, σ = {fam[-1]} q")
        ax.bar(x + (j - 0.5) * 0.38 + 0.09, nai, 0.18, color="white", edgecolor=col, linewidth=0.8, hatch=hatch,
               hatchcolor=mf.PC["ink"], label=f"{mf.TXT['naive']}, σ = {fam[-1]} q")
    ax.axhline(0, color=mf.PC["ink"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Short wait", "Short wait,\npartial supply", "Long wait", "Long wait,\npartial supply"])
    ax.set_ylabel(mf.TXT["value"])
    h, l = ax.get_legend_handles_labels()
    mf._legend_mm(fig, W, H, h, l, 59.5, 70, ncol=2)
    mf._save_exact(fig, "fig_naive")


def main():
    main_values()
    policy_illustration()
    tornado()
    naive()
    mf._p_value_vs_tau()


if __name__ == "__main__":
    main()
