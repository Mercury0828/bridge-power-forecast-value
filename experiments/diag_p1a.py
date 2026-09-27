"""EXPLORATORY diagnosis of the Phase-1a FAIL (owner gate D-019: method repair). Not pre-registered and not used for
any claim. Reads the frozen rows files only.

    python experiments/diag_p1a.py [econ]

For each context, dispersion level and campus shift it prints the mean of J_CSP − J_m (positive = m cheaper) for the
DRO variants, BMIX and BOX, next to the oracle room J_CSP − v.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
METHODS = ("CDRO50", "CDRO80", "BMIX", "BOX", "B1", "PSP")


def main(econ="main"):
    rows = json.loads((ROOT / "data" / f"p1a_summary_p1a_{econ}.rows.json").read_text(encoding="utf-8"))
    for z in ("A", "B"):
        for lv in ("low", "high"):
            print(f"== {econ} {z} {lv}: J_CSP − J_m ($M), oracle room = J_CSP − v")
            print("  shift | room  | " + " | ".join(f"{m:>7s}" for m in METHODS))
            for s in (-2, -1, 0, 1, 2):
                R = [r for r in rows if r["ctx"] == z and r["level"] == lv and r["shift"] == s]
                room = np.mean([r["J_CSP"] - r["oracle"] for r in R])
                vals = [np.mean([r["J_CSP"] - r[f"J_{m}"] for r in R]) for m in METHODS]
                print(f"  {s:+d}    | {room:5.2f} | " + " | ".join(f"{v:+7.2f}" for v in vals))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
