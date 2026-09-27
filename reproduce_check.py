"""Regenerate the released results from the stored runs and compare them with the released files.

    python reproduce_check.py            (after unzipping runs.zip into experiments/)

Runs the analyses and report generators, then compares every file in data/ with its released version (line endings
normalized) and the paper figures with a reference directory when one is given (REF_FIGS). Prints one line per
changed file; exit code 0 when nothing changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "data").rglob("*") if p.is_file()}


def run(args):
    r = subprocess.run([PY] + args, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8")
    print(f"{'ok ' if r.returncode == 0 else 'ERR'} {' '.join(args)}")
    if r.returncode:
        print(r.stdout[-800:], r.stderr[-1500:])
    return r.returncode


def main():
    before = snapshot()
    rc = 0
    rc |= run(["experiments/analyze_p1c.py", "--run", "p1c_main"])
    rc |= run(["experiments/make_p1c_report.py", "--run", "p1c_main"])
    manifest = json.loads((ROOT / "data" / "run_manifest.json").read_text(encoding="utf-8"))
    runs = manifest["runs"] if isinstance(manifest, dict) and "runs" in manifest else manifest
    for r in runs:
        if r.get("harness_kind") == "p1d" and r.get("variant") and (ROOT / "experiments" / "r2_runs" / r["run"]).exists():
            rc |= run(["experiments/analyze_p1d.py", "--variant", r["variant"], "--run", r["run"]])
    for script in ("make_p1d_report.py", "make_p1e_report.py", "make_p1f_report.py", "make_p1g_report.py"):
        rc |= run([f"experiments/{script}"])
    rc |= run(["experiments/make_figures.py", "--paper"])
    after = snapshot()
    changed = sorted(k for k in after if before.get(k) != after[k])
    added = sorted(k for k in after if k not in before)
    print(f"data files compared {len(before)}; changed {len(changed)}; new {len(added)}")
    for k in changed:
        print("  changed", k)
    for k in added:
        print("  new", k)
    ref = os.environ.get("REF_FIGS")
    if ref:
        from PIL import Image, ImageChops
        diff = []
        for p in sorted((ROOT / "paper" / "figs").glob("*.png")):
            q = pathlib.Path(ref) / p.name
            if not q.exists():
                diff.append(f"{p.name}: no reference")
                continue
            a, b = Image.open(p).convert("RGB"), Image.open(q).convert("RGB")
            if a.size != b.size or ImageChops.difference(a, b).getbbox() is not None:
                diff.append(p.name)
        print(f"figures compared {len(list((ROOT / 'paper' / 'figs').glob('*.png')))}; differing: {diff}")
    sys.exit(1 if (rc or changed) else 0)


if __name__ == "__main__":
    main()
