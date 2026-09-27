"""Checkpoint/resume sweep harness for R2 (guide section 7, long-run rules 1-5).

* Each sweep item writes its own JSON atomically (tmp file + os.replace). A finished item is never recomputed.
* The manifest (done/pending, config hashes) is rewritten atomically after every completed item. The previous good
  manifest is kept as manifest.last_good.json, and a corrupt latest manifest falls back to it.
* On start the harness resumes automatically; --fresh moves the old run directory aside and starts over.
* Item outputs contain no timings (timings go to <id>.timing.json), so an interrupted-and-resumed run is bit-identical
  to an uninterrupted one (V4).

Usage:
  python experiments/harness.py --run r2_main --workers 16
  python experiments/harness.py --run toy_resume --toy --workers 4        # used by tests/test_resume.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def loro_items():
    """R2b: the R2 realistic region (residence on; delay x1, x3; all rental levels), both contexts."""
    return [dict(kind="loro", ctx=ctx, p_R=pR, cap_R=cap, delay_mult=dm, residence_on=True, eps_mult=1.0, toy=False)
            for ctx, pR, cap, dm in itertools.product(["A", "B"], [15.0, 25.0, 40.0], [50.0, 100.0, 200.0], [1.0, 3.0])]


def sweep_items(toy=False):
    """Pre-registered sweep (data/r2_preregistration.md section 6) plus the base-point radius sensitivity."""
    items = []
    if toy:
        grid = itertools.product(["A", "B"], [15.0, 40.0], [30.0], [1.0, 3.0], [True, False])
    else:
        grid = itertools.product(["A", "B"], [15.0, 25.0, 40.0], [50.0, 100.0, 200.0], [1.0, 3.0, 10.0], [True, False])
    for ctx, pR, cap, dm, res in grid:
        items.append(dict(ctx=ctx, p_R=pR, cap_R=cap, delay_mult=dm, residence_on=res, eps_mult=1.0, toy=toy))
    if not toy:
        for ctx in ("A", "B"):
            for em in (0.5, 2.0):
                items.append(dict(ctx=ctx, p_R=25.0, cap_R=100.0, delay_mult=1.0, residence_on=True, eps_mult=em,
                                  toy=False, methods=["CDRO", "B4"]))
    return items


def p1a_items(econ, limit=0, only_method=None):
    """Phase 1a: one item per unique policy job (data/p1a_preregistration.md). `limit` keeps the first N jobs in key
    order and `only_method` one job kind (smoke and resume tests only)."""
    from experiments.p1a_core import plan
    cells, jobs = plan(econ)
    keys = sorted(k for k in jobs if only_method is None or jobs[k]["method"] == only_method)
    if limit:
        keys = keys[:limit]
    return [dict(kind="p1a", key=k, spec=jobs[k]) for k in keys], dict(cells=cells, jobs=jobs)


def p1c_items(limit=0, only_method=None):
    """Phase 1c: one item per unique job (data/p1c_preregistration.md). `limit` keeps the first N jobs in key order
    and `only_method` one job kind (smoke and resume tests only)."""
    from experiments.p1c_core import plan
    cells, jobs = plan()
    keys = sorted(k for k in jobs if only_method is None or jobs[k]["kind"] == only_method)
    if limit:
        keys = keys[:limit]
    return [dict(kind="p1c", key=k, spec=jobs[k]) for k in keys], dict(cells=cells, jobs=jobs)


def p1d_items(variant, limit=0):
    """Phase 1d exploratory runs (docs/phase1d_plan.md v2): one item per unique job of the variant."""
    from experiments.p1d_core import plan
    cells, jobs = plan(variant)
    keys = sorted(jobs)
    if limit:
        keys = keys[:limit]
    return [dict(kind="p1d", key=k, spec=jobs[k]) for k in keys], dict(variant=variant, cells=cells, jobs=jobs)


def item_id(it):
    if it.get("kind") in ("p1a", "p1c", "p1d"):
        return f"J_{it['key']}"
    if it.get("kind") == "loro":
        return f"L_{it['ctx']}_pR{it['p_R']:g}_cap{it['cap_R']:g}_d{it['delay_mult']:g}_res{int(it['residence_on'])}"
    tag = f"{it['ctx']}_pR{it['p_R']:g}_cap{it['cap_R']:g}_d{it['delay_mult']:g}_res{int(it['residence_on'])}_e{it['eps_mult']:g}"
    return tag


def file_sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def code_fingerprint(kind="main"):
    if kind == "p1a":
        files = [ROOT / "model" / "bridge.py", ROOT / "model" / "evidence.py", ROOT / "experiments" / "p1a_core.py"]
    elif kind == "p1c":
        files = [ROOT / "model" / "bridge.py", ROOT / "model" / "evidence.py", ROOT / "model" / "signal.py",
                 ROOT / "experiments" / "p1a_core.py", ROOT / "experiments" / "p1c_core.py"]
    elif kind == "p1d":
        files = [ROOT / "model" / "bridge.py", ROOT / "model" / "evidence.py", ROOT / "model" / "signal.py",
                 ROOT / "experiments" / "p1a_core.py", ROOT / "experiments" / "p1c_core.py",
                 ROOT / "experiments" / "p1d_core.py"]
    else:
        files = [ROOT / "model" / "bridge.py", ROOT / "experiments" / "r2_core.py", ROOT / "data" / "r1_centres.csv"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def atomic_write_json(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _worker(it):
    from experiments.r2_core import run_item, run_item_loro
    t0 = time.time()
    if it.get("kind") == "p1a":
        from experiments.p1a_core import run_job
        return item_id(it), run_job(it["spec"], time_limit=it.get("time_limit", 600.0)), round(time.time() - t0, 2)
    if it.get("kind") == "p1c":
        from experiments.p1c_core import run_job
        return item_id(it), run_job(it["spec"], time_limit=it.get("time_limit", 1800.0)), round(time.time() - t0, 2)
    if it.get("kind") == "p1d":
        from experiments.p1d_core import run_job
        return item_id(it), run_job(it["spec"], time_limit=it.get("time_limit", 1800.0)), round(time.time() - t0, 2)
    if it.get("kind") == "loro":
        res = run_item_loro(it["ctx"], it["p_R"], it["cap_R"], it["delay_mult"], it["residence_on"])
        return item_id(it), res, round(time.time() - t0, 2)
    kw = {k: v for k, v in it.items() if k in ("ctx", "p_R", "cap_R", "delay_mult", "residence_on", "eps_mult", "toy",
                                               "methods")}
    res = run_item(**kw)
    return item_id(it), res, round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r2_main")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="stop after N newly completed items (testing)")
    ap.add_argument("--kind", default="main", choices=["main", "loro", "p1a", "p1c", "p1d"])
    ap.add_argument("--variant", default=None, help="p1d variant (experiments/p1d_core.VARIANTS)")
    ap.add_argument("--econ", default="main", help="p1a economics case (experiments/p1a_core.ECON)")
    ap.add_argument("--limit-jobs", type=int, default=0, help="p1a: keep the first N jobs (smoke/resume tests)")
    ap.add_argument("--only-method", default=None, help="p1a: keep one job kind (smoke/resume tests)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds per MILP (default: 600 for p1a, 1800 for p1c)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="p1a: re-queue finished jobs whose result is not ok (pre-registered remedy: longer limit)")
    a = ap.parse_args()

    run_dir = ROOT / "experiments" / "r2_runs" / a.run
    if a.fresh and run_dir.exists():
        shutil.move(str(run_dir), str(run_dir) + f".old_{int(time.time())}")
    (run_dir / "items").mkdir(parents=True, exist_ok=True)
    fp = code_fingerprint(a.kind)
    if a.kind in ("p1a", "p1c", "p1d"):
        if a.kind == "p1a":
            items, plan_obj = p1a_items(a.econ, a.limit_jobs, a.only_method)
            stored = dict(econ=a.econ, **plan_obj)
        elif a.kind == "p1c":
            items, plan_obj = p1c_items(a.limit_jobs, a.only_method)
            stored = dict(**plan_obj)
        else:
            items, plan_obj = p1d_items(a.variant, a.limit_jobs)
            stored = dict(**plan_obj)
        tl = a.time_limit if a.time_limit is not None else (600.0 if a.kind == "p1a" else 1800.0)
        for it in items:
            it["time_limit"] = tl
        plan_path = run_dir / "plan.json"
        if not plan_path.exists():
            atomic_write_json(plan_path, stored)
        elif load_json(plan_path) != json.loads(json.dumps(stored, sort_keys=True)):
            print("WARNING: the stored plan differs from the current plan; the stored plan is kept. Use --fresh.")
    else:
        items = loro_items() if a.kind == "loro" else sweep_items(toy=a.toy)
    man_path, lastgood = run_dir / "manifest.json", run_dir / "manifest.last_good.json"
    man = load_json(man_path) or load_json(lastgood) or {}
    if man.get("fingerprint") not in (None, fp):
        print(f"WARNING: code/data fingerprint changed ({man.get('fingerprint')} -> {fp}); finished items are kept "
              f"and reported under their original fingerprint. Use --fresh to recompute everything.")
    man.setdefault("fingerprint", fp)
    man.setdefault("items", {})
    for it in items:
        iid = item_id(it)
        out = run_dir / "items" / f"{iid}.json"
        res = load_json(out) if out.exists() else None
        if a.retry_failed and a.kind in ("p1a", "p1c", "p1d"):
            # the pre-registered remedy re-runs every job that fails the analysis's usability rule, once
            if a.kind == "p1a":
                from experiments.analyze_p1a import usable
            elif a.kind == "p1c":
                from experiments.analyze_p1c import usable
            else:
                from experiments.analyze_p1d import usable
            done = res is not None and usable(it["spec"], res)
        else:
            done = res is not None
        man["items"][iid] = dict(config={k: v for k, v in it.items() if k != "spec"}, status="done" if done else "pending")
    atomic_write_json(man_path, man)
    pending = [it for it in items if man["items"][item_id(it)]["status"] != "done"]
    print(f"run {a.run}: {len(items)} items, {len(items) - len(pending)} done, {len(pending)} pending "
          f"(fingerprint {fp})", flush=True)
    if not pending:
        return
    newly = 0
    with mp.get_context("spawn").Pool(processes=max(1, a.workers)) as pool:
        for iid, res, secs in pool.imap_unordered(_worker, pending):
            atomic_write_json(run_dir / "items" / f"{iid}.json", res)
            atomic_write_json(run_dir / "items" / f"{iid}.timing.json", dict(seconds=secs))
            if man_path.exists():
                shutil.copyfile(man_path, lastgood)
            man["items"][iid]["status"] = "done"
            atomic_write_json(man_path, man)
            newly += 1
            print(f"done {iid} ({secs}s) [{newly}/{len(pending)}]", flush=True)
            if a.max_items and newly >= a.max_items:
                print("max-items reached; stopping (resume by relaunching)", flush=True)
                pool.terminate()
                break


if __name__ == "__main__":
    main()
