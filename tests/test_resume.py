"""V4: kill -> relaunch -> resume must skip finished items and reproduce an uninterrupted run bit for bit."""
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = ROOT / "experiments" / "r2_runs"


def _run(args, **kw):
    return subprocess.run([PY, str(ROOT / "experiments" / "harness.py"), *args], cwd=ROOT, capture_output=True,
                          text=True, **kw)


def _clean(*runs):
    """Remove stale run directories first: a leftover directory from an earlier session would otherwise satisfy the
    'items on disk' poll before the new process has moved it aside, and the kill would land too early."""
    for r in runs:
        shutil.rmtree(RUNS / r, ignore_errors=True)
        for old in RUNS.glob(r + ".old_*"):
            shutil.rmtree(old, ignore_errors=True)


def _items(run):
    d = RUNS / run / "items"
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.json")) if not p.name.endswith(".timing.json")}


def test_v4_kill_relaunch_resume_bit_identical():
    _clean("v4_ref", "v4_kill")
    ref = _run(["--run", "v4_ref", "--toy", "--fresh", "--workers", "4"])
    assert ref.returncode == 0, ref.stderr[-2000:]
    ref_items = _items("v4_ref")
    assert len(ref_items) == 16

    # 1) hard kill of the whole process tree after at least 3 items are on disk
    proc = subprocess.Popen([PY, str(ROOT / "experiments" / "harness.py"), "--run", "v4_kill", "--toy", "--fresh",
                             "--workers", "2"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = RUNS / "v4_kill" / "items"
    t0 = time.time()
    while time.time() - t0 < 600:
        if d.exists() and len([p for p in d.glob("*.json") if not p.name.endswith(".timing.json")]) >= 3:
            break
        time.sleep(0.5)
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()
    proc.wait(timeout=60)
    partial = _items("v4_kill")
    assert 3 <= len(partial) < 16, len(partial)

    # 2) relaunch without --fresh: resumes, skips finished items, completes the rest
    res = _run(["--run", "v4_kill", "--toy", "--workers", "4"])
    assert res.returncode == 0, res.stderr[-2000:]
    assert f"{len(partial)} done" in res.stdout or "done" in res.stdout
    full = _items("v4_kill")
    assert full.keys() == ref_items.keys()
    for name in ref_items:
        assert full[name] == ref_items[name], name     # bit-identical
    # finished items were not recomputed: their bytes are unchanged from before the relaunch
    for name, b in partial.items():
        assert full[name] == b, name


def test_v4_max_items_then_resume():
    _clean("v4_max")
    r1 = _run(["--run", "v4_max", "--toy", "--fresh", "--workers", "2", "--max-items", "2"])
    assert r1.returncode == 0, r1.stderr[-2000:]
    n1 = len(_items("v4_max"))
    assert 2 <= n1 < 16
    r2 = _run(["--run", "v4_max", "--toy", "--workers", "4"])
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert len(_items("v4_max")) == 16


def test_v4_p1a_kind_max_items_then_resume_bit_identical():
    """The Phase-1a harness kind resumes and reproduces an uninterrupted run bit for bit."""
    _clean("v4_p1a_ref", "v4_p1a_max")
    args = ["--kind", "p1a", "--limit-jobs", "3", "--only-method", "saa"]
    ref = _run(["--run", "v4_p1a_ref", "--fresh", "--workers", "3", *args])
    assert ref.returncode == 0, ref.stderr[-2000:]
    ref_items = _items("v4_p1a_ref")
    assert len(ref_items) == 3
    r1 = _run(["--run", "v4_p1a_max", "--fresh", "--workers", "1", "--max-items", "1", *args])
    assert r1.returncode == 0, r1.stderr[-2000:]
    partial = _items("v4_p1a_max")
    assert 1 <= len(partial) < 3
    r2 = _run(["--run", "v4_p1a_max", "--workers", "3", *args])
    assert r2.returncode == 0, r2.stderr[-2000:]
    full = _items("v4_p1a_max")
    assert full.keys() == ref_items.keys()
    for name in ref_items:
        assert full[name] == ref_items[name], name
    for name, b in partial.items():
        assert full[name] == b, name
