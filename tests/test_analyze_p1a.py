"""The Phase-1a analysis logic on synthetic job results (no real result is read): radius selection, admissibility,
success rule, and the pooled/oracle bookkeeping."""
import json
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1a_core as core  # noqa: E402
from experiments import analyze_p1a as an  # noqa: E402

RUNS = ROOT / "experiments" / "r2_runs"


@pytest.fixture(scope="module")
def plan_main():
    return core.plan("main")


def _fake_run(name, plan_obj, dro_delta, broken=()):
    """Every job gets C_t = 500 + 2t; DRO jobs get C + dro_delta (a constant, so the effect is known exactly)."""
    cells, jobs = plan_obj
    d = RUNS / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "items").mkdir(parents=True)
    (d / "plan.json").write_text(json.dumps(dict(econ="main", cells=cells, jobs=jobs), sort_keys=True), encoding="utf-8")
    base = 500.0 + 2.0 * np.arange(core.ev.Q + 1)
    for k, s in jobs.items():
        C = base + (dro_delta if s["method"] == "dro" else 0.0)
        r = dict(ok=k not in broken, tiebreak_ok=True, bookkeeping_maxabs=0.0, backup_shortfall=0.0,
                 dual_primal_gap=0.0, status=dict(primary_obj=float(C.mean())), C=C.tolist())
        (d / "items" / f"J_{k}.json").write_text(json.dumps(r), encoding="utf-8")
    return name


def _cleanup(name):
    shutil.rmtree(RUNS / name, ignore_errors=True)


def test_uniformly_cheaper_dro_is_selected_and_passes(plan_main):
    name = _fake_run("test_fake_p1a_pass", plan_main, dro_delta=-5.0)
    try:
        S = an.summarize(an.analyze(name))
        for z in ("A", "B"):
            c = S["contexts"][z]
            assert c["eps_star_freq"]["0"] == 0                         # SP never selected when DRO is cheaper
            assert c["primary_dJ"] == pytest.approx(5.0)
            assert c["mc_ci95"][0] == pytest.approx(5.0)
            assert c["crit_i"] and c["crit_ii"]
            assert c["factorial"]["robustness_pooled"]["diff"] == pytest.approx(5.0)
            assert c["factorial"]["robustness_pooled"]["n"] == c["n_rows"]
        assert S["verdict"] == "PASS"
    finally:
        _cleanup(name)


def test_uniformly_costlier_dro_is_never_selected_and_fails(plan_main):
    name = _fake_run("test_fake_p1a_fail", plan_main, dro_delta=+5.0)
    try:
        S = an.summarize(an.analyze(name))
        for z in ("A", "B"):
            c = S["contexts"][z]
            assert c["eps_star_freq"]["0"] == sum(c["eps_star_freq"].values())   # eps* = 0: CDRO = CSP
            assert c["primary_dJ"] == pytest.approx(0.0) and not c["crit_i"]
        assert S["verdict"] == "FAIL"
    finally:
        _cleanup(name)


def test_cells_with_a_failed_primary_job_are_excluded_and_listed(plan_main):
    cells, jobs = plan_main
    broken = {cells[0]["jobs"]["CDRO80"]}
    name = _fake_run("test_fake_p1a_broken", plan_main, dro_delta=-5.0, broken=broken)
    try:
        out = an.analyze(name)
        assert any("CDRO80" in e["failed"] for e in out["excluded"])
        assert len(out["excluded"]) == sum(1 for c in cells if c["jobs"]["CDRO80"] in broken)
        assert an.summarize(out)["verdict"] == "INCOMPLETE"
    finally:
        _cleanup(name)


def test_an_absent_context_can_never_pass(plan_main):
    """A alone passing must not produce an overall PASS when B is missing."""
    cells, jobs = plan_main
    broken = {c["jobs"]["CSP"] for c in cells if c["ctx"] == "B"}
    name = _fake_run("test_fake_p1a_noB", plan_main, dro_delta=-5.0, broken=broken)
    try:
        S = an.summarize(an.analyze(name))
        assert S["contexts"]["A"]["crit_i"] and S["contexts"]["B"]["n_rows"] == 0
        assert S["verdict"] == "INCOMPLETE"
    finally:
        _cleanup(name)


def test_secondary_comparisons_use_matched_rows(plan_main):
    """A missing secondary job must not turn a selection effect into a reported effect."""
    cells, jobs = plan_main
    broken = {c["jobs"]["PSP"] for c in cells if c["ctx"] == "A" and c["level"] == "high"}
    name = _fake_run("test_fake_p1a_psp", plan_main, dro_delta=-5.0, broken=broken)
    try:
        S = an.summarize(an.analyze(name))
        f = S["contexts"]["A"]["factorial"]
        # pooled jobs are shared by cells with identical pooled training information, so count by job key
        expected = 5 * sum(1 for c in cells if c["ctx"] == "A" and c["jobs"]["PSP"] not in broken)
        assert 0 < f["conditioning_SP"]["n"] == expected < S["contexts"]["A"]["n_rows"]
        assert f["conditioning_SP"]["diff"] == pytest.approx(0.0)
    finally:
        _cleanup(name)
