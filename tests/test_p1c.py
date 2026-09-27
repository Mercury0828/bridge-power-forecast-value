"""Phase 1c (experiments/p1c_core.py, experiments/analyze_p1c.py): the signal families, and the analysis logic on
synthetic results only."""
import json
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments import analyze_p1c as an  # noqa: E402
from model.signal import joint_law  # noqa: E402

RUNS = ROOT / "experiments" / "r2_runs"


def test_kernels_are_conditional_distributions_and_sharpen_with_accuracy():
    th = (10, 11)
    for fam in core.FAMILIES:
        K = core.true_kernel(th, fam, np.ones(core.ev.Q + 1) / (core.ev.Q + 1))
        assert np.allclose(K.sum(axis=0), 1.0) and K.min() >= 0
    K1, K4 = core.kernel(th, 1.0), core.kernel(th, 4.0)
    cat = [core.category(i + 1, th) for i in range(core.ev.Q + 1)]
    acc = lambda K: np.mean([K[cat[i], i] for i in range(core.ev.Q + 1)])       # noqa: E731
    assert acc(K1) > acc(K4)
    assert np.allclose(core.planner_kernel(th).sum(axis=0), 1.0)


def test_uninformative_family_is_independent_of_T():
    p = core.truth("B", "low", 0)
    K = core.true_kernel((14, 18), "uninf", p)
    assert np.allclose(K, K[:, [0]])


def test_joint_law_preserves_the_physical_marginal():
    p = core.truth("A", "high", 0)
    law = joint_law(p, core.true_kernel((10, 11), "sym2", p), core.TAU)
    marg = np.array(law["pre"]) + np.sum(law["post"], axis=0)
    assert np.allclose(marg, p)


def test_optimistic_family_reports_earlier_categories():
    p = core.truth("B", "low", 0)
    th = (14, 18)
    e = joint_law(p, core.true_kernel(th, "sym2", p), core.TAU)
    o = joint_law(p, core.true_kernel(th, "opt2", p), core.TAU)
    assert np.sum(o["post"][0]) > np.sum(e["post"][0])                  # more "early" forecasts when biased early


def test_categories_come_from_the_training_centre_only():
    cells, jobs = core.plan()
    for c in cells[:5]:
        centre = core.ev.estimate(core.base_ctx(c["cfg"]), c["reports"])
        assert tuple(c["thresholds"]) == core.thresholds(centre)


def test_report_outcomes_integrate_the_training_distribution():
    """Every report outcome is a weighted cell, including B's rare 2- and 4-year reports."""
    cells, jobs = core.plan()
    for cfg in core.CONFIGS:
        for lv in core.LEVELS:
            cs = [c for c in cells if c["cfg"] == cfg and c["level"] == lv]
            tot = sum(c["p"] for c in cs)
            assert 1 - 1e-6 - 1e-12 <= tot <= 1 + 1e-9
    b_high = [c for c in cells if c["cfg"] == "B" and c["level"] == "high"]
    years = {c["reports"][0]["typical_years"] for c in b_high}
    assert {2.0, 3.0, 4.0} <= years


@pytest.fixture(scope="module")
def plan_c():
    return core.plan()


def _fake(name, plan_obj, delta_fn, broken=()):
    cells, jobs = plan_obj
    d = RUNS / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "items").mkdir(parents=True)
    (d / "plan.json").write_text(json.dumps(dict(cells=cells, jobs=jobs), sort_keys=True), encoding="utf-8")
    base = 500.0 + 2.0 * np.arange(core.ev.Q + 1)
    for k, s in jobs.items():
        if s["kind"] == "nosig":
            r = dict(ok=True, tiebreak_ok=True, C=base.tolist(), bookkeeping_maxabs=0.0, backup_shortfall=0.0)
        else:
            dl = delta_fn(s["cfg"]) if s["kind"] == "sig" else 0.0
            r = dict(ok=True, tiebreak_ok=True, costs=[(base - dl).tolist()] * core.NCAT, bookkeeping_maxabs=0.0,
                     backup_shortfall=0.0, dual_bound=float(base.mean() - 50))
        if k in broken:
            r["ok"] = False
        (d / "items" / f"J_{k}.json").write_text(json.dumps(r), encoding="utf-8")
    return name


def test_claims_follow_the_registered_rule(plan_c):
    # a signal policy exactly $5M cheaper in every scenario: H1/H2 pass; the staging interaction is 0 and fails
    name = _fake("test_fake_p1c_pass", plan_c, lambda cfg: 5.0)
    try:
        S = an.summarize(an.analyze(name))
        assert S["complete"]
        for h in ("H1a", "H1b", "H2a", "H2b"):
            assert S["claims"][h]["verdict"] == "PASS" and S["claims"][h]["mean"] == pytest.approx(5.0)
        for h in ("H3a", "H3b"):
            assert S["claims"][h]["verdict"] == "FAIL" and S["claims"][h]["mean"] == pytest.approx(0.0)
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def test_staging_interaction_is_paired_and_can_pass(plan_c):
    name = _fake("test_fake_p1c_stage", plan_c, lambda cfg: 6.0 if cfg.endswith("+staged") else 1.0)
    try:
        S = an.summarize(an.analyze(name))
        assert S["claims"]["H3a"]["mean"] == pytest.approx(5.0) and S["claims"]["H3a"]["verdict"] == "PASS"
        assert S["claims"]["H1a"]["verdict"] == "FAIL"                  # B unstaged gains only $1M
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)


def test_an_unusable_policy_job_makes_the_verdict_incomplete(plan_c):
    cells, jobs = plan_c
    broken = {next(k for k, s in jobs.items() if s["kind"] == "sig" and s["cfg"] == "B")}
    name = _fake("test_fake_p1c_broken", plan_c, lambda cfg: 5.0, broken=broken)
    try:
        S = an.summarize(an.analyze(name))
        assert not S["complete"] and S["claims"]["H1a"]["verdict"] == "INCOMPLETE"
    finally:
        shutil.rmtree(RUNS / name, ignore_errors=True)
