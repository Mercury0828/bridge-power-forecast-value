"""Phase-1a evidence model (model/evidence.py): laws, reports, estimator, calibration family, W1."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import evidence as ev  # noqa: E402

T = np.arange(1, ev.Q + 2)


@pytest.mark.parametrize("ctx", ["A", "B"])
def test_laws_are_distributions_and_shift_moves_the_mean(ctx):
    for cv in ev.CV_RANGE[ctx]:
        means = []
        for s in (-2, -1, 0, 1, 2):
            p = ev.truth(ctx, cv, s)
            assert p.min() >= 0 and abs(p.sum() - 1) < 1e-12
            means.append(float(T @ p))
        assert all(b > a + 0.5 for a, b in zip(means, means[1:]))


def test_context_a_starts_at_the_requested_date_and_negative_shift_piles_onto_it():
    p0, pm = ev.truth("A", 0.75, 0), ev.truth("A", 0.75, -2)
    assert p0[: ev.E0_A - 1].sum() == 0 and pm[: ev.E0_A - 1].sum() == 0
    assert pm[ev.E0_A - 1] > 0.3 and p0[ev.E0_A - 1] < 1e-12


def test_context_b_respects_the_seven_year_bound_without_shift():
    for cv in ev.CV_RANGE["B"]:
        p = ev.truth("B", cv, 0)
        assert p[28:].sum() < 1e-12 and p[:3].sum() == 0          # T <= 28 and no energization during the study


def test_real_evidence_reproduces_the_stated_statistics():
    pa = ev.estimate("A", ev.REAL_REPORTS["A"])
    assert abs(float(T @ pa) - (ev.E0_A + ev.MU_D_A + 0.5)) < 0.1  # ceil adds about half a quarter on average
    pb = ev.estimate("B", ev.REAL_REPORTS["B"])
    assert int(np.searchsorted(np.cumsum(pb), 0.5)) + 1 in (15, 16)


def test_reports_and_family_are_seed_reproducible_and_centred():
    r1 = ev.draw_reports("A", 0.75, 3, np.random.default_rng(7))
    r2 = ev.draw_reports("A", 0.75, 3, np.random.default_rng(7))
    assert r1 == r2
    fam = ev.calibration_family("A", r1, 300, np.random.default_rng(3))
    centre = ev.estimate("A", r1)
    assert abs(float(T @ fam.mean(axis=0)) - float(T @ centre)) < 0.2
    rad = ev.radii(centre, fam)
    assert 0 < rad[0.5] < rad[0.8]


def test_w1_is_the_l1_distance_between_cdfs():
    p = ev.truth("B", 0.25, 0)
    assert ev.w1(p, p) == 0.0
    shifted = np.roll(p, 1)                                          # exactly one quarter later (no overflow mass)
    assert abs(ev.w1(p, shifted) - 1.0) < 1e-12


def test_allowed_states_contain_the_centres():
    for ctx in ("A", "B"):
        c = ev.estimate(ctx, ev.REAL_REPORTS[ctx])
        allowed = set(ev.allowed_states(ctx))
        assert all(i in allowed for i in range(ev.Q + 1) if c[i] > 1e-12)


def test_b_likelihood_table_is_a_distribution_over_reports():
    tab = ev.lik_table_B(ev.S_RANGE_B)
    assert np.allclose(tab.sum(axis=2), 1.0)


def test_b_posterior_respects_rounding_without_hard_certainty():
    """Reports round noisy sample medians, so they are evidence, not exact constraints."""
    r3 = [dict(study_months=(9.0, 12.0), typical_years=3.0)]
    _, par = ev.calibration_family("B", r3, 400, np.random.default_rng(2), return_params=True)
    assert 10.0 < np.quantile(par[:, 0], 0.1) < 12.0 < np.quantile(par[:, 0], 0.9) < 16.0
    mixed = r3 * 2 + [dict(study_months=(9.0, 12.0), typical_years=2.0)]
    _, par2 = ev.calibration_family("B", mixed, 400, np.random.default_rng(2), return_params=True)
    assert par2[:, 0].std() > 0.3 and 9.0 < par2[:, 0].mean() < 12.0     # between the reports, not collapsed


def test_a_posterior_concentrates_on_the_reported_mean():
    _, par = ev.calibration_family("A", ev.REAL_REPORTS["A"], 400, np.random.default_rng(5), return_params=True)
    assert abs(par[:, 0].mean() - ev.MU_D_A) < 0.3
    assert ev.CV_RANGE["A"][0] <= par[:, 1].min() and par[:, 1].max() <= ev.CV_RANGE["A"][1]


def test_b_adversary_is_not_capped_at_28_quarters():
    """R02 is a regional summary, not a probability-zero tail."""
    assert ev.Q in ev.allowed_states("B") and 2 not in ev.allowed_states("B")


def test_every_pmf_reaching_the_solver_has_no_mass_below_the_floor():
    """Masses in (0, 1e-9) make HiGHS reject the tie-break row; laws and mixtures are floored (declared convention)."""
    from experiments import p1a_core as core
    bad = lambda p: bool(np.any((p > 0) & (p < ev.PMF_FLOOR)))           # noqa: E731
    for ctx in ("A", "B"):
        for cv in ev.CV_RANGE[ctx]:
            for s in (-2, 0, 2):
                assert not bad(ev.truth(ctx, cv, s))
    tA = core.training("A", core.reports_of("A", "high", 3, 0))
    tB = core.training("B", core.reports_of("B", "high", 3, 0))
    tp = core.pooled_training(tA, tB)
    for p in (tA["centre"], tA["p_mix"], tB["centre"], tB["p_mix"], tp["centre"], tp["p_mix"], *tA["fam"][:50]):
        assert not bad(p) and abs(p.sum() - 1) < 1e-12
