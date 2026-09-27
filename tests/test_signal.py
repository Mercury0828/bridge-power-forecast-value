"""Signal tree (model/signal.py; Gate 2)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.bridge import Model, make_params  # noqa: E402
from model.signal import SignalModel, joint_law, evaluate  # noqa: E402

PHAT = np.array([0.0, 0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05])
P1A = dict(backup_rating="DCC", backup_sym=True, age_salvage=True, backup_ramp=True, service_hard=True)


def toy(**kw):
    over = dict(Q=8, it_ramp=((2, 20.0), (5, 40.0)), lead={"GE": 3, "DG": 2, "ABS": 1, "BESS": 1}, L_permit=2)
    over.update(kw)
    return make_params("B", p_R=25.0, cap_R=30.0, delay_mult=3.0, residence_on=True, **over)


def kernel(r, n=9):
    """2 categories: early (T <= 5) vs late; accuracy r."""
    cat = np.array([0 if i + 1 <= 5 else 1 for i in range(n)])
    return np.array([[r if cat[i] == y else 1 - r for i in range(n)] for y in range(2)])


def no_signal_value(P):
    m = Model(P)
    st = m.solve_saa(PHAT, tiebreak=False)
    assert st["ok"]
    return st["obj"]


@pytest.mark.parametrize("tau", [2, 3])
def test_uninformative_signal_has_zero_value(tau):
    P = toy(**P1A)
    sm = SignalModel(P, 2, tau)
    st = sm.solve_saa(joint_law(PHAT, kernel(0.5), tau))
    assert st["ok"]
    assert st["obj"] == pytest.approx(no_signal_value(P), rel=1e-6)


def test_signal_after_the_horizon_equals_no_signal():
    P = toy(**P1A)
    sm = SignalModel(P, 2, P.Q + 1)                  # every node linked: the signal can never be used
    st = sm.solve_saa(joint_law(PHAT, kernel(0.9), P.Q + 1))
    assert st["ok"] and st["obj"] == pytest.approx(no_signal_value(P), rel=1e-6)


def test_informative_signal_has_nonnegative_value_and_polished_costs_reproduce():
    P = toy(**P1A)
    tau = 2
    law = joint_law(PHAT, kernel(0.9), tau)
    sm = SignalModel(P, 2, tau)
    st = sm.solve_saa(law)
    assert st["ok"] and st["obj"] <= no_signal_value(P) + 1e-6
    costs = sm.polished_costs()
    assert evaluate(costs, law, tau) <= st["obj"] + 1e-6          # polishing only improves branches


def test_robust_signal_use_is_never_worse_than_ignoring_the_signal():
    """Min-max over a set of joint laws with a common marginal is <= the no-signal optimum."""
    P = toy(**P1A)
    tau = 2
    laws = [joint_law(PHAT, kernel(r), tau) for r in (0.55, 0.75, 0.95)]
    sm = SignalModel(P, 2, tau)
    st = sm.solve_robust(laws)
    assert st["ok"] and st["obj"] <= no_signal_value(P) + 1e-6


def test_tiebreak_keeps_the_primary_optimum():
    P = toy(**P1A)
    law = joint_law(PHAT, kernel(0.8), 2)
    a = SignalModel(P, 2, 2).solve_saa(law)
    b = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True)
    assert a["ok"] and b["ok"] and b["tiebreak_ok"] is True
    assert b["primary_obj"] == pytest.approx(a["obj"], rel=1e-6)


def test_lp_tiebreak_in_the_signal_tree_keeps_the_primary_optimum():
    P = toy(**P1A)
    law = joint_law(PHAT, kernel(0.8), 2)
    a = SignalModel(P, 2, 2).solve_saa(law)
    b = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True, tiebreak_mode="lp")
    assert b["ok"] and b["tiebreak_ok"] is True and b["primary_obj"] == pytest.approx(a["obj"], rel=1e-6)


def test_lp_pure_tiebreak_in_the_signal_tree_matches_lp():
    P = toy(**P1A)
    law = joint_law(PHAT, kernel(0.8), 2)
    a = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True, tiebreak_mode="lp")
    b = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True, tiebreak_mode="lp_pure")
    assert a["tiebreak_ok"] is True and b["tiebreak_ok"] is True
    assert b["primary_obj"] == pytest.approx(a["primary_obj"], rel=1e-9)
    assert b["obj"] == pytest.approx(a["obj"], rel=1e-7, abs=1e-6)


def test_tiebreak_without_presolve_in_the_signal_tree_matches():
    P = toy(**P1A)
    law = joint_law(PHAT, kernel(0.8), 2)
    a = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True, tiebreak_mode="lp")
    b = SignalModel(P, 2, 2).solve_saa(law, tiebreak=True, tiebreak_mode="lp", tiebreak_presolve=False)
    assert a["tiebreak_ok"] is True and b["tiebreak_ok"] is True
    assert b["primary_obj"] == pytest.approx(a["primary_obj"], rel=1e-9)
    assert b["obj"] == pytest.approx(a["obj"], rel=1e-7, abs=1e-6)


@pytest.mark.parametrize("q, linked", [(1, True), (2, True), (3, False)])
def test_rsb_spine_rentals_are_linked_across_signal_copies_before_tau(q, linked):
    """Phase 1e E1 (plan v2 "Tests"): the rented standby for quarter q is contracted at spine
    node q - 1. It is therefore common to every signal copy while q - 1 < tau, and may differ once the signal is seen:
    forcing different levels in two copies is infeasible for q <= tau and feasible for q = tau + 1."""
    from model.bridge import node_names
    P = toy(**P1A, backup_mode="spine", rsb_on=True, rsb_price_mult=0.3)
    tau = 2
    assert all(f"{key}|{k + 1}" in node_names(P, k) for k in range(tau) for key in ("RSB", "RSBs"))
    sm = SignalModel(P, 2, tau)
    sm.h.addConstr(1.0 * sm.copies[0]._lookup(f"RSB|{q}") <= 0.0)
    sm.h.addConstr(1.0 * sm.copies[1]._lookup(f"RSB|{q}") >= 5.0)
    st = sm.solve_saa(joint_law(PHAT, kernel(0.9), tau))
    assert st["ok"] is (not linked)
    if linked:
        assert "Infeasible" in st["status"]
        # control: the same level in both copies is feasible, so the infeasibility above comes from the link
        sm2 = SignalModel(P, 2, tau)
        for c in sm2.copies:
            sm2.h.addConstr(1.0 * c._lookup(f"RSB|{q}") >= 5.0)
        assert sm2.solve_saa(joint_law(PHAT, kernel(0.9), tau))["ok"]


def test_clean_law_floors_and_preserves_total_mass_and_tiebreak_guard():
    from model.signal import clean_law
    P = toy(**P1A)
    law = joint_law(PHAT, kernel(0.9), 2)
    law["post"][0][5] = 5e-11                                        # a tiny mass that HiGHS cannot take in a row
    with pytest.raises(ValueError):
        SignalModel(P, 2, 2).solve_saa(law, tiebreak=True)
    cl = clean_law(law)
    tot = sum(cl["pre"]) + sum(sum(q) for q in cl["post"])
    assert abs(tot - 1) < 1e-12 and cl["post"][0][5] == 0.0
    assert SignalModel(P, 2, 2).solve_saa(cl, tiebreak=True)["ok"]
