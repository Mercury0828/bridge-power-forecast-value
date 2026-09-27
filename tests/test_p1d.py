"""Phase 1d: the B4 rounding-and-repair rule (experiments/p1d_core.round_repair) keeps the declared equipment accounting
:
- every physical decision is a whole number of units;
- retained units never exceed delivered units, and disposed units are whole;
- the repaired policy is feasible, with independent bookkeeping and zero backup shortfall;
- signal copies keep common decisions at common histories (nodes before tau)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.bridge import make_params, Model, polish, recompute_costs, backup_shortfall  # noqa: E402
from model.signal import SignalModel, joint_law, node_names  # noqa: E402
from experiments.p1d_core import round_repair, batch_repair, UNITS, BU_UNIT, RENT_UNIT  # noqa: E402

P1A = dict(backup_rating="DCC", backup_sym=True, age_salvage=True, backup_ramp=True, service_hard=True)
PHAT = np.array([0.0, 0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05])


def toy(**kw):
    over = dict(Q=8, it_ramp=((2, 20.0), (5, 40.0)), lead={"GE": 3, "DG": 2, "ABS": 1, "BESS": 1}, L_permit=2)
    over.update(kw)
    return make_params("A", p_R=25.0, cap_R=30.0, delay_mult=3.0, residence_on=True, **over)


def _whole(x, u):
    return abs(x / u - round(x / u)) < 1e-6


def _check_accounting(P, m2):
    for j in P.assets:
        for k in range(P.Q + 1):
            assert _whole(m2.val(m2.v["x"][j][k]), UNITS[j]), (j, k)
    for T in range(1, P.Q + 1):
        for j in P.assets:
            for k in range(T):
                y, x = m2.val(m2.v["y"][T][j][k]), m2.val(m2.v["x"][j][k])
                assert y <= x + 1e-6, (T, j, k)
                assert _whole(y, UNITS[j]) and _whole(x - y, UNITS[j]), (T, j, k)
        for (_, _, var) in m2.bu_orders[T]:
            assert _whole(m2.val(var), BU_UNIT)
    for q in range(1, P.Q + 1):
        assert _whole(m2.val(m2.v["RST"][q]), RENT_UNIT) and _whole(m2.val(m2.v["RNR"][q]), RENT_UNIT)
    assert np.max(np.abs(m2.C_values() - recompute_costs(P, m2))) < 1e-6
    assert backup_shortfall(P, m2) < 1e-6


def test_round_repair_whole_units_conservation_and_feasibility():
    for extra in ({}, dict(stage_len=2, stage_frac=0.5), dict(backup_mode="spine")):
        P = toy(**P1A, **extra)
        m = Model(P)
        assert m.solve_saa(PHAT, tiebreak_mode="lp")["ok"]
        pm, st = polish(P, m.spine_values())
        assert st["ok"]
        m2, st2, added = round_repair(P, pm)
        assert st2["ok"], (extra, st2["status"])
        _check_accounting(P, m2)
        assert all(v >= -1e-6 for v in added.values())                 # the rule only ever rounds up


def test_round_repair_keeps_common_decisions_at_common_histories():
    P = toy(**P1A)
    K = np.array([[0.8] * 9, [0.2] * 9])
    K[:, 5:] = [[0.2], [0.8]]
    sm = SignalModel(P, 2, 2)
    assert sm.solve_saa(joint_law(PHAT, K, 2), tiebreak=True, tiebreak_mode="lp")["ok"]
    spines = []
    for c in sm.copies:
        pm, st = polish(P, c.spine_values())
        assert st["ok"]
        m2, st2, _ = round_repair(P, pm)
        assert st2["ok"]
        _check_accounting(P, m2)
        spines.append(m2.spine_values())
    shared = [n for k in range(2) for n in node_names(P, k)]
    assert shared
    for n in shared:
        if n in spines[0]:
            assert abs(spines[1][n] - spines[0][n]) < 1e-9, n


def test_batch_repair_cumulative_dominance_whole_units_and_common_histories():
    """Phase 1e E2: cumulative batching never lowers installed capacity, is whole-unit, keeps common decisions at common
    histories, and the integer branch re-optimization is feasible with exact bookkeeping."""
    P = toy(**P1A, stage_len=2, stage_frac=0.5)
    K = np.array([[0.8] * 9, [0.2] * 9])
    K[:, 5:] = [[0.2], [0.8]]
    sm = SignalModel(P, 2, 2)
    assert sm.solve_saa(joint_law(PHAT, K, 2), tiebreak=True, tiebreak_mode="lp")["ok"]
    spines = []
    for c in sm.copies:
        pm, st = polish(P, c.spine_values())
        assert st["ok"]
        m2, st2 = batch_repair(P, pm)
        assert st2["ok"], st2["status"]
        _check_accounting(P, m2)
        for j in P.assets:
            cum = cum_b = 0.0
            for k in range(P.Q + 1):
                cum += pm.val(pm.v["x"][j][k])
                cum_b += m2.val(m2.v["x"][j][k])
                assert m2.val(m2.v["x"][j][k]) >= -1e-9
                assert cum_b >= cum - 1e-6                       # capacity dominance at every node
                assert cum_b < cum + UNITS[j] + 1e-6             # at most one partial unit above
        spines.append(m2.spine_values())
    shared = [n for k in range(2) for n in node_names(P, k)]
    for n in shared:
        if n in spines[0]:
            assert abs(spines[1][n] - spines[0][n]) < 1e-9, n


def test_batch_repair_with_rented_standby_keeps_the_contract_rules():
    """Phase 1e E2 on E1 policies: rounded rented standby keeps the minimum commitment and whole standby units."""
    P = toy(**P1A, backup_mode="spine", rsb_on=True, rsb_price_mult=0.3)
    m = Model(P)
    assert m.solve_saa(PHAT, tiebreak_mode="lp")["ok"]
    pm, st = polish(P, m.spine_values())
    assert st["ok"]
    m2, st2 = batch_repair(P, pm)
    assert st2["ok"], st2["status"]
    _check_accounting(P, m2)
    M = P.rsb_min_q
    for T in range(1, P.Q + 1):
        path_r = {q: m2.val(m2.v["RSB"][q]) for q in range(1, T + 1)}
        path_s = {q: m2.val(m2.v["RSBs"][q]) for q in range(1, T + 1)}
        for q in range(T + 1, P.Q + 1):
            path_r[q], path_s[q] = m2.val(m2.v["RSBb"][T][q]), m2.val(m2.v["RSBbs"][T][q])
        for q in range(1, P.Q + 1):
            assert _whole(path_r[q], BU_UNIT) and _whole(path_s[q], BU_UNIT)
            assert path_s[q] >= path_r[q] - path_r.get(q - 1, 0.0) - 1e-6
            assert path_r[q] >= sum(path_s[j] for j in range(max(1, q - M + 1), q + 1)) - 1e-6
