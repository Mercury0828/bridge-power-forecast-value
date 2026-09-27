"""Pre-sweep verification V1-V3 (data/r2_preregistration.md section 7) on toy instances, under the R2 economics and
the Phase-1a economics (D-011..D-015). V4 lives in test_resume.py."""
import itertools
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.bridge import (make_params, Model, recompute_costs, worst_case_expectation, polish, backup_shortfall,  # noqa: E402
                          ASSETS)


def toy(ctx="A", residence_on=True, delay_mult=3.0, p_R=25.0, cap_R=30.0, **kw):
    over = dict(Q=8, it_ramp=((2, 20.0), (5, 40.0)), lead={"GE": 3, "DG": 2, "ABS": 1, "BESS": 1}, L_permit=2)
    over.update(kw)
    return make_params(ctx, p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on, **over)


PHAT = np.array([0.0, 0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05])   # T = 1..9 (9 = beyond Q)

# Phase-1a main-case economics: DCC backup counting (D-012), symmetric backup (D-015), age-based salvage (D-015),
# ramp-following backup requirement, must-energize service (D-014).
P1A = dict(backup_rating="DCC", backup_sym=True, age_salvage=True, backup_ramp=True, service_hard=True)
FLAGSETS = {
    "r2": {},
    "p1a": P1A,
    "p1a_spine_backup": dict(P1A, backup_mode="spine"),
    "p1a_int_orders": dict(P1A, int_orders=True),
    "p1a_dcc_bridge": dict(P1A, dg_bridge_mult=2.5 / 2.1),
    "p1a_refurb_band": dict(P1A, dg_refurb=0.05, salv_mult=0.5),
    "p1a_soft_service": dict(P1A, service_hard=False, vod_base=1.5),
    "p1a_same_function": dict(P1A, same_function_standby=True),
    "legacy_structure_sym": dict(backup_rating="DCC", backup_sym=True, age_salvage=True),
    # Phase 1d (docs/phase1d_plan.md v2)
    "p1d_backup_delay": dict(P1A, backup_delay=2),
    "p1d_staged_backup_from_T": dict(P1A, stage_len=2, stage_frac=0.5, backup_delay=0),
    "p1d_refurb_stress": dict(P1A, dg_refurb=0.05, refurb_delay=1, refurb_derate=0.8),
    "p1d_staged_refurb_spine": dict(P1A, stage_len=2, stage_frac=0.5, dg_refurb=0.05, refurb_delay=1,
                                    refurb_derate=0.8, backup_mode="spine"),
    # Phase 1e E1 (docs/phase1e_plan.md v2): rented standby
    "p1e_rsb": dict(P1A, rsb_on=True, rsb_price_mult=0.3),
    "p1e_rsb_spine": dict(P1A, rsb_on=True, backup_mode="spine", rsb_price_mult=0.3),
    "p1e_rsb_spine_staged": dict(P1A, rsb_on=True, backup_mode="spine", stage_len=2, stage_frac=0.5,
                                 rsb_price_mult=0.3),
}


def primary(st):
    return st.get("primary_obj", st["obj"])


@pytest.mark.parametrize("flags", ["r2", "p1a"])
@pytest.mark.parametrize("kind", ["saa", "dro"])
@pytest.mark.parametrize("residence_on", [True, False])
def test_v1_bruteforce_root_enumeration(kind, residence_on, flags):
    P = toy(residence_on=residence_on, **FLAGSETS[flags])
    eps = 0.8
    full = Model(P)
    st = full.solve_saa(PHAT) if kind == "saa" else full.solve_dro(PHAT, eps)
    assert st["ok"]
    best = np.inf
    for nge, ndg in itertools.product(range(0, 9), range(0, 31)):
        m = Model(P, fix={"n|GE": nge, "n|DG": ndg})
        s = m.solve_saa(PHAT, tiebreak=False) if kind == "saa" else m.solve_dro(PHAT, eps, tiebreak=False)
        if s["ok"]:
            best = min(best, s["obj"])
    assert abs(best - primary(st)) <= 1e-6 * max(1.0, abs(best)), (best, primary(st))


@pytest.mark.parametrize("flags", list(FLAGSETS))
def test_v2_bookkeeping_matches_independent_recomputation(flags):
    for ctx in ("A", "B"):
        for res in (True, False):
            P = toy(ctx, residence_on=res, **FLAGSETS[flags])
            m = Model(P)
            assert m.solve_dro(PHAT, 0.8)["ok"]
            C, C2 = m.C_values(), recompute_costs(P, m)
            assert np.max(np.abs(C - C2)) < 1e-6
            assert backup_shortfall(P, m) < 1e-6


@pytest.mark.parametrize("flags,asset", [("r2", "GE"), ("p1a", "GE"), ("p1a_spine_backup", "SB")])
def test_v2_nonanticipativity_node_changes_do_not_leak_backwards(flags, asset):
    """Changing a decision at spine node k must leave C_T unchanged for every T <= k, since scenario T only
    traverses nodes m_0..m_{T-1}. Checked for extra orders at every node."""
    P = toy(**FLAGSETS[flags])
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    spine = m.spine_values()
    base_m, st = polish(P, spine)
    assert st["ok"]
    base = base_m.C_values()
    for k in range(1, P.Q):
        pert = dict(spine)
        pert[f"x|{asset}|{k}"] = pert[f"x|{asset}|{k}"] + 10.0   # an extra 10 MW order placed at node k
        pm, pst = polish(P, pert)
        assert pst["ok"]
        C = pm.C_values()
        for T in range(1, k + 1):                                # scenarios connected at or before quarter k
            assert abs(C[T - 1] - base[T - 1]) < 1e-6, (k, T, C[T - 1], base[T - 1])
        assert any(abs(C[T - 1] - base[T - 1]) > 1e-6 for T in range(k + 1, P.Q + 2)), k


@pytest.mark.parametrize("flags", ["r2", "p1a", "p1a_spine_backup"])
def test_v3_dro_duality(flags):
    for ctx in ("A", "B"):
        P = toy(ctx, **FLAGSETS[flags])
        for eps in (0.3, 0.8, 2.0):
            m = Model(P)
            st = m.solve_dro(PHAT, eps)
            assert st["ok"]
            wc = worst_case_expectation(m.C_values(), PHAT, eps)
            assert abs(wc - primary(st)) <= 1e-6 * max(1.0, abs(wc)), (ctx, eps, wc, primary(st))


@pytest.mark.parametrize("z", [0, 1])
def test_regime_rule_limits_bridge_window(z):
    """Residence rule on. z = 1 (NR regime): all on-site bridge generation runs in at most 3 consecutive quarters.
    z = 0: no NR rental at all. The regime is forced so that both branches are exercised."""
    P = toy(residence_on=True, p_R=5.0, delay_mult=10.0)      # cheap rentals + high delay value: wants long bridging
    m = Model(P, fix={"z": z})
    assert m.solve_saa(PHAT)["ok"]
    sv = m.spine_values()
    on = [q for q in range(1, P.Q + 1)
          if sum(sv[f"{g}|{q}|{b}"] for g in ("gGE", "gDG", "gNR", "gST") for b in (0, 1)) > 1e-6]
    if z == 1:
        assert len(on) <= 3 and (not on or max(on) - min(on) <= 2), on
        assert len(on) >= 1                                   # the regime is actually used in this instance
    else:
        assert all(sv[f"RNR|{q}"] < 1e-9 for q in range(1, P.Q + 1))
        assert len(on) > 3                                     # without NR, bridging runs longer than the window


def test_st_requires_prior_authorization():
    P = toy(residence_on=False)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    sv = m.spine_values()
    for q in range(1, P.Q + 1):
        if sv[f"RST|{q}"] > 1e-9:
            assert any(sv[f"f|{k}"] == 1 for k in range(P.Q) if k <= q - 1 - P.L_permit), q


def test_bf1_in_transit_cohort_can_be_retained_and_runs_only_after_delivery():
    """A gas unit ordered at the root that is still in transit at connection can be retained; it may
    dispatch only from its delivery quarter, and retaining it must never cost more than forced disposal."""
    P = toy("B")                                               # context B prices make post-connection CCHP economic
    spine = None
    m = Model(P, fix={"n|GE": 2, "n|DG": 0})
    assert m.solve_det(2)["ok"]                                # grid at quarter 2, GE (lead 3) arrives at quarter 3
    spine = m.spine_values()
    pm, st = polish(P, spine)
    assert st["ok"]
    T = 2
    yk0 = pm.val(pm.v["y"][T]["GE"][0])
    for q in range(T, P.lead["GE"]):                          # before delivery: no gas dispatch
        for b in (0, 1):
            assert pm.val(pm.v["bg"][T, q, b]) < 1e-9
    # compare with forced disposal of the in-transit cohort
    fm = Model(P, fix=dict(spine, **{}))
    fm.h.changeColBounds(fm.v["y"][T]["GE"][0].index, 0.0, 0.0)
    fm.h.minimize(sum(fm.C[t] for t in range(1, P.Q + 2)))
    assert fm._status()["ok"]
    assert pm.C_values()[T - 1] <= fm.C_values()[T - 1] + 1e-9
    assert yk0 >= 0.0


def test_bf3_failed_tiebreak_is_flagged_not_reported_as_success():
    """If the second (tie-break) solve fails, the result must be flagged, never silently 'ok'."""
    P = toy()
    m = Model(P)
    real_minimize = m.h.minimize
    calls = {"n": 0}

    def minimize_then_starve(expr):
        calls["n"] += 1
        if calls["n"] >= 2:                                   # starve the tie-break (and the restore) of time
            m.h.setOptionValue("time_limit", 1e-9)
        return real_minimize(expr)

    m.h.minimize = minimize_then_starve
    st = m.solve_saa(PHAT)
    assert st["tiebreak_ok"] is False
    assert st["ok"] is False or abs(st["obj"] - st["primary_obj"]) <= 1e-6 * max(1.0, abs(st["primary_obj"]))


def test_bf3_analysis_excludes_items_with_failed_tiebreak():
    """The analysis must exclude (and report) items where any method's tie-break failed."""
    from experiments.analyze_r2 import item_admissible
    good = {"methods": {"CDRO": {"ok": True, "tiebreak_ok": True}, "B1": {"ok": True, "tiebreak_ok": True}}}
    flagged = {"methods": {"CDRO": {"ok": True, "tiebreak_ok": False}, "B1": {"ok": True, "tiebreak_ok": True}}}
    failed = {"methods": {"CDRO": {"ok": False}, "B1": {"ok": True, "tiebreak_ok": True}}}
    assert item_admissible(good)[0] is True
    ok, f, fl = item_admissible(flagged)
    assert ok is False and fl == ["CDRO"] and f == []
    ok, f, fl = item_admissible(failed)
    assert ok is False and f == ["CDRO"]


def test_b5_box_robust_objective_equals_max_cost_in_box():
    from model.bridge import percentile_quarter
    P = toy()
    lo, hi = percentile_quarter(PHAT, 0.05), percentile_quarter(PHAT, 0.95)
    m = Model(P)
    st = m.solve_box(lo, hi)
    assert st["ok"]
    C = m.C_values()
    assert abs(max(C[T - 1] for T in range(lo, hi + 1)) - primary(st)) <= 1e-6 * max(1.0, abs(primary(st)))


# ---------------------------------------------------------------------------------------------------------------
# Phase-1a economics (D-011..D-015)
# ---------------------------------------------------------------------------------------------------------------
def test_d012_backup_counted_at_dcc_and_r2_values_unchanged():
    P = make_params("A", backup_rating="DCC")
    assert P.bk_ratio == pytest.approx(2.5 / 2.1)
    assert P.c_bu == pytest.approx(0.600 * 2.75 / 2.5)          # same machine, priced per DCC MW
    assert P.fom_bu * 2.5 == pytest.approx(0.010 / 4)           # $10k per unit-year
    R2 = make_params("A")
    assert R2.bk_ratio == R2.dg_esp_ratio and R2.c_bu == R2.c_backup and R2.vod == 157.0 * 3 * 1000 / 1e6


@pytest.mark.parametrize("structure", [dict(), dict(backup_ramp=True)])
@pytest.mark.parametrize("rating", ["ESP", "DCC"])
def test_d015_retention_vs_purchase_differs_only_by_round_trip_loss(structure, rating):
    """Double-credit audit. Price a purchased standby MW like a retained DG MW (c_bu * bk_ratio = DG capex
    per MW-COP) and retain a DG cohort delivered exactly at T (age 0, like a new purchase). With D-015 symmetry,
    keeping it instead of selling it and buying backup saves exactly the round-trip loss (1 - sigma_ret(0)) * capex:
    fixed O&M and terminal value are identical. Without symmetry (R2), keeping also earned a terminal value that a
    purchase lacked."""
    T = 5
    c_backup = (2.0 / 2.1) / (2.75 / 2.1)            # makes c_bu * bk_ratio = DG capex per MW-COP at either rating
    for sym in (True, False):
        P = toy("A", backup_rating=rating, backup_sym=sym, age_salvage=True, c_backup=c_backup, **structure)
        assert P.c_bu * P.bk_ratio == pytest.approx(P.capex["DG"])
        k = T - P.lead["DG"]
        m0 = Model(P, fix={"n|DG": 0, **{f"x|DG|{kk}": 0.0 for kk in range(1, P.Q + 1)}})
        assert m0.solve_saa(PHAT)["ok"]
        spine = m0.spine_values()
        spine[f"x|DG|{k}"] = P.dg_unit                          # one DG unit, delivered exactly at T
        cost = {}
        for keep in (True, False):
            m = Model(P, fix=spine)
            yv = m.v["y"][T]["DG"][k]
            m.h.changeColBounds(yv.index, P.dg_unit if keep else 0.0, P.dg_unit if keep else 0.0)
            m.h.minimize(sum(m.C[t] for t in range(1, P.Q + 2)))
            assert m._status()["ok"]
            assert backup_shortfall(P, m) < 1e-6
            cost[keep] = m.C_values()[T - 1]
        loss = P.disc(T) * (1 - P.sig_ret("DG", 0)) * P.capex["DG"] * P.dg_unit
        if sym:
            assert cost[False] - cost[True] == pytest.approx(loss, abs=1e-6)
        else:
            assert cost[False] - cost[True] > loss + 0.5        # the R2 asymmetry: an unmatched terminal credit


def test_d015_age_salvage_is_monotone_one_curve_and_banded():
    P = make_params("A", age_salvage=True)
    for j in ASSETS:
        vals = [P.sig_ret(j, a) for a in range(0, 200)]
        assert vals[0] == pytest.approx(P.salv[j])
        assert all(b <= a for a, b in zip(vals, vals[1:])) and min(vals) >= 0.0
        assert P.sig_term(j, 12) == P.sig_ret(j, 12)             # retirement and terminal value share one curve
        assert P.sig_ret(j, -3) == P.sig_ret(j, 0)               # in transit: valued as new
    assert make_params("A", age_salvage=True, salv_mult=0.5).sig_ret("DG", 10) == pytest.approx(
        0.5 * P.sig_ret("DG", 10))


def test_d014_must_energize_serves_all_load_and_excludes_the_nr_window():
    P = toy(**P1A)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    sv = m.spine_values()
    assert all(abs(sv[f"S|{q}"] - P.L(q)) < 1e-9 for q in range(1, P.Q + 1))
    assert sv["z"] == 0                    # a 3-quarter NR window cannot carry a load that must run until connection
    assert not Model(P, fix={"z": 1}).solve_saa(PHAT)["ok"]


def test_d013_same_function_variant_removes_nr_and_is_not_load_bearing_when_nr_is_unused():
    for kw in ({}, P1A):
        m0, m1 = Model(toy(**kw)), Model(toy(same_function_standby=True, **kw))
        s0, s1 = m0.solve_saa(PHAT), m1.solve_saa(PHAT)
        assert s0["ok"] and s1["ok"] and m1.spine_values()["z"] == 0
        if m0.spine_values()["z"] == 0:
            assert primary(s1) == pytest.approx(primary(s0), rel=1e-7)
    m = Model(toy(p_R=5.0, delay_mult=10.0, same_function_standby=True))   # an instance that wants the NR window
    assert m.solve_saa(PHAT)["ok"] and m.spine_values()["z"] == 0


def test_d015_spine_standby_is_emergency_only():
    """Standby ordered on the spine (backup_mode = spine) counts only toward post-connection backup: it can neither
    serve the bridge load nor count in the bridge reserve."""
    P = toy(backup_mode="spine", **P1A)
    fix = {"n|GE": 0, "n|DG": 0, "x|SB|0": 1000.0}
    fix.update({f"x|{j}|{k}": 0.0 for j in ("GE", "DG", "BESS") for k in range(1, P.Q + 1)})
    fix.update({f"RST|{q}": 0.0 for q in range(1, P.Q + 1)})
    assert not Model(P, fix=fix).solve_saa(PHAT)["ok"]
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"] and backup_shortfall(P, m) < 1e-6


def test_int_orders_whole_units_and_cost_not_below_continuous():
    m0, m1 = Model(toy(**P1A)), Model(toy(int_orders=True, **P1A))
    s0, s1 = m0.solve_saa(PHAT), m1.solve_saa(PHAT)
    assert s0["ok"] and s1["ok"]
    sv = m1.spine_values()
    for j, unit in (("GE", m1.P.ge_unit), ("DG", m1.P.dg_unit)):
        for k in range(m1.P.Q + 1):
            n = sv[f"x|{j}|{k}"] / unit
            assert abs(n - round(n)) < 1e-6, (j, k, n)
    assert primary(s1) >= primary(s0) - 1e-7 * abs(primary(s0))


def test_d011_site_specific_bridge_rating_is_a_relaxation():
    s0 = Model(toy(**P1A)).solve_saa(PHAT)
    s1 = Model(toy(dg_bridge_mult=2.5 / 2.1, **P1A)).solve_saa(PHAT)
    assert s0["ok"] and s1["ok"] and primary(s1) <= primary(s0) + 1e-7 * abs(primary(s0))


@pytest.mark.parametrize("allowed,later", [(list(range(1, 9)), False), (None, True), (list(range(1, 9)), True)])
def test_v3_dro_duality_with_stage_bounds_and_directional_transport(allowed, later):
    """Ambiguity-set variants keep exact LP duality."""
    P = toy(**P1A)
    for eps in (0.3, 1.5):
        m = Model(P)
        st = m.solve_dro(PHAT, eps, allowed=allowed, later_only=later)
        assert st["ok"]
        wc = worst_case_expectation(m.C_values(), PHAT, eps, allowed=allowed, later_only=later)
        assert abs(wc - primary(st)) <= 1e-6 * max(1.0, abs(wc)), (eps, wc, primary(st))


def test_dro_rejects_a_centre_outside_the_allowed_states():
    with pytest.raises(ValueError):
        Model(toy()).solve_dro(PHAT, 0.5, allowed=list(range(1, 8)))


def test_pmf_with_masses_below_the_highs_floor_is_rejected_clearly():
    ph = PHAT.copy()
    ph[0] = 1e-11
    with pytest.raises(ValueError):
        Model(toy()).solve_saa(ph / ph.sum())


def test_p1a_job_exception_becomes_a_not_ok_result():
    from experiments.p1a_core import run_job
    r = run_job(dict(econ="main", ctx="A", method="no-such-method"))
    assert r["ok"] is False and "no-such-method" in r["error"]


@pytest.mark.parametrize("flags", ["r2", "p1a"])
def test_v3_dro_duality_with_survival_bands(flags):
    """W1 intersected with survival bands (Phase-1a diagnosis D3) keeps exact LP duality, and the bands
    can only lower the worst case."""
    P = toy(**FLAGSETS[flags])
    S = {q: float(PHAT[q:].sum()) for q in range(1, 9)}
    su = {q: min(1.0, S[q] + 0.05) for q in (5, 6, 7, 8)}
    sl = {q: max(0.0, S[q] - 0.05) for q in (2, 3)}
    for eps in (0.5, 2.0):
        m = Model(P)
        st = m.solve_dro(PHAT, eps, surv_upper=su, surv_lower=sl)
        assert st["ok"]
        wc = worst_case_expectation(m.C_values(), PHAT, eps, surv_upper=su, surv_lower=sl)
        assert abs(wc - primary(st)) <= 1e-6 * max(1.0, abs(wc)), (eps, wc, primary(st))
        assert wc <= worst_case_expectation(m.C_values(), PHAT, eps) + 1e-9


@pytest.mark.parametrize("stage", [(2, 0.5), (3, 0.25), (1, 0.0)])
def test_staged_connection_bookkeeping_and_residual_service(stage):
    """Partial grid capacity for stage_len quarters after T. Bookkeeping must match the independent
    recomputation, grid import must respect the cap, and backup must start at full service."""
    L, frac = stage
    for ctx in ("A", "B"):
        P = toy(ctx, stage_len=L, stage_frac=frac, **P1A)
        m = Model(P)
        assert m.solve_dro(PHAT, 0.8)["ok"]
        assert np.max(np.abs(m.C_values() - recompute_costs(P, m))) < 1e-6
        assert backup_shortfall(P, m) < 1e-6
        for (T, q, b), var in m.v["m"].items():
            if q < T + L:
                assert m.val(var) <= P.G_part + 1e-6


def test_staged_connection_can_only_cost_more_than_immediate_full_service():
    """Partial service is a restriction of full service, apart from the later backup start. With backup starting at T
    in both cases (L = 1 and a zero cap only shift backup by one quarter), the staged optimum is not cheaper than
    full service by more than that quarter's backup carrying cost."""
    s_full = Model(toy(**P1A)).solve_saa(PHAT)
    s_part = Model(toy(stage_len=2, stage_frac=0.5, **P1A)).solve_saa(PHAT)
    assert s_full["ok"] and s_part["ok"]
    assert primary(s_part) >= primary(s_full) - 1.0          # $1M allowance covers the deferred backup quarters


@pytest.mark.parametrize("flags", ["r2", "p1a", "p1a_soft_service", "p1a_refurb_band", "p1d_backup_delay",
                                   "p1d_staged_backup_from_T", "p1d_refurb_stress", "p1d_staged_refurb_spine",
                                   "p1e_rsb", "p1e_rsb_spine", "p1e_rsb_spine_staged"])
def test_cost_components_sum_to_the_scenario_costs(flags):
    from model.bridge import cost_components
    for extra in ({}, dict(stage_len=2, stage_frac=0.5)):
        if extra and flags == "r2":
            continue
        P = toy(**dict(FLAGSETS[flags], **extra))
        m = Model(P)
        assert m.solve_saa(PHAT)["ok"]
        comp = cost_components(P, m)
        assert np.max(np.abs(sum(comp.values()) - recompute_costs(P, m))) < 1e-9


def test_lp_tiebreak_keeps_the_primary_optimum_and_integer_decisions():
    P = toy(**P1A)
    a, b = Model(P), Model(P)
    sa, sb = a.solve_saa(PHAT), b.solve_saa(PHAT, tiebreak_mode="lp")
    assert sa["ok"] and sb["ok"] and sb["tiebreak_ok"] is True
    assert sb["primary_obj"] == pytest.approx(sa["primary_obj"], rel=1e-6)
    assert abs(PHAT @ b.C_values() - sb["primary_obj"]) <= 1e-6 * abs(sb["primary_obj"]) + 1e-6


def test_lp_pure_tiebreak_solves_the_same_lp():
    """D-022: "lp_pure" is the "lp" tie-break solved as a pure LP. Same primary optimum, same tie-break optimum, and the
    integer decisions stay at their fixed integer values."""
    P = toy(**P1A)
    a, b = Model(P), Model(P)
    sa, sb = a.solve_saa(PHAT, tiebreak_mode="lp"), b.solve_saa(PHAT, tiebreak_mode="lp_pure")
    assert sa["tiebreak_ok"] is True and sb["tiebreak_ok"] is True
    assert sb["primary_obj"] == pytest.approx(sa["primary_obj"], rel=1e-9)
    assert sb["obj"] == pytest.approx(sa["obj"], rel=1e-7, abs=1e-6)
    assert all(abs(b.h.val(v) - round(b.h.val(v))) < 1e-9 for v in b._ints)
    assert abs(PHAT @ b.C_values() - sb["primary_obj"]) <= 1e-6 * abs(sb["primary_obj"]) + 1e-6
    with pytest.raises(ValueError):
        Model(P).solve_saa(PHAT, tiebreak_mode="bogus")


def test_tiebreak_without_presolve_keeps_both_optima():
    """D-022 as amended: presolve off in the second stage changes the solver path, not the problem."""
    P = toy(**P1A)
    a, b = Model(P), Model(P)
    sa = a.solve_saa(PHAT, tiebreak_mode="lp")
    sb = b.solve_saa(PHAT, tiebreak_mode="lp", tiebreak_presolve=False)
    assert sa["tiebreak_ok"] is True and sb["tiebreak_ok"] is True
    assert sb["primary_obj"] == pytest.approx(sa["primary_obj"], rel=1e-9)
    assert sb["obj"] == pytest.approx(sa["obj"], rel=1e-7, abs=1e-6)
    v = b.h.getOptionValue("presolve")
    assert (v[-1] if isinstance(v, tuple) else v) == "choose"


def test_p1d_defaults_reproduce_phase1c_backup_timing():
    """backup_delay = -1 means stage_len; an explicit equal delay gives the identical model optimum."""
    for extra in ({}, dict(stage_len=2, stage_frac=0.5)):
        a = Model(toy(**P1A, **extra)).solve_saa(PHAT, tiebreak=False)
        b = Model(toy(**P1A, **extra, backup_delay=extra.get("stage_len", 0))).solve_saa(PHAT, tiebreak=False)
        assert a["ok"] and b["ok"] and b["obj"] == pytest.approx(a["obj"], rel=1e-9, abs=1e-9)


def test_p1d_backup_obligation_start_is_monotone():
    """A later full-backup start relaxes the problem; an earlier one restricts it (B1 cells 01 and 10)."""
    base = Model(toy(**P1A)).solve_saa(PHAT, tiebreak=False)["obj"]
    later = Model(toy(**P1A, backup_delay=2)).solve_saa(PHAT, tiebreak=False)["obj"]
    staged = Model(toy(**P1A, stage_len=2, stage_frac=0.5)).solve_saa(PHAT, tiebreak=False)["obj"]
    staged_T = Model(toy(**P1A, stage_len=2, stage_frac=0.5, backup_delay=0)).solve_saa(PHAT, tiebreak=False)["obj"]
    assert later <= base + 1e-6
    assert staged_T >= staged - 1e-6


def test_p1d_backup_delay_obligation_holds_from_the_stated_quarter():
    P = toy(**P1A, stage_len=2, stage_frac=0.5, backup_delay=0)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    assert backup_shortfall(P, m) < 1e-6
    for T in range(1, P.Q + 1):                                  # obligation from T, inside the partial phase
        q = T
        if P.B_need(q) <= 0:
            continue
        have = sum(m.val(v) for (_, qd, v) in m.bu_orders[T] if qd <= q)
        have += P.bk_ratio * sum(m.val(m.v["y"][T]["DG"][k]) for k in range(T) if k + P.lead["DG"] <= q)
        assert have >= P.B_need(q) - 1e-6


def test_p1d_recommissioning_restricts_backup_counting_and_costs_more():
    """Bridge-run DG counts only after recommissioning, derated; the optimum can only rise, and the independent
    shortfall check applies the same rule."""
    base = Model(toy(**P1A)).solve_saa(PHAT, tiebreak=False)["obj"]
    for extra in (dict(refurb_delay=1), dict(refurb_derate=0.8), dict(dg_refurb=0.05),
                  dict(dg_refurb=0.05, refurb_delay=1, refurb_derate=0.8, stage_len=2, stage_frac=0.5)):
        P = toy(**P1A, **extra)
        m = Model(P)
        st = m.solve_saa(PHAT, tiebreak=False)
        assert st["ok"]
        if "stage_len" not in extra:
            assert st["obj"] >= base - 1e-6, extra
        assert backup_shortfall(P, m) < 1e-6
        assert np.max(np.abs(m.C_values() - recompute_costs(P, m))) < 1e-6


def test_p1d_legacy_structure_rejects_backup_delay_and_recommissioning():
    with pytest.raises(ValueError):
        toy(backup_delay=1)
    with pytest.raises(ValueError):
        toy(refurb_delay=1)


def test_p1e_rsb_is_used_and_can_only_lower_the_optimum():
    """With five-quarter standby purchases, a cheap rented-standby option is used and cannot raise the optimum."""
    base = Model(toy(**P1A, backup_mode="spine")).solve_saa(PHAT, tiebreak=False)
    m = Model(toy(**P1A, backup_mode="spine", rsb_on=True, rsb_price_mult=0.3))
    st = m.solve_saa(PHAT, tiebreak=False)
    assert base["ok"] and st["ok"]
    assert st["obj"] <= base["obj"] + 1e-6
    used = sum(m.val(m.v["RSB"][q]) for q in m.v["RSB"]) + sum(m.val(v) for T in m.v["RSBb"] for v in m.v["RSBb"][T].values())
    assert used > 1e-6


def test_p1e_rsb_contract_inheritance_minimum_commitment_and_horizon():
    P = toy(**P1A, backup_mode="spine", rsb_on=True, rsb_price_mult=0.3)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    val, Q, M = m.val, P.Q, P.rsb_min_q
    for T in range(1, Q + 1):
        rb = {q: (val(m.v["RSB"][T]) if q == T else val(m.v["RSBb"][T][q])) for q in range(T, Q + 1)}
        sb = {q: (val(m.v["RSBs"][q]) if q <= T else val(m.v["RSBbs"][T][q])) for q in range(1, Q + 1)}
        path_r = {q: val(m.v["RSB"][q]) for q in range(1, T)}
        path_r.update(rb)
        for q in range(1, Q + 1):
            prev = path_r.get(q - 1, 0.0)
            assert sb[q] >= path_r[q] - prev - 1e-6, (T, q)
            assert path_r[q] >= sum(sb[j] for j in range(max(1, q - M + 1), q + 1)) - 1e-6, (T, q)
    # the residual rent beyond the horizon: only a start in quarter Q commits Q + 1 (M = 2)
    assert P.rsb_residual(Q) == pytest.approx(P.rent_rsb * P.disc(Q + 1))
    assert P.rsb_residual(Q - 1) == 0.0


def test_p1e_rsb_spine_decisions_do_not_leak_backwards():
    """Raising the rented standby contracted at node k (quarter k + 1) changes no scenario T <= k."""
    P = toy(**P1A, backup_mode="spine", rsb_on=True, rsb_price_mult=0.3)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    spine = m.spine_values()
    base_m, st = polish(P, spine)
    assert st["ok"]
    base = base_m.C_values()
    for k in range(1, P.Q - 1):
        pert = dict(spine)
        for key in (f"RSB|{k + 1}", f"RSBs|{k + 1}", f"RSB|{k + 2}"):
            pert[key] = pert[key] + 1.0                        # a consistent extra 1 MW, 2-quarter contract
        pm, pst = polish(P, pert)
        assert pst["ok"], k
        C = pm.C_values()
        for T in range(1, k + 1):
            assert abs(C[T - 1] - base[T - 1]) < 1e-6, (k, T)
        assert any(abs(C[T - 1] - base[T - 1]) > 1e-6 for T in range(k + 1, P.Q + 2)), k


def test_p1e_rsb_default_off_reproduces():
    a = Model(toy(**P1A, backup_mode="spine")).solve_saa(PHAT, tiebreak=False)
    b = Model(toy(**P1A, backup_mode="spine", rsb_on=False)).solve_saa(PHAT, tiebreak=False)
    assert a["obj"] == pytest.approx(b["obj"], rel=1e-12, abs=1e-12)


def test_policy_detail_reports_decisions_consistently():
    from model.bridge import policy_detail
    P = toy(stage_len=2, stage_frac=0.5, **P1A)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    d = policy_detail(P, m)
    sv = m.spine_values()
    assert all(abs(d["orders"][j][k] - sv[f"x|{j}|{k}"]) < 1e-5 for j in P.assets for k in range(P.Q + 1))
    assert set(d["branches"]) == set(range(1, P.Q + 1))
    for T, b in d["branches"].items():
        assert len(b["staged_rental"]) == min(T + P.stage_len, P.Q + 1) - T
        for j in P.assets:
            assert b["retained"][j] <= sum(d["orders"][j][k] for k in range(T)) + 1e-5


# ---- Phase 1f (docs/phase1f_plan.md): the CCHP ablation flag and operating blocks as a parameter -------------------
def test_p1f_abs_off_builds_and_uses_no_absorption_and_cannot_lower_the_optimum():
    P_on, P_off = toy(**P1A), toy(**P1A, abs_on=False)
    a, m = Model(P_on).solve_saa(PHAT, tiebreak=False), Model(P_off)
    b = m.solve_saa(PHAT, tiebreak=False)
    assert a["ok"] and b["ok"]
    assert max(m.val(v) for v in m.v["x"]["ABS"].values()) == 0.0
    assert max(m.val(v) for v in m.v["a"].values()) <= 1e-9
    assert max(m.val(v) for v in m.v["ba"].values()) <= 1e-9
    assert b["obj"] >= a["obj"] - 1e-6 * max(1.0, abs(a["obj"]))       # removing an option


def test_p1f_abs_default_on_reproduces():
    a = Model(toy(**P1A)).solve_saa(PHAT, tiebreak=False)
    b = Model(toy(**P1A, abs_on=True)).solve_saa(PHAT, tiebreak=False)
    assert a["obj"] == pytest.approx(b["obj"], rel=1e-12, abs=1e-12)


def test_p1f_blocks_default_and_validation():
    P = toy(**P1A)
    assert P.blocks == (0, 1) and P.H * 2 == pytest.approx(91.3125 * 24)
    four = tuple((d, d, n, n) for d, n in P.cop_e)
    with pytest.raises(ValueError):
        toy(**P1A, cop_e=four)                                          # H not halved
    with pytest.raises(ValueError):
        toy(**P1A, cop_e=((7.5, 8.5), (6.0, 7.0, 6.5), (4.5, 5.5), (6.0, 7.0)))   # unequal block counts


@pytest.mark.parametrize("flags", [dict(P1A), dict(P1A, stage_len=2, stage_frac=0.5, backup_mode="spine")])
def test_p1f_four_equal_blocks_reproduce_two_blocks(flags):
    """Splitting each 12-h half into two 6-h blocks with the same COP leaves the model's optimum unchanged."""
    P2 = toy(**flags)
    P4 = toy(**flags, cop_e=tuple((d, d, n, n) for d, n in P2.cop_e), H=P2.H / 2)
    assert P4.blocks == (0, 1, 2, 3) and P4.B_req == pytest.approx(P2.B_req)
    a = Model(P2).solve_saa(PHAT, tiebreak=False)
    b = Model(P4).solve_saa(PHAT, tiebreak=False)
    assert a["ok"] and b["ok"]
    assert b["obj"] == pytest.approx(a["obj"], rel=1e-7, abs=1e-6)


def test_p1f_four_blocks_are_linked_in_the_signal_tree():
    from model.bridge import node_names
    P2 = toy(**P1A)
    P4 = toy(**P1A, cop_e=tuple((d, d, n, n) for d, n in P2.cop_e), H=P2.H / 2)
    names = node_names(P4, 2)
    assert all(f"gGE|2|{b}" in names for b in range(4))


# ---- Phase 1g (docs/phase1g_plan.md): rental carry-over at staged connection (G1) and gas rentals (G2) --------------
STAGED = dict(P1A, stage_len=2, stage_frac=0.5)
GAS = dict(P1A, gr_on=True, p_GR=5.0, gr_cap=20.0)


def test_p1g_defaults_reproduce():
    """Carry-over has no effect without staged service; gas rentals with zero availability change nothing."""
    base = Model(toy(**P1A)).solve_saa(PHAT, tiebreak=False)
    carry = Model(toy(**P1A, stage_rent_carry=True)).solve_saa(PHAT, tiebreak=False)
    none = Model(toy(**P1A, gr_on=True, gr_cap=0.0)).solve_saa(PHAT, tiebreak=False)
    assert base["ok"] and carry["ok"] and none["ok"]
    assert carry["obj"] == pytest.approx(base["obj"], rel=1e-12, abs=1e-9)
    assert none["obj"] == pytest.approx(base["obj"], rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("flags", [dict(STAGED, stage_rent_carry=True), GAS, dict(GAS, **STAGED),
                                   dict(STAGED, stage_rent_carry=True, gr_on=True, p_GR=5.0, gr_cap=20.0),
                                   dict(STAGED, stage_rent_carry=True, backup_mode="spine")])
def test_p1g_bookkeeping_components_and_standby(flags):
    from model.bridge import cost_components
    P = toy(**flags)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    rc = recompute_costs(P, m)
    assert np.max(np.abs(m.C_values() - rc)) < 1e-6
    assert np.max(np.abs(sum(cost_components(P, m).values()) - rc)) < 1e-9
    assert backup_shortfall(P, m) < 1e-6


def test_p1g_carry_uses_the_spine_rental_in_quarter_T_and_new_rentals_after():
    P = toy(**STAGED, stage_rent_carry=True)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    for T in range(1, P.Q + 1):
        assert m.v["brst"][T, T] is m.v["RST"][T]                   # quarter T: the spine's contract
        for q in range(T + 1, min(T + P.stage_len, P.Q + 1)):
            assert m.v["brst"][T, q] is not m.v["RST"][q]            # later staged quarters: new branch contracts
        for b in P.blocks:
            assert m.val(m.v["bgs"][T, T, b]) <= m.val(m.v["RST"][T]) + 1e-6


def test_p1g_carry_only_changes_staged_branches():
    """Without staged service the carry flag builds the same model; with it, only scenarios T <= Q can differ."""
    a = Model(toy(**STAGED))
    b = Model(toy(**STAGED, stage_rent_carry=True))
    sa, sb = a.solve_saa(PHAT, tiebreak=False), b.solve_saa(PHAT, tiebreak=False)
    assert sa["ok"] and sb["ok"]
    spine = a.spine_values()
    pa, _ = polish(toy(**STAGED), spine)
    pb, _ = polish(toy(**STAGED, stage_rent_carry=True), spine)
    assert abs(pa.C_values()[-1] - pb.C_values()[-1]) < 1e-6        # T = Q + 1 has no connection


def test_p1g_gas_rentals_are_used_when_cheap_and_can_only_lower_the_optimum():
    base = Model(toy(**P1A)).solve_saa(PHAT, tiebreak=False)
    m = Model(toy(**GAS))
    st = m.solve_saa(PHAT, tiebreak=False)
    assert base["ok"] and st["ok"]
    assert st["obj"] <= base["obj"] + 1e-6
    assert sum(m.val(m.v["RGR"][q]) for q in m.v["RGR"]) > 1e-6


def test_p1g_gas_rental_contract_cap_permits_and_tail():
    P = toy(**GAS)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    val, Q, M = m.val, P.Q, P.gr_min_q
    r = {q: val(m.v["RGR"][q]) for q in range(1, Q + 1)}
    s = {q: val(m.v["RGRs"][q]) for q in range(1, Q + 1)}
    for q in range(1, Q + 1):
        assert s[q] >= r[q] - r.get(q - 1, 0.0) - 1e-6
        assert r[q] >= sum(s[j] for j in range(max(1, q - M + 1), q + 1)) - 1e-6
        assert r[q] <= P.gr_cap + 1e-6
        assert val(m.v["RNR"][q]) + val(m.v["RST"][q]) + r[q] <= P.cap_R + 1e-6
        filed = [k for k in range(Q) if k <= q - 1 - P.L_permit and round(val(m.v["f"][k])) == 1]
        if not filed:
            assert r[q] <= 1e-6
    # tails: a start in quarter T (M = 2) still owes quarter T + 1 in scenario T; none owed for earlier starts
    assert P.gr_tail(3, 3) == pytest.approx(P.rent_gr * P.disc(4))
    assert P.gr_tail(2, 3) == 0.0
    assert P.gr_tail(Q, Q) == pytest.approx(P.rent_gr * P.disc(Q + 1))


def test_p1g_gas_rental_spine_decisions_do_not_leak_backwards():
    """Raising the gas rental contracted at node k (quarter k + 1) changes no scenario T <= k."""
    P = toy(**GAS)
    m = Model(P)
    assert m.solve_saa(PHAT)["ok"]
    spine = m.spine_values()
    base_m, st = polish(P, spine)
    assert st["ok"]
    base = base_m.C_values()
    for k in range(P.L_permit + 2, P.Q - 1):
        pert = dict(spine)
        for key in (f"RGR|{k + 1}", f"RGRs|{k + 1}", f"RGR|{k + 2}"):
            pert[key] = pert[key] + 1.0                        # a consistent extra 1 MW, 2-quarter contract
        pm, pst = polish(P, pert)
        if not pst["ok"]:                                      # the cap or permits may bind at this node
            continue
        C = pm.C_values()
        for T in range(1, k + 1):
            assert abs(C[T - 1] - base[T - 1]) < 1e-6, (k, T)


def test_p1g_gas_rentals_are_linked_in_the_signal_tree():
    from model.bridge import node_names
    P = toy(**GAS)
    names = node_names(P, 2)
    assert "RGR|3" in names and "RGRs|3" in names and all(f"gGR|2|{b}" in names for b in P.blocks)
