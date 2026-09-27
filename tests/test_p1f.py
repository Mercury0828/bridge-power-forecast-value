"""Phase 1f plumbing (docs/phase1f_plan.md): the per-variant signal law (E-N naive planner, E-T forecast time tau).
- The recomputed registered planner law (tau = 4) equals the frozen Phase-1c law exactly.
- The exact (naive) kernel is an indicator.
- A tau variant moves the pre/post split of the law and keeps its mass."""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments import p1d_core as p1d  # noqa: E402

P1C = ROOT / "experiments" / "r2_runs" / "p1c_main" / "plan.json"


def _cells(n=6):
    plan = json.loads(P1C.read_text(encoding="utf-8"))
    out = []
    for cfg in core.CONFIGS:
        out += [c for c in plan["cells"] if c["cfg"] == cfg][: max(1, n // 4)]
    return plan, out


@pytest.mark.skipif(not P1C.exists(), reason="frozen Phase-1c run not present")
def test_registered_signal_law_reproduces_the_frozen_phase1c_law():
    plan, cells = _cells()
    for c in cells:
        centre = plan["jobs"][c["jobs"]["nosig"]]["centre"]
        frozen = plan["jobs"][c["jobs"]["sig"]]["law"]
        new = p1d.signal_law(centre, c["thresholds"], {})
        assert new["tau"] == core.TAU and new["planner"] == "registered"
        assert np.array_equal(np.asarray(new["law"]["pre"]), np.asarray(frozen["pre"]))
        for a, b in zip(new["law"]["post"], frozen["post"]):
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_exact_kernel_is_an_indicator_of_the_category():
    th = (8, 12)
    K = core.exact_kernel(th)
    assert K.shape == (core.NCAT, len(K[0]))
    assert np.all((K == 0) | (K == 1)) and np.allclose(K.sum(axis=0), 1.0)
    for i in range(K.shape[1]):
        assert K[core.category(i + 1, th), i] == 1.0


@pytest.mark.skipif(not P1C.exists(), reason="frozen Phase-1c run not present")
@pytest.mark.parametrize("tau", [2, 6])
def test_tau_variant_law_keeps_mass_and_moves_the_signal_time(tau):
    plan, cells = _cells(4)
    c = cells[-1]
    centre = np.asarray(plan["jobs"][c["jobs"]["nosig"]]["centre"])
    law = p1d.signal_law(centre, c["thresholds"], dict(tau=tau))["law"]
    pre, post = np.asarray(law["pre"]), [np.asarray(q) for q in law["post"]]
    assert abs(pre.sum() + sum(q.sum() for q in post) - 1.0) < 1e-12
    assert np.all(np.concatenate([q[:tau] for q in post]) == 0)              # no estimate before tau
    assert pre[:tau].sum() == pytest.approx(centre[:tau].sum(), abs=5e-9)      # P(T <= tau) is the pre mass


@pytest.mark.skipif(not P1C.exists(), reason="frozen Phase-1c run not present")
def test_naive_law_differs_from_the_registered_law():
    plan, cells = _cells(4)
    c = cells[-1]
    centre = plan["jobs"][c["jobs"]["nosig"]]["centre"]
    a = p1d.signal_law(centre, c["thresholds"], {})["law"]
    b = p1d.signal_law(centre, c["thresholds"], dict(planner="exact"))["law"]
    assert not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["post"], b["post"]))


def test_four_block_profile_preserves_the_half_means_and_hours():
    from model.bridge import Params
    for ctx in ("A", "B"):
        cop4, H = p1d.cop_four_blocks(ctx)
        for (day, night), s in zip(Params().cop_e, cop4):
            assert 0.5 * (1 / s[0] + 1 / s[1]) == pytest.approx(1 / day, rel=1e-12)
            assert 0.5 * (1 / s[2] + 1 / s[3]) == pytest.approx(1 / night, rel=1e-12)
        P = p1d.params_for(ctx, dict(cop_e=cop4, H=H))
        assert P.blocks == (0, 1, 2, 3) and P.H * 4 == pytest.approx(91.3125 * 24)
        assert P.B_req >= p1d.params_for(ctx, {}).B_req - 1e-9                  # the lowest COP sets the peak


def test_carbon_adders_match_the_declared_factors():
    for ctx, (gas, diesel, grid) in (("A", (3.50, 3.60, 50.0)), ("B", (5.00, 3.70, 75.0))):
        o = p1d._es_co2(ctx, 100.0)
        assert o["gas"] - gas == pytest.approx(100 * 0.05306)
        assert o["diesel"] - diesel == pytest.approx(100 * 0.07396 * 5.772 / 42)
        assert o["grid"] - grid == pytest.approx(100 * p1d.EF_GRID[ctx])


def test_es_variants_change_only_their_parameter_group():
    base = p1d.params_for("B", {})
    lo = p1d.params_for("B", p1d.VARIANTS["s_gas"]["B|gas_lo"][1])
    assert lo.gas == 4.00 and lo.grid == base.grid and lo.diesel == base.diesel and lo.capex == base.capex
    cap = p1d.params_for("B", p1d.VARIANTS["s_capex"]["B|capex_lo"][1])
    assert cap.capex["GE"] == pytest.approx(2.280) and cap.gas == base.gas


# Every Phase-1f parameter variant changes exactly its declared Params fields (full-field comparison, including the
# derived quantities; Phase 1f reading §4, diagnosis of the E-S mismatches).
_DECLARED = {"s_disc": {"disc_year"}, "s_gas": {"gas"}, "s_diesel": {"diesel"}, "s_grid": {"grid"},
             "s_capex": {"capex", "c_backup"}, "s_life": {"life_q"}, "s_rent": {"p_R"},
             "s_eff": {"ge_hr", "dg_gal_per_mwh"}, "s_co2": {"gas", "grid", "diesel"}, "nocchp": {"abs_on"},
             "size075": {"it_ramp", "cap_R"}, "size300": {"it_ramp", "cap_R"}, "blocks4": {"H", "cop_e"},
             # Phase 1g (docs/phase1g_plan.md)
             "carry": {"stage_rent_carry"}, "gasrent_25": {"gr_on"}, "gasrent_35": {"gr_on", "p_GR"},
             "gasrent_50": {"gr_on", "p_GR"}}
_DERIVED = {"gas": {"fuel_ge"}, "ge_hr": {"fuel_ge"}, "diesel": {"fuel_dg"}, "dg_gal_per_mwh": {"fuel_dg"},
            "it_ramp": {"B_req", "G_part"}, "cop_e": {"B_req", "G_part"}}


def _all_fields(P):
    import dataclasses
    d = {f.name: getattr(P, f.name) for f in dataclasses.fields(P)}
    for name in ("B_req", "G_part", "fuel_ge", "fuel_dg"):
        v = getattr(P, name)
        d[name] = v() if callable(v) else v
    return d


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("variant", sorted(_DECLARED))
def test_variant_changes_exactly_its_declared_fields(variant):
    for key, entry in p1d.VARIANTS[variant].items():
        cfg, over = entry[0], entry[1]
        base, var = _all_fields(p1d.params_for(cfg, {})), _all_fields(p1d.params_for(cfg, over))
        changed = {k for k in base if not _same(base[k], var[k])}
        allowed = set(_DECLARED[variant]).union(*(_DERIVED.get(f, set()) for f in _DECLARED[variant]))
        assert changed and changed <= allowed, (key, changed)
