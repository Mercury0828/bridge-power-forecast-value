"""Face-value fallback kernel and the two variants naive_eps and s_salv."""
import json
import pathlib

import numpy as np
import pytest

from experiments import p1c_core as core
from experiments import p1d_core as d

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_eps_kernel_is_column_stochastic_and_keeps_all_histories():
    th = (9, 10)
    for eps in (0.01, 0.05):
        K = core.eps_kernel(th, eps)
        assert np.allclose(K.sum(axis=0), 1.0)
        # every quarter the broad kernel can map into the one-quarter middle category keeps positive weight
        assert np.all(K[1][core.planner_kernel(th)[1] > 0] > 0)
    assert np.allclose(core.eps_kernel(th, 0.0), core.exact_kernel(th))


def test_earlier_variant_job_keys_unchanged():
    p = ROOT / "experiments" / "r2_runs" / "p1f_naive" / "plan.json"
    if not p.exists():
        pytest.skip("stored runs not unzipped (runs.zip)")
    stored = json.loads(p.read_text(encoding="utf-8"))
    _, jobs = d.plan("naive")
    assert set(jobs) == set(stored["jobs"])


def test_new_variants_sizes_and_settings():
    _, jobs = d.plan("naive_eps")
    assert len(jobs) == 280
    eps = sorted({j["eps"] for j in jobs.values() if j.get("planner") == "eps"})
    assert eps == [0.01, 0.05]
    _, jobs = d.plan("s_salv")
    assert len(jobs) == 352
    P = d.params_for("B", d.VARIANTS["s_salv"]["B|salv_lo"][1])
    assert P.age_salvage and abs(P.salv["GE"] - 0.25) < 1e-12 and abs(P.sig_ret("GE", 0) - 0.25) < 1e-12
