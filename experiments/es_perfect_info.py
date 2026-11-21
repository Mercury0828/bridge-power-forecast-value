"""Perfect-information reference of the Energy Systems paper (Section 3.4, Table 7).

For each configuration and connection quarter t, solve the deterministic path model that knows T = t from quarter 0
(`Model(..., only_T=t)`: only branch t, no service requirement in spine quarters q >= t), then compute
WS = E[C*(T)], EVPI = v0 - WS and U = N - WS against the frozen v0, N, v_Y, Delta and V_true.

Run:  python experiments/es_perfect_info.py
Writes data/es_perfect_info.json. Solves sequentially (one HiGHS thread). A partial file keeps finished solves; only
optimal solves of the current model version are reused, every other entry is solved again. Aggregation runs only if
every solve is optimal; the script exits non-zero (and marks the output STOP) if a falsifier fails.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import p1c_core as core  # noqa: E402
from experiments.p1d_accounting import _plan_items, eval_law, marginal  # noqa: E402
from model import evidence as ev  # noqa: E402
from model.bridge import Model  # noqa: E402

OUT = ROOT / "data" / "es_perfect_info.json"
PART = ROOT / "data" / "es_perfect_info.partial.json"
VERSION = "path_model_v1.3"
FAMS = ("sym1", "sym2", "sym4", "opt2", "uninf")
TOL = 1e-5


def solve_all():
    done = json.loads(PART.read_text(encoding="utf-8")) if PART.exists() else {}
    for cfg in core.CONFIGS:
        P = core.params(cfg)
        for t in range(1, P.Q + 2):
            key = f"{cfg}|{t}"
            old = done.get(key)
            if old and old.get("ok") and old.get("version") == VERSION:
                continue
            t0 = time.time()
            m = Model(P, gap=TOL, time_limit=1800, only_T=t)
            st = m.solve_det(t, tiebreak=False)
            info = m.h.getInfo()
            done[key] = dict(version=VERSION, ok=bool(st["ok"]), status=st["status"], obj=float(st["obj"]),
                             dual_bound=float(info.mip_dual_bound), gap=float(st["gap"]), seconds=time.time() - t0)
            PART.write_text(json.dumps(done, indent=1), encoding="utf-8")
            print(key, done[key]["status"], round(done[key]["obj"], 3), round(done[key]["seconds"], 1), flush=True)
    return done


def main():
    sol = solve_all()
    bad = [k for k, v in sol.items() if not (v["ok"] and v.get("version") == VERSION)]
    if bad:
        OUT.write_text(json.dumps(dict(STOP="non-optimal solves", bad=bad, solves=sol), indent=1), encoding="utf-8")
        print("STOP: non-optimal solves", bad)
        sys.exit(2)

    v0 = {(r["cfg"], r["level"]): r for r in json.loads((ROOT / "data/p1d/v0.json").read_text(encoding="utf-8"))}
    dec = json.loads((ROOT / "data/p1d/oracle_decomposition.json").read_text(encoding="utf-8"))["decomposition"]
    rows = json.loads((ROOT / "data/p1c_rows.json").read_text(encoding="utf-8"))
    plan, _ = _plan_items()
    cells = {(c["cfg"], c["level"], c["r"]): c for c in plan["cells"]}

    res, checks = {}, []
    for cfg in core.CONFIGS:
        P = core.params(cfg)
        ts = range(1, P.Q + 2)
        Cs = np.array([sol[f"{cfg}|{t}"]["obj"] for t in ts])
        Ls = np.array([sol[f"{cfg}|{t}"]["dual_bound"] for t in ts])
        by = {}
        for lv in core.LEVELS:
            # Statistics 1-2: the true law of T used by v0, and v0 = J_train_law
            p = ev.clean(core.truth(cfg, lv, 0))
            C0 = np.array(v0[cfg, lv]["C"])
            for t in ts:  # spot check: knowing T cannot cost more than the frozen true-law no-forecast plan in scenario t
                checks.append(dict(kind="Cstar_le_C0", cfg=cfg, level=lv, t=t, diff=float(Cs[t - 1] - C0[t - 1]),
                                   ok=bool(Cs[t - 1] <= C0[t - 1] + TOL * abs(C0[t - 1]))))
            WS, WSL = float(p @ Cs), float(p @ Ls)
            v0l = v0[cfg, lv]["J_train_law"]
            row = dict(WS=WS, WS_lower=WSL, v0=v0l, EVPI=v0l - WS, EVPI_cert_upper=v0l - WSL)
            checks.append(dict(kind="WS_le_v0", cfg=cfg, level=lv, diff=WS - v0l, ok=bool(WS <= v0l + TOL * abs(v0l))))
            # Per family, exact: average C* under each report's cleaned evaluation marginal, matched to N, v0, v_Y
            for fam in FAMS:
                R = [r for r in rows if r["cfg"] == cfg and r["family"] == fam and r["shift"] == 0
                     and r["level"] == lv and "J_oracle" in r]
                w = np.array([r["p"] for r in R])
                w = w / w.sum()
                margs = [marginal(eval_law(cells[cfg, lv, r["r"]]["thresholds"], cfg, lv, fam)) for r in R]
                WSf = float(w @ [mg @ Cs for mg in margs])
                WSLf = float(w @ [mg @ Ls for mg in margs])
                d = dec[f"{cfg}|{fam}"]["by_level"][lv]
                U, Uc = d["N"] - WSf, d["N"] - WSLf
                EV = d["v0"] - WSf
                row[fam] = dict(WS=WSf, WS_lower=WSLf, N=d["N"], v0=d["v0"], vY=d["vY"], d_info=d["d_info"],
                                V_true=d["V_true"], EVPI=EV, U=U, U_cert=Uc,
                                d_over_U=d["d_info"] / U if U else None, Vtrue_over_EVPI=d["V_true"] / EV if EV else None)
                checks.append(dict(kind="WS_le_vY", cfg=cfg, level=lv, fam=fam, diff=WSf - d["vY"],
                                   ok=bool(WSf <= d["vY"] + TOL * abs(d["vY"]))))
                checks.append(dict(kind="WS_le_v0_fam", cfg=cfg, level=lv, fam=fam, diff=WSf - d["v0"],
                                   ok=bool(WSf <= d["v0"] + TOL * abs(d["v0"]))))
            by[lv] = row
        mean = {k: float(np.mean([by[lv][k] for lv in core.LEVELS])) for k in ("WS", "WS_lower", "v0", "EVPI",
                                                                                  "EVPI_cert_upper")}
        for fam in FAMS:
            mf = {k: float(np.mean([by[lv][fam][k] for lv in core.LEVELS]))
                  for k in ("WS", "N", "v0", "vY", "d_info", "V_true", "EVPI", "U", "U_cert")}
            mf["d_over_U"] = mf["d_info"] / mf["U"] if mf["U"] else None
            mf["Vtrue_over_EVPI"] = mf["V_true"] / mf["EVPI"] if mf["EVPI"] else None
            mean[fam] = mf
        res[cfg] = dict(by_level=by, mean=mean)

    ok = all(c["ok"] for c in checks)
    out = dict(model_version=VERSION, all_optimal=True,
               falsifier_all_ok=ok, failed_checks=[c for c in checks if not c["ok"]], checks=checks,
               solves=sol, results=res)
    if not ok:
        out["STOP"] = "falsifier violated"
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("all_optimal True", "falsifier_ok", ok, "max solve s", round(max(v["seconds"] for v in sol.values()), 1))
    for cfg, r in res.items():
        m = r["mean"]
        print(cfg, {k: round(m[k], 3) for k in ("WS", "v0", "EVPI")},
              {f: {k: (round(v, 3) if v is not None else None) for k, v in m[f].items()
                   if k in ("d_info", "V_true", "EVPI", "U", "d_over_U", "Vtrue_over_EVPI")} for f in ("sym1", "sym2")})
    if not ok:
        print("STOP: falsifier violated", out["failed_checks"][:5])
        sys.exit(3)


if __name__ == "__main__":
    main()
