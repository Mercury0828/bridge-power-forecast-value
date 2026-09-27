"""BRIDGE-CDRO Phase-0 stylized model (R2), implementing data/r2_preregistration.md section 1 exactly.

Structure
---------
* Spine nodes m_0..m_Q carry every pre-connection decision; scenario T (grid available from quarter T) uses nodes
  m_0..m_{T-1}. Non-anticipativity therefore holds by construction.
* Connection node c_T and post-connection quarters T..Q carry scenario-specific decisions.
* Scenario cost C_T is a linear expression. The methods differ only in the objective over {C_T}: Wasserstein-1 DRO
  (dual), SAA, or deterministic at one scenario.
* `recompute_costs` is an independent numpy re-implementation of C_T from decision values. It is used to check the
  MILP bookkeeping (V2) and to evaluate policies after branch polishing.

Phase-1a economics (D-011..D-015) are `Params` flags whose defaults reproduce R2 bit for bit:
* `backup_rating` "DCC": retained DG counts toward backup at its DCC rating; purchased standby is priced per DCC MW.
* `backup_sym`: purchased standby pays fixed O&M from delivery and has terminal value, like a kept unit (no double
  credit for retention).
* `age_salvage`, `salv_mult`: one straight-line depreciation curve from the resale value `salv` over `life_q` for
  retirement at T and terminal value at Q+1, with a band multiplier. Wear from bridge hours is in DG VOM.
* `backup_ramp` / `backup_mode`: the requirement holds in every post-connection quarter with IT load (ramp-following
  if set); "spine" mode orders standby (asset SB) non-anticipatively with the DG lead time.
* `service_hard`: must-energize (S_q = L_q). The NR regime is then infeasible: a window of at most 3 quarters cannot
  carry load that must run until connection.
* `dg_bridge_mult`, `same_function_standby`, `int_orders`, `dg_refurb`, `vod_base`: sensitivities and audits.
* Phase 1d. The defaults reproduce Phase 1c:
  * `backup_delay`: the full-backup requirement starts at T + backup_delay. The default -1 means `stage_len`.
  * `refurb_delay`, `refurb_derate`, `dg_refurb`: recommissioning of bridge-run DG. These are cohorts delivered by the
    last bridge-service quarter t_b = T - 1 + stage_len. They count toward backup only from t_b + 1 + refurb_delay, at
    refurb_derate x the counting ratio. The cost dg_refurb ($M/MW-COP) is charged at t_b + 1 on their retained MW.
* Phase 1e E1 (docs/phase1e_plan.md v2): `rsb_on` adds rented standby (RSB), a separate standby rental fleet counted
  toward the full-backup requirement (emergency use only; never in bridge dispatch or reserve).
  * Capacity for quarter q is contracted at node q - 1 (1-quarter mobilization). Spine RSB|q and RSBs|q (the starts)
    belong to node q - 1, and are therefore linked across signal copies before tau.
  * Branch T inherits RSB_T and RSBs_T and contracts quarters T + 1..Q after observing T. Every path carries a
    minimum commitment of rsb_min_q quarters.
  * Rent is charged per quarter where decided, and installation per start. A start whose commitment runs past Q is
    charged its residual rent at disc(t) for t > Q.

Units: MW, MWh, $M. Quarter index q = 1..Q; spine node index k = 0..Q (node m_k is the start of quarter k, T > k known;
m_0 is the planning epoch).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import highspy
import numpy as np

ASSETS = ("GE", "DG", "ABS", "BESS")
BLOCKS = (0, 1)  # day, night


# ----------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Params:
    ctx: str = "A"
    Q: int = 28
    H: float = 1095.75                        # hours per block per quarter (91.3125 d x 12 h)
    it_ramp: tuple = ((5, 50.0), (9, 100.0), (13, 150.0))   # (first quarter, MW-IT)
    a_oth: float = 0.08
    kappa: float = 1.0
    cop_e: tuple = ((7.5, 8.5), (6.0, 7.0), (4.5, 5.5), (6.0, 7.0))  # (day, night) winter, spring, summer, fall
    # gas reciprocating CHP (DOE CHP fact sheet 2023, 4.5 MW lean burn)
    ge_unit: float = 4.455
    ge_hr: float = 8.404                       # MMBtu/MWh (HHV)
    eta_h: float = 0.948                       # MWth recovered per MWe
    # diesel genset (Cummins DQLH, Tier 4F): 2.1 MW continuous, 2.75 MW standby
    dg_unit: float = 2.1
    dg_esp_ratio: float = 2.75 / 2.1
    dg_gal_per_mwh: float = 70.0
    cop_abs: float = 0.70
    capex: dict = field(default_factory=lambda: {"GE": 2.6625, "DG": 2.0 / 2.1, "ABS": 1.590 * 1.1 / 3.517,
                                                 "BESS": 0.436 * 2 * 1.05})         # $M per MW (MWth for ABS)
    fom_q: dict = field(default_factory=lambda: {"GE": 0.0, "DG": 0.010 / 4 / 2.1, "ABS": 0.0,
                                                 "BESS": 0.040 / 4})                 # $M per MW per quarter
    vom: dict = field(default_factory=lambda: {"GE": 19.375, "DG": 10.0})            # $/MWh
    abs_maint: float = 0.005 * 1.1 * 1e6 / 3.517 / 1e6 * 1000 / 1.0                 # placeholder, set in __post_init__
    lead: dict = field(default_factory=lambda: {"GE": 6, "DG": 5, "ABS": 3, "BESS": 3})
    salv: dict = field(default_factory=lambda: {"GE": 0.50, "DG": 0.60, "ABS": 0.30, "BESS": 0.60})
    salv_term: dict = field(default_factory=lambda: {"GE": 0.40, "DG": 0.50, "ABS": 0.20, "BESS": 0.40})
    r_reserve: float = 0.10
    c_backup: float = 0.600                    # $M per MW-ESP
    backup_factor: float = 1.10
    c_file: float = 0.5                        # $M per permit filing
    st_premium: float = 1.25
    disc_year: float = 0.08
    # context-specific (set by make_params)
    gas: float = 3.50
    grid: float = 50.0
    diesel: float = 3.60
    L_permit: int = 4
    # sweep
    p_R: float = 25.0                          # $/kW-month
    cap_R: float = 100.0                       # MW
    delay_mult: float = 1.0
    residence_on: bool = True
    # derived
    int_root: bool = True
    # ---- Phase 1a economics (D-011..D-015). The defaults reproduce R2 exactly. -----------------
    backup_rating: str = "ESP"                 # rating at which DG counts toward emergency backup: "ESP" (R2) | "DCC"
    dg_dcc_ratio: float = 2.5 / 2.1            # DQLH data-center-continuous / continuous (D-012)
    backup_sym: bool = False                   # D-015: purchased backup pays DG fixed O&M and gets terminal value
    backup_mode: str = "instant"               # "instant": bought when needed | "spine": ordered with DG lead time
    backup_ramp: bool = False                  # requirement follows the IT ramp in each post-connection quarter
    age_salvage: bool = False                  # D-015: sigma_j(age) = salv_mult * salv_j * max(0, 1 - age / life_j)
    life_q: dict = field(default_factory=lambda: {"GE": 80, "DG": 80, "ABS": 92, "BESS": 60})   # quarters
    salv_mult: float = 1.0                     # D-015 salvage band multiplier
    dg_refurb: float = 0.0                     # $M per MW-COP of bridge-run DG retained as standby (recommissioning)
    service_hard: bool = False                 # D-014 must-energize: S_q = L_q in every pre-connection quarter
    vod_base: float = 157.0 * 3 * 1000 / 1e6   # $M per MW-IT-quarter before delay_mult (R2: P6 rent)
    dg_bridge_mult: float = 1.0                # D-011 sensitivity: bridge dispatch rating per MW-COP (2.5/2.1 = DCC)
    same_function_standby: bool = False        # D-013 variant: standby is the bridge function, so no NR regime
    int_orders: bool = False                   # every GE/DG order in whole units (procurement audit)
    # ---- staged grid connection. Default = full service from T (R2 / Phase 1a). ------
    stage_len: int = 0                         # quarters of partial service after initial energization T
    stage_frac: float = 1.0                    # partial grid capacity as a fraction of peak facility demand
    # ---- Phase 1d (docs/phase1d_plan.md v2). Defaults reproduce Phase 1c. ------------------------------------------
    backup_delay: int = -1                     # full backup from T + backup_delay; -1 = stage_len (Phase 1c)
    refurb_delay: int = 0                      # quarters after bridge service before bridge-run DG counts as backup
    refurb_derate: float = 1.0                 # counting-rating multiplier for recommissioned bridge-run DG
    # ---- Phase 1e E1: rented standby (RSB). Default off = Phase 1c/1d. ---------------------------------------------
    rsb_on: bool = False
    rsb_cap: float = 100.0                     # counted MW; a separate standby rental fleet (not the bridge NR/ST cap)
    rsb_price_mult: float = 1.0                # rent = rsb_price_mult * p_R ($/kW-month) per counted MW
    rsb_install: float = 15.0                  # $/kW per started counted MW (installation and mobilization)
    rsb_min_q: int = 2                         # minimum commitment per start, quarters
    # ---- Phase 1f (docs/phase1f_plan.md). Defaults reproduce Phase 1c. ---------------------------------------------
    abs_on: bool = True                        # E-C ablation: False forbids absorption chillers (electric cooling only)
    # ---- Phase 1g (docs/phase1g_plan.md). Defaults reproduce Phase 1c. ---------------------------------------------
    stage_rent_carry: bool = False             # G1: the spine's quarter-T ST rental serves staged branch T in quarter T;
                                               #     new staged rentals exist from T + 1 (one-quarter mobilization)
    gr_on: bool = False                        # G2: rented gas gensets (GR) for bridge supply before connection
    gr_cap: float = 50.0                       # MW of GR available, inside the site rental cap cap_R
    p_GR: float = 25.0                         # $/kW-month on the contracted level (O&M included)
    gr_min_q: int = 2                          # minimum commitment per GR start, quarters

    def __post_init__(self):
        # absorption maintenance: 0.5 c/ton-hr x 1.1; 1 MWh_th = 1000/3.517 = 284.33 ton-hr -> $/MWh_th
        self.abs_maint = 0.005 * 1.1 * (1000.0 / 3.517)
        # operating blocks per quarter = entries per season in cop_e (Phase 1c: day, night); equal-length blocks
        if len({len(c) for c in self.cop_e}) != 1:
            raise ValueError("every season needs the same number of operating blocks in cop_e")
        if abs(self.H * len(self.cop_e[0]) - 91.3125 * 24) > 1e-6:
            raise ValueError("H x blocks must equal the hours of a quarter (91.3125 d x 24 h)")
        if self.backup_rating not in ("ESP", "DCC"):
            raise ValueError(self.backup_rating)
        if self.backup_mode not in ("instant", "spine"):
            raise ValueError(self.backup_mode)
        if self.stage_len > 0 and self.backup_mode == "instant" and not self.backup_ramp:
            raise ValueError("staged connection needs the per-quarter backup formulation (backup_ramp or spine)")
        if (self.bk_start > 0 or self.refurb_active) and self.backup_mode == "instant" and not self.backup_ramp:
            raise ValueError("backup_delay / recommissioning need the per-quarter backup formulation")
        if self.rsb_on and self.backup_mode == "instant" and not self.backup_ramp:
            raise ValueError("rented standby needs the per-quarter backup formulation")
        if self.backup_mode == "spine":
            # Standby gensets ordered on the spine (asset SB, counted MW): emergency-only, never dispatched for the
            # bridge and not counted in the bridge reserve. Same machine class as purchased backup.
            for name in ("capex", "fom_q", "lead", "salv", "salv_term", "life_q"):
                setattr(self, name, dict(getattr(self, name)))
            self.capex["SB"], self.fom_q["SB"] = self.c_bu, self.fom_bu
            self.lead["SB"], self.salv["SB"] = self.lead["DG"], self.salv["DG"]
            self.salv_term["SB"], self.life_q["SB"] = self.salv_term["DG"], self.life_q["DG"]

    # --- derived helpers -------------------------------------------------------------------------------------
    @property
    def blocks(self):                          # operating blocks per quarter (Phase 1c: (0, 1) = day, night)
        return tuple(range(len(self.cop_e[0])))

    @property
    def delta(self):
        return (1.0 + self.disc_year) ** (-0.25)

    def disc(self, q):
        return self.delta ** q

    def L(self, q):
        v = 0.0
        for q0, mw in self.it_ramp:
            if q >= q0:
                v = mw
        return v

    def cop(self, q, b):
        return self.cop_e[(q - 1) % 4][b]

    @property
    def fuel_ge(self):                         # $/MWh_e
        return self.ge_hr * self.gas

    @property
    def fuel_dg(self):                         # $/MWh_e
        return self.dg_gal_per_mwh * self.diesel

    @property
    def rent_nr(self):                         # $M per MW-quarter
        return self.p_R * 3 * 1000 / 1e6

    @property
    def rent_st(self):
        return self.st_premium * self.rent_nr

    @property
    def vod(self):                             # $M per MW-IT-quarter
        return self.vod_base * self.delay_mult

    @property
    def B_req(self):                           # MW at the backup counting rating
        Lmax = max(mw for _, mw in self.it_ramp)
        peak = Lmax * (1 + self.a_oth) + self.kappa * Lmax / min(min(c) for c in self.cop_e)
        return self.backup_factor * peak

    @property
    def G_part(self):                          # MW of grid import during the partial-service period
        Lmax = max(mw for _, mw in self.it_ramp)
        return self.stage_frac * (Lmax * (1 + self.a_oth) + self.kappa * Lmax / min(min(c) for c in self.cop_e))

    @property
    def bk_start(self):                        # quarters after T at which the full backup requirement starts
        return self.stage_len if self.backup_delay < 0 else self.backup_delay

    @property
    def refurb_active(self):
        return self.dg_refurb > 0 or self.refurb_delay > 0 or self.refurb_derate != 1.0

    def dg_bridge_last(self, T):               # last quarter in which DG can have served bridge load, scenario T
        return T - 1 + self.stage_len

    @property
    def rent_rsb(self):                        # $M per counted MW-quarter
        return self.rsb_price_mult * self.p_R * 3 * 1000 / 1e6

    @property
    def inst_rsb(self):                        # $M per started counted MW
        return self.rsb_install * 1000 / 1e6

    def rsb_residual(self, q):                 # $M per MW started in quarter q: discounted rent committed beyond Q
        return self.rent_rsb * sum(self.disc(t) for t in range(self.Q + 1, q + self.rsb_min_q))

    @property
    def rent_gr(self):                         # $M per MW-quarter of rented gas gensets
        return self.p_GR * 3 * 1000 / 1e6

    def gr_tail(self, q, after):
        """$M per MW started in quarter q: discounted rent for the committed quarters later than `after` (the quarters
        q..q + gr_min_q - 1 that no spine level of the scenario pays for; beyond Q they are still owed)."""
        return self.rent_gr * sum(self.disc(t) for t in range(max(after + 1, q), q + self.gr_min_q))

    def B_need(self, q):
        """Backup required in post-connection quarter q (general formulation): none before IT load; the full-ramp
        requirement B_req, or with backup_ramp the requirement of the load present in q."""
        Lq = self.L(q)
        if Lq <= 0:
            return 0.0
        if not self.backup_ramp:
            return self.B_req
        return self.backup_factor * (Lq * (1 + self.a_oth) + self.kappa * Lq / min(min(c) for c in self.cop_e))

    @property
    def assets(self):
        return ASSETS + ("SB",) if self.backup_mode == "spine" else ASSETS

    @property
    def bk_ratio(self):                        # counted backup MW per MW-COP of DG
        return self.dg_esp_ratio if self.backup_rating == "ESP" else self.dg_dcc_ratio

    @property
    def c_bu(self):                            # $M per counted MW of purchased standby (same machine, re-rated)
        return self.c_backup if self.backup_rating == "ESP" else self.c_backup * self.dg_esp_ratio / self.dg_dcc_ratio

    @property
    def fom_bu(self):                          # $M per counted MW per quarter (DG per-unit fixed O&M)
        return self.fom_q["DG"] / self.bk_ratio

    def sig_ret(self, j, age_q):
        """Resale value (fraction of capex) of cohort j of age age_q quarters when retired at connection."""
        if not self.age_salvage:
            return self.salv[j] * self.salv_mult
        return self.salv_mult * self.salv[j] * max(0.0, 1.0 - max(0.0, age_q) / self.life_q[j])

    def sig_term(self, j, age_q):
        """Terminal value (fraction of capex) at Q+1. With age_salvage it is the same depreciation curve as sig_ret."""
        if not self.age_salvage:
            return self.salv_term[j] * self.salv_mult
        return self.salv_mult * self.salv[j] * max(0.0, 1.0 - max(0.0, age_q) / self.life_q[j])

    @property
    def bigM(self):
        Lmax = max(mw for _, mw in self.it_ramp)
        return 1.5 * (Lmax * (1 + self.a_oth) + self.kappa * Lmax / min(min(c) for c in self.cop_e))


def make_params(ctx: str, p_R=25.0, cap_R=100.0, delay_mult=1.0, residence_on=True, **over) -> Params:
    if ctx == "A":
        base = dict(ctx="A", gas=3.50, grid=50.0, diesel=3.60, L_permit=4)
    elif ctx == "B":
        base = dict(ctx="B", gas=5.00, grid=75.0, diesel=3.70, L_permit=5)
    else:
        raise ValueError(ctx)
    base.update(dict(p_R=p_R, cap_R=cap_R, delay_mult=delay_mult, residence_on=residence_on))
    base.update(over)
    return Params(**base)


def _check_pmf(phat):
    """Masses in (1e-12, 1e-9) become objective and tie-break-row coefficients below HiGHS's small_matrix_value, and
    adding such a row fails. Reject them with a clear message; model.evidence.clean floors pmfs at 1e-9."""
    p = np.asarray(phat, dtype=float)
    if np.any((p > 1e-12) & (p < 1e-9)):
        raise ValueError("pmf has masses in (1e-12, 1e-9); floor it first (model.evidence.clean)")


# ----------------------------------------------------------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------------------------------------------------------
class Model:
    """Builds the joint spine + branch model with scenario-cost variables C[T], T = 1..Q+1."""

    def __init__(self, P: Params, fix: dict | None = None, time_limit=600.0, gap=1e-4, threads=1, h=None, prefix=""):
        """`h` and `prefix` let several policy copies share one HiGHS instance (the signal tree, SignalModel). By
        default each Model owns its instance, as before."""
        self.P = P
        self.prefix = prefix
        if h is None:
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("mip_rel_gap", gap)
            h.setOptionValue("time_limit", float(time_limit))
            h.setOptionValue("threads", int(threads))
        self.h = h
        self.fix = fix or {}
        self._ints = []                 # integer variables, for the LP tie-break (fix_integers)
        self._build()

    # small helpers -------------------------------------------------------------------------------------------
    def _var(self, lb=0.0, ub=highspy.kHighsInf, integer=False, name=None):
        t = highspy.HighsVarType.kInteger if integer else highspy.HighsVarType.kContinuous
        if name is not None and self.prefix:
            name = self.prefix + name
        var = self.h.addVariable(lb=lb, ub=ub, type=t, name=name)
        if integer:
            self._ints.append(var)
        return var

    def fix_integers(self, relax=False):
        """Fix every integer variable at its current (rounded) value. The model becomes an LP in the rest; used by the
        LP tie-break. With `relax`, the fixed columns are also marked continuous, so HiGHS solves a pure LP instead of a
        MIP whose integer columns are all fixed. The problem is the same (owner decision D-022: the MIP path returned
        kSolveError on one Phase-1c job)."""
        for var in self._ints:
            v = float(round(self.h.val(var)))
            self.h.changeColBounds(var.index, v, v)
        if relax and self._ints:
            idx = np.array([var.index for var in self._ints], dtype=np.int32)
            self.h.changeColsIntegrality(len(idx), idx,
                                         np.array([highspy.HighsVarType.kContinuous] * len(idx)))

    def _build(self):
        P, h = self.P, self.h
        Q = P.Q
        inf = highspy.kHighsInf
        self.v = v = {}
        # ---- spine variables ------------------------------------------------------------------------------
        # orders x[j][k], k = 0..Q
        v["x"] = {j: {} for j in P.assets}
        v["n"] = {}
        v["nk"] = {}
        for j in P.assets:
            for k in range(Q + 1):
                if k == 0 and j in ("GE", "DG") and P.int_root:
                    n = self._var(0, 200, integer=True, name=f"n_{j}")
                    v["n"][j] = n
                    unit = P.ge_unit if j == "GE" else P.dg_unit
                    x = self._var(0, inf, name=f"x_{j}_0")
                    h.addConstr(x - unit * n == 0)
                    v["x"][j][k] = x
                elif k >= 1 and j in ("GE", "DG") and P.int_orders:
                    n = self._var(0, 200, integer=True, name=f"n_{j}_{k}")
                    v["nk"][j, k] = n
                    unit = P.ge_unit if j == "GE" else P.dg_unit
                    x = self._var(0, inf, name=f"x_{j}_{k}")
                    h.addConstr(x - unit * n == 0)
                    v["x"][j][k] = x
                else:
                    ub = 0.0 if (j == "ABS" and not P.abs_on) else inf        # Phase 1f E-C: no absorption chillers
                    v["x"][j][k] = self._var(0, ub, name=f"x_{j}_{k}")
        # rentals for quarter q = 1..Q (decided at node q-1)
        v["RNR"] = {q: self._var(0, P.cap_R, name=f"RNR_{q}") for q in range(1, Q + 1)}
        v["RST"] = {q: self._var(0, P.cap_R, name=f"RST_{q}") for q in range(1, Q + 1)}
        if P.rsb_on:
            # Phase 1e E1: rented standby for quarter q, contracted at node q-1 (levels and starts), min commitment
            v["RSB"] = {q: self._var(0, P.rsb_cap, name=f"RSB_{q}") for q in range(1, Q + 1)}
            v["RSBs"] = {q: self._var(0, P.rsb_cap, name=f"RSBs_{q}") for q in range(1, Q + 1)}
            for q in range(1, Q + 1):
                prev = v["RSB"][q - 1] if q >= 2 else 0.0
                h.addConstr(v["RSBs"][q] - v["RSB"][q] + prev >= 0)
                h.addConstr(v["RSB"][q] - sum(v["RSBs"][j] for j in range(max(1, q - P.rsb_min_q + 1), q + 1)) >= 0)
        if P.gr_on:
            # Phase 1g G2: rented gas gensets for quarter q, contracted at node q-1 (levels and starts), min commitment
            v["RGR"] = {q: self._var(0, P.gr_cap, name=f"RGR_{q}") for q in range(1, Q + 1)}
            v["RGRs"] = {q: self._var(0, P.gr_cap, name=f"RGRs_{q}") for q in range(1, Q + 1)}
            v["gGR"] = {(q, b): self._var(0, inf, name=f"gGR_{q}_{b}") for q in range(1, Q + 1) for b in P.blocks}
            for q in range(1, Q + 1):
                prev = v["RGR"][q - 1] if q >= 2 else 0.0
                h.addConstr(v["RGRs"][q] - v["RGR"][q] + prev >= 0)
                h.addConstr(v["RGR"][q] - sum(v["RGRs"][j] for j in range(max(1, q - P.gr_min_q + 1), q + 1)) >= 0)
        # permit filing f[k], k = 0..Q-1 (binary)
        v["f"] = {k: self._var(0, 1, integer=True, name=f"f_{k}") for k in range(Q)}
        # NR regime
        if P.residence_on:
            # D-013 same-function variant: post-connection standby engines would join the bridge function at the
            # location, so the combined residence always exceeds 12 months once standby exists; no NR regime.
            v["z"] = self._var(0, 0 if P.same_function_standby else 1, integer=True, name="zNR")
            v["w"] = {q: self._var(0, 1, integer=True, name=f"w_{q}") for q in range(1, Q + 1)}
            v["vs"] = {q: self._var(0, 1, name=f"vs_{q}") for q in range(1, Q + 1)}
        # pre-connection dispatch in quarter q = 1..Q (at node m_q); D-014 must-energize fixes S_q = L_q
        v["S"] = {q: self._var(P.L(q) if P.service_hard else 0, P.L(q), name=f"S_{q}") for q in range(1, Q + 1)}
        for key in ("gGE", "gDG", "gNR", "gST", "a", "e"):
            v[key] = {(q, b): self._var(0, inf, name=f"{key}_{q}_{b}") for q in range(1, Q + 1) for b in P.blocks}

        # fleet available in quarter q: F_j(q) = sum_{k <= q - lead_j} x_j[k]
        def F(j, q):
            ks = [k for k in range(0, Q + 1) if k <= q - P.lead[j]]
            return sum((v["x"][j][k] for k in ks), 0.0) if ks else 0.0
        self.F = F

        def Phi(j, T):  # all orders on the path of scenario T (nodes m_0..m_{T-1})
            return sum((v["x"][j][k] for k in range(0, min(T, Q + 1))), 0.0)
        self.Phi = Phi

        for q in range(1, Q + 1):
            FGE, FDG, FABS, FB = F("GE", q), F("DG", q), F("ABS", q), F("BESS", q)
            if P.dg_bridge_mult != 1.0:                        # D-011 sensitivity: site-specific unlimited rating
                FDG = P.dg_bridge_mult * FDG
            RGRq = v["RGR"][q] if P.gr_on else 0.0
            h.addConstr(v["RNR"][q] + v["RST"][q] + RGRq <= P.cap_R)
            # ST authorization (G2: gas rentals are stationary sources under the same filings)
            ks = [k for k in range(Q) if k <= q - 1 - P.L_permit]
            if ks:
                h.addConstr(v["RST"][q] - P.cap_R * sum(v["f"][k] for k in ks) <= 0)
                if P.gr_on:
                    h.addConstr(v["RGR"][q] - P.cap_R * sum(v["f"][k] for k in ks) <= 0)
            else:
                h.addConstr(v["RST"][q] <= 0)
                if P.gr_on:
                    h.addConstr(v["RGR"][q] <= 0)
            if P.residence_on:
                h.addConstr(v["RNR"][q] - P.cap_R * v["z"] <= 0)
            for b in P.blocks:
                gGE, gDG, gNR, gST = v["gGE"][q, b], v["gDG"][q, b], v["gNR"][q, b], v["gST"][q, b]
                a, e, S = v["a"][q, b], v["e"][q, b], v["S"][q]
                c = P.cop(q, b)
                gGR = 0.0
                if P.gr_on:                                    # G2: gas rentals add to the bridge supply
                    gGR = v["gGR"][q, b]
                    h.addConstr(gGR - v["RGR"][q] <= 0)
                h.addConstr(gGE + gDG + gNR + gST + gGR - (1 + P.a_oth) * S - (1.0 / c) * e == 0)
                h.addConstr(a + e - P.kappa * S == 0)
                h.addConstr(a - P.cop_abs * P.eta_h * gGE <= 0)
                if isinstance(FABS, float):
                    h.addConstr(a <= FABS)
                else:
                    h.addConstr(a - FABS <= 0)
                for g, cap in ((gGE, FGE), (gDG, FDG)):
                    if isinstance(cap, float):
                        h.addConstr(g <= cap)
                    else:
                        h.addConstr(g - cap <= 0)
                h.addConstr(gNR - v["RNR"][q] <= 0)
                h.addConstr(gST - v["RST"][q] <= 0)
                # reserve
                firm = v["RNR"][q] + v["RST"][q] + (FGE if not isinstance(FGE, float) else 0) \
                    + (FDG if not isinstance(FDG, float) else 0) + (FB if not isinstance(FB, float) else 0)
                if P.gr_on:
                    firm = firm + v["RGR"][q]
                h.addConstr(firm - (1 + P.r_reserve) * ((1 + P.a_oth) * S + (1.0 / c) * e) >= 0)
                if P.residence_on:
                    h.addConstr(gGE + gDG + gNR + gST + gGR - P.bigM * v["w"][q] <= 0)
        if P.residence_on:
            h.addConstr(sum(v["w"][q] for q in range(1, Q + 1)) + Q * v["z"] <= 3 + Q)
            for q in range(1, Q + 1):
                prev = v["w"][q - 1] if q > 1 else 0.0
                h.addConstr(v["vs"][q] - v["w"][q] + (prev if not isinstance(prev, float) else 0) >= 0)
            h.addConstr(sum(v["vs"][q] for q in range(1, Q + 1)) + Q * v["z"] <= 1 + Q)

        # ---- spine cost pieces (node k), used by C[T] ---------------------------------------------------------
        def node_cost(k):
            """Cost booked at spine node m_k: orders at k, rentals for quarter k+1, filing at k, and (k >= 1) the
            operation of quarter k (fuel, VOM, absorption maintenance, fixed O&M of arrived fleet, delay)."""
            expr = 0.0
            dk = P.disc(k)
            for j in P.assets:
                expr = expr + P.capex[j] * dk * v["x"][j][k]
            if k + 1 <= Q:
                expr = expr + P.disc(k + 1) * (P.rent_nr * v["RNR"][k + 1] + P.rent_st * v["RST"][k + 1])
            if P.rsb_on and k + 1 <= Q:
                q1 = k + 1
                expr = expr + P.disc(q1) * (P.rent_rsb * v["RSB"][q1] + P.inst_rsb * v["RSBs"][q1]) \
                    + P.rsb_residual(q1) * v["RSBs"][q1]
            if P.gr_on and k + 1 <= Q:
                expr = expr + P.disc(k + 1) * P.rent_gr * v["RGR"][k + 1]
            if k < Q:
                expr = expr + P.c_file * dk * v["f"][k]
            if k >= 1:
                q = k
                op = 0.0
                for b in P.blocks:
                    op = op + P.H * ((P.fuel_ge + P.vom["GE"]) * v["gGE"][q, b]
                                     + (P.fuel_dg + P.vom["DG"]) * v["gDG"][q, b]
                                     + P.fuel_dg * (v["gNR"][q, b] + v["gST"][q, b])
                                     + P.abs_maint * v["a"][q, b]) / 1e6
                    if P.gr_on:                                # G2: gas-rental fuel at the GE heat rate, O&M in the rent
                        op = op + P.H * P.fuel_ge * v["gGR"][q, b] / 1e6
                for j in P.assets:
                    Fj = F(j, q)
                    if not isinstance(Fj, float):
                        op = op + P.fom_q[j] * Fj
                op = op + P.vod * (P.L(q) - v["S"][q])
                expr = expr + dk * op
            return expr

        # ---- branches ------------------------------------------------------------------------------------------
        v["y"] = {}
        v["BU"] = {}
        for key in ("m", "bg", "ba", "be"):
            v[key] = {}
        self.C = {}
        self.bu_orders = {}             # T -> [(order quarter, delivery quarter, var)]: purchased standby, counted MW
        legacy_bu = P.backup_mode == "instant" and not P.backup_ramp
        node_exprs = [node_cost(k) for k in range(0, Q + 1)]
        for T in range(1, Q + 2):
            spine = sum((node_exprs[k] for k in range(0, T)), 0.0)
            if T <= Q:
                # Retention by order cohort k (orders placed at spine nodes m_0..m_{T-1}). A cohort retained while still
                # in transit becomes usable post-connection from its original delivery quarter k + lead_j.
                yk = {jj: {k: self._var(0, inf, name=f"y_{jj}_{T}_{k}") for k in range(0, T)} for jj in P.assets}
                v["y"][T] = yk
                if legacy_bu:
                    BU = self._var(0, inf, name=f"BU_{T}")
                    v["BU"][T] = BU
                for jj in P.assets:
                    for k in range(0, T):
                        h.addConstr(yk[jj][k] - v["x"][jj][k] <= 0)

                def kept(jj, q, yk=yk, T=T):
                    ks = [k for k in range(0, T) if k + P.lead[jj] <= q]
                    return sum((yk[jj][k] for k in ks), 0.0) if ks else 0.0

                if legacy_bu:
                    # R2 structure: one purchase at T sized to the full requirement; only DG delivered by T counts.
                    dg_now = kept("DG", T)
                    if isinstance(dg_now, float):
                        h.addConstr(BU >= P.B_req)
                    else:
                        h.addConstr(P.bk_ratio * dg_now + BU >= P.B_req)
                    orders = [(T, T, BU)]
                    conn = P.disc(T) * (P.c_bu * BU)
                else:
                    # The requirement holds in every post-connection quarter with IT load. Kept DG and spine-ordered
                    # standby (SB) count from delivery; purchases arrive after the backup lead time (0 when instant).
                    lb = 0 if P.backup_mode == "instant" else P.lead["SB"]
                    bu = {q: self._var(0, inf, name=f"BU_{T}_{q}") for q in range(T, Q + 1)}
                    v["BU"][T] = bu
                    orders = [(q, q + lb, bu[q]) for q in range(T, Q + 1)]
                    rb = None
                    if P.rsb_on:
                        # continuation of the rented-standby contract after T is observed; quarter T is the spine's
                        rb, sb = {T: v["RSB"][T]}, {T: v["RSBs"][T]}
                        for q in range(T + 1, Q + 1):
                            rb[q] = self._var(0, P.rsb_cap, name=f"RSBb_{T}_{q}")
                            sb[q] = self._var(0, P.rsb_cap, name=f"RSBbs_{T}_{q}")
                            h.addConstr(sb[q] - rb[q] + rb[q - 1] >= 0)
                            starts = [sb[j] if j >= T else v["RSBs"][j]
                                      for j in range(max(1, q - P.rsb_min_q + 1), q + 1)]
                            h.addConstr(rb[q] - sum(starts) >= 0)
                        v.setdefault("RSBb", {})[T] = rb
                        v.setdefault("RSBbs", {})[T] = sb
                    tb = P.dg_bridge_last(T)
                    for q in range(T, Q + 1):
                        need = P.B_need(q)
                        if need <= 0 or q < T + P.bk_start:           # full backup from T + bk_start (Phase 1c: stage_len)
                            continue
                        if P.refurb_active:
                            # bridge-run cohorts (delivered by tb) count only after recommissioning, derated
                            late = [k for k in range(0, T) if tb < k + P.lead["DG"] <= q]
                            dg = sum((yk["DG"][k] for k in late), 0.0)
                            if q >= tb + 1 + P.refurb_delay:
                                run = [k for k in range(0, T) if k + P.lead["DG"] <= tb]
                                dg = dg + P.refurb_derate * sum((yk["DG"][k] for k in run), 0.0)
                        else:
                            dg = kept("DG", q)
                        lhs = P.bk_ratio * dg + sum((var for (_, qd, var) in orders if qd <= q), 0.0)
                        if "SB" in P.assets:
                            lhs = lhs + kept("SB", q)
                        if rb is not None:
                            lhs = lhs + rb[q]
                        if isinstance(lhs, float):
                            raise ValueError(f"no backup can be in place in quarter {q} of scenario T = {T}")
                        h.addConstr(lhs >= need)
                    conn = sum((P.disc(qo) * (P.c_bu * var) for (qo, _, var) in orders), 0.0)
                    if rb is not None:
                        for q in range(T + 1, Q + 1):
                            conn = conn + P.disc(q) * (P.rent_rsb * rb[q] + P.inst_rsb * sb[q]) \
                                + P.rsb_residual(q) * sb[q]
                self.bu_orders[T] = orders
                for jj in P.assets:
                    for k in range(0, T):
                        conn = conn - P.disc(T) * P.sig_ret(jj, T - k - P.lead[jj]) * P.capex[jj] \
                            * (v["x"][jj][k] - yk[jj][k])
                if P.dg_refurb > 0:
                    tb = P.dg_bridge_last(T)
                    run = [k for k in range(0, T) if k + P.lead["DG"] <= tb]
                    if run and tb + 1 <= Q:                           # recommissioned when bridge service ends
                        conn = conn + P.disc(tb + 1) * P.dg_refurb * sum(yk["DG"][k] for k in run)
                post = 0.0
                for q in range(T, Q + 1):
                    Lq = P.L(q)
                    kGE, kABS = kept("GE", q), kept("ABS", q)
                    opq = 0.0
                    partial = q < T + P.stage_len
                    if partial:
                        # Staged connection: grid import is capped at G_part. Retained DG and ST rentals authorized by
                        # spine filings serve the residual; grid partial capacity counts as firm in the reserve.
                        if P.stage_rent_carry and q == T:
                            # G1: quarter T's rental is the spine's contract (node T-1), already paid and permitted
                            rst = v["RST"][T]
                        else:
                            rst = self._var(0, P.cap_R, name=f"brst_{T}_{q}")
                            ks = [k for k in range(min(T, Q)) if k <= q - 1 - P.L_permit]
                            if ks:
                                h.addConstr(rst - P.cap_R * sum(v["f"][k] for k in ks) <= 0)
                            else:
                                h.addConstr(rst <= 0)
                            opq = opq + P.rent_st * rst
                        v.setdefault("brst", {})[T, q] = rst
                        kDG, kB = kept("DG", q), kept("BESS", q)
                        kDGb = P.dg_bridge_mult * kDG if P.dg_bridge_mult != 1.0 else kDG
                    for b in P.blocks:
                        m = self._var(0, inf, name=f"m_{T}_{q}_{b}")
                        g = self._var(0, inf, name=f"bg_{T}_{q}_{b}")
                        a = self._var(0, inf, name=f"ba_{T}_{q}_{b}")
                        e = self._var(0, inf, name=f"be_{T}_{q}_{b}")
                        v["m"][T, q, b], v["bg"][T, q, b], v["ba"][T, q, b], v["be"][T, q, b] = m, g, a, e
                        c = P.cop(q, b)
                        if partial:
                            gd = self._var(0, inf, name=f"bgd_{T}_{q}_{b}")
                            gs = self._var(0, inf, name=f"bgs_{T}_{q}_{b}")
                            v.setdefault("bgd", {})[T, q, b], v.setdefault("bgs", {})[T, q, b] = gd, gs
                            h.addConstr(m + g + gd + gs - (1.0 / c) * e == Lq * (1 + P.a_oth))
                            h.addConstr(m <= P.G_part)
                            h.addConstr(gs - rst <= 0)
                            if isinstance(kDGb, float):
                                h.addConstr(gd <= kDGb)
                            else:
                                h.addConstr(gd - kDGb <= 0)
                            firm = rst + P.G_part
                            for cap in (kGE, kDGb, kB):
                                if not isinstance(cap, float):
                                    firm = firm + cap
                            h.addConstr(firm - (1 + P.r_reserve) * ((1 + P.a_oth) * Lq + (1.0 / c) * e) >= 0)
                            opq = opq + P.H * ((P.fuel_dg + P.vom["DG"]) * gd + P.fuel_dg * gs) / 1e6
                        else:
                            h.addConstr(m + g - (1.0 / c) * e == Lq * (1 + P.a_oth))
                        h.addConstr(a + e == P.kappa * Lq)
                        h.addConstr(a - P.cop_abs * P.eta_h * g <= 0)
                        if isinstance(kABS, float):
                            h.addConstr(a <= kABS)
                        else:
                            h.addConstr(a - kABS <= 0)
                        if isinstance(kGE, float):
                            h.addConstr(g <= kGE)
                        else:
                            h.addConstr(g - kGE <= 0)
                        opq = opq + P.H * (P.grid * m + (P.fuel_ge + P.vom["GE"]) * g + P.abs_maint * a) / 1e6
                    for jj in P.assets:
                        kj = kept(jj, q)
                        if not isinstance(kj, float):
                            opq = opq + P.fom_q[jj] * kj
                    post = post + P.disc(q) * opq
                # cohort age at Q+1 is counted from delivery k + lead_j (0 if still in transit)
                term = sum(P.disc(Q + 1) * P.sig_term(jj, Q + 1 - k - P.lead[jj]) * P.capex[jj] * yk[jj][k]
                           for jj in P.assets for k in range(0, T))
                if P.backup_sym:
                    # D-015: purchased standby pays fixed O&M from delivery and has terminal value, like a kept unit
                    for (_, qd, var) in orders:
                        post = post + (P.fom_bu * sum(P.disc(q) for q in range(qd, Q + 1))) * var
                        term = term + P.disc(Q + 1) * P.sig_term("DG", Q + 1 - qd) * P.c_bu * var
                expr = spine + conn + post - term
            else:
                if P.age_salvage:
                    term = sum(P.disc(Q + 1) * P.sig_term(j, Q + 1 - k - P.lead[j]) * P.capex[j] * v["x"][j][k]
                               for j in P.assets for k in range(0, Q + 1))
                else:
                    term = sum(P.disc(Q + 1) * P.sig_term(j, 0) * P.capex[j] * Phi(j, Q + 1) for j in P.assets)
                expr = spine - term
            if P.gr_on:
                # G2: gas-rental starts whose minimum commitment runs past the last quarter the scenario's spine pays
                # (T, or Q without connection) still owe that rent
                after = min(T, Q)
                expr = expr + sum((P.gr_tail(q, after) * v["RGRs"][q]
                                   for q in range(max(1, after - P.gr_min_q + 2), after + 1)), 0.0)
            CT = self._var(-inf, inf, name=f"C_{T}")
            h.addConstr(CT - expr == 0)
            self.C[T] = CT

        # ---- fixing (for rolling policies and branch polishing) -------------------------------------------------
        for name, val in self.fix.items():
            var = self._lookup(name)
            self.h.changeColBounds(var.index, float(val), float(val))

    def _lookup(self, name):
        parts = name.split("|")
        kind = parts[0]
        v = self.v
        if kind in ("gGE", "gDG", "gNR", "gST", "gGR", "a", "e"):
            return v[kind][int(parts[1]), int(parts[2])]
        if kind == "x":
            return v["x"][parts[1]][int(parts[2])]
        if kind == "n":
            return v["n"][parts[1]]
        if kind == "z":
            return v["z"]
        return v[kind][int(parts[1])]

    # ---- objectives -------------------------------------------------------------------------------------------
    # Every solve is lexicographic: (1) the method's own objective; (2) among its optimal solutions, minimize the
    # uniform sum of scenario costs. Step 2 only breaks ties. It leaves each method's objective value unchanged and
    # removes arbitrary decisions at spine nodes that the method's distribution gives zero weight (degeneracy).
    def _lexi(self, primary_expr, tiebreak=True, tiebreak_mode="milp", tiebreak_presolve=True):
        """tiebreak_mode "milp" (D-008, R2 and Phase 1a): the second stage re-optimizes every decision. "lp" (Phase 1c):
        the second stage fixes the first stage's integer decisions, so it is an LP. It is the same continuation rule
        restricted to the continuous decisions: fast and deterministic, and used where the MILP stage cannot finish.
        "lp_pure" is "lp" solved as a pure LP (the fixed columns are marked continuous; D-022).
        `tiebreak_presolve=False` switches HiGHS presolve off for the second stage only (D-022 as amended: presolve made
        the second stage of one Phase-1c job fail)."""
        if tiebreak_mode not in ("milp", "lp", "lp_pure"):
            raise ValueError(f"unknown tiebreak_mode {tiebreak_mode!r}")
        h = self.h
        h.minimize(primary_expr)
        st = self._status()
        st.update(primary_obj=st["obj"], primary_status=st["status"], primary_gap=st["gap"], tiebreak_ok=None)
        if not st["ok"] or not tiebreak:
            return st
        z = st["obj"]
        tol = 1e-7 * abs(z) + 1e-6
        if tiebreak_mode in ("lp", "lp_pure"):
            self.fix_integers(relax=(tiebreak_mode == "lp_pure"))
        h.addConstr(primary_expr <= z + tol)
        if not tiebreak_presolve:
            h.setOptionValue("presolve", "off")
        h.minimize(sum(self.C[T] for T in range(1, self.P.Q + 2)))
        st2 = self._status()
        if not tiebreak_presolve:
            h.setOptionValue("presolve", "choose")
        if st2["ok"]:
            st2.update(primary_obj=z, primary_status=st["status"], primary_gap=st["gap"], tiebreak_ok=True)
            return st2
        # Tie-break failed. The model's current values are not trustworthy, so restore a valid
        # primary-optimal solution by re-solving the primary objective (the added bound keeps it optimal) and flag it.
        h.minimize(primary_expr)
        st3 = self._status()
        st3.update(primary_obj=z, primary_status=st["status"], primary_gap=st["gap"], tiebreak_ok=False,
                   tiebreak_status=st2["status"])
        return st3

    def solve_dro(self, phat, eps, tiebreak=True, allowed=None, later_only=False, surv_upper=None, surv_lower=None):
        """Wasserstein-1 DRO over the finite support, dualized. `allowed` (0-based states) restricts where the adversary
        may move mass (stage/support bounds); `later_only` forbids earlier moves (directional
        transport). The centre must lie inside `allowed`.
        `surv_upper` / `surv_lower` ({q: bound}) add survival bands S_q = P(T > q) <= u_q / >= l_q on the adversary's
        distribution, with dual prices beta_q, alpha_q >= 0. State index j has T = j + 1, so T > q means
        j >= q. The centre must satisfy the bands."""
        P, h = self.P, self.h
        Q = P.Q
        _check_pmf(phat)
        sup = [i for i in range(Q + 1) if phat[i] > 1e-12]           # index i -> scenario T = i + 1
        dest = list(range(Q + 1)) if allowed is None else sorted(allowed)
        if not set(sup) <= set(dest):
            raise ValueError("the centre puts mass outside the allowed states")
        su, sl = dict(surv_upper or {}), dict(surv_lower or {})
        S_c = {q: float(np.sum(phat[q:])) for q in set(su) | set(sl)}
        if any(S_c[q] > su[q] + 1e-9 for q in su) or any(S_c[q] < sl[q] - 1e-9 for q in sl):
            raise ValueError("the centre violates the survival bands")
        lam = self._var(0, highspy.kHighsInf, name="lambda")
        mu = {i: self._var(-highspy.kHighsInf, highspy.kHighsInf, name=f"mu_{i}") for i in sup}
        beta = {q: self._var(0, highspy.kHighsInf, name=f"beta_{q}") for q in su}
        alpha = {q: self._var(0, highspy.kHighsInf, name=f"alpha_{q}") for q in sl}
        for i in sup:
            for jdx in dest:
                if later_only and jdx < i:
                    continue
                expr = mu[i] - self.C[jdx + 1] + abs(i - jdx) * lam
                for q in su:
                    if jdx >= q:
                        expr = expr + beta[q]
                for q in sl:
                    if jdx >= q:
                        expr = expr - alpha[q]
                h.addConstr(expr >= 0)
        self._lam, self._mu, self._sup = lam, mu, sup
        obj = eps * lam + sum(phat[i] * mu[i] for i in sup)
        if su or sl:
            obj = obj + sum(su[q] * beta[q] for q in su) - sum(sl[q] * alpha[q] for q in sl)
        return self._lexi(obj, tiebreak)

    def solve_saa(self, phat, tiebreak=True, tiebreak_mode="milp", tiebreak_presolve=True):
        Q = self.P.Q
        _check_pmf(phat)
        return self._lexi(sum(phat[i] * self.C[i + 1] for i in range(Q + 1) if phat[i] > 1e-12), tiebreak,
                          tiebreak_mode, tiebreak_presolve)

    def solve_box(self, lo, hi, tiebreak=True):
        """B5 box-robust: min over policies of max_{T in [lo, hi]} C_T (R2b pre-registration)."""
        eta = self._var(-highspy.kHighsInf, highspy.kHighsInf, name="eta")
        for T in range(int(lo), int(hi) + 1):
            self.h.addConstr(eta - self.C[T] >= 0)
        self._eta = eta
        return self._lexi(eta + 0.0, tiebreak)

    def solve_det(self, T_hat, tiebreak=True):
        return self._lexi(self.C[T_hat] + 0.0, tiebreak)

    def _status(self):
        st = self.h.getModelStatus()
        info = self.h.getInfo()
        return dict(status=str(st), obj=info.objective_function_value,
                    gap=getattr(info, "mip_gap", float("nan")),
                    ok=st in (highspy.HighsModelStatus.kOptimal,))

    # ---- extraction -------------------------------------------------------------------------------------------
    def val(self, var):
        return float(self.h.val(var)) if not isinstance(var, float) else var

    def spine_values(self):
        """All spine decision values as a flat dict name -> value (names usable in `fix`)."""
        P, v = self.P, self.v
        out = {}
        for j in P.assets:
            for k in range(P.Q + 1):
                out[f"x|{j}|{k}"] = self.val(v["x"][j][k])
        if P.int_root:
            for j in ("GE", "DG"):
                out[f"n|{j}"] = round(self.val(v["n"][j]))
        for q in range(1, P.Q + 1):
            out[f"RNR|{q}"] = self.val(v["RNR"][q])
            out[f"RST|{q}"] = self.val(v["RST"][q])
            if P.rsb_on:
                out[f"RSB|{q}"] = self.val(v["RSB"][q])
                out[f"RSBs|{q}"] = self.val(v["RSBs"][q])
            if P.gr_on:
                out[f"RGR|{q}"] = self.val(v["RGR"][q])
                out[f"RGRs|{q}"] = self.val(v["RGRs"][q])
            out[f"S|{q}"] = self.val(v["S"][q])
            for key in ("gGE", "gDG", "gNR", "gST", "a", "e") + (("gGR",) if P.gr_on else ()):
                for b in P.blocks:
                    out[f"{key}|{q}|{b}"] = self.val(v[key][q, b])
        for k in range(P.Q):
            out[f"f|{k}"] = round(self.val(v["f"][k]))
        if P.residence_on:
            out["z"] = round(self.val(v["z"]))
            for q in range(1, P.Q + 1):
                out[f"w|{q}"] = round(self.val(v["w"][q]))
                out[f"vs|{q}"] = self.val(v["vs"][q])
        return out

    def C_values(self):
        return np.array([self.val(self.C[T]) for T in range(1, self.P.Q + 2)])


# node-level grouping of spine decisions (for rolling policies): which names belong to node m_k
def node_names(P: Params, k: int):
    names = [f"x|{j}|{k}" for j in P.assets]
    if k == 0 and P.int_root:
        names += ["n|GE", "n|DG"]
    if k + 1 <= P.Q:
        names += [f"RNR|{k + 1}", f"RST|{k + 1}"]
        if P.rsb_on:
            names += [f"RSB|{k + 1}", f"RSBs|{k + 1}"]
        if P.gr_on:
            names += [f"RGR|{k + 1}", f"RGRs|{k + 1}"]
    if k < P.Q:
        names += [f"f|{k}"]
    if k >= 1:
        keys = ("gGE", "gDG", "gNR", "gST", "a", "e") + (("gGR",) if P.gr_on else ())
        names += [f"S|{k}"] + [f"{key}|{k}|{b}" for key in keys for b in P.blocks]
        if P.residence_on:
            names += [f"w|{k}", f"vs|{k}"]
    if k == 0 and P.residence_on:
        names += ["z"]
    return names


def _fix_dict_for(model_names_vals: dict, names):
    return {n: model_names_vals[n] for n in names if n in model_names_vals}


# ----------------------------------------------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------------------------------------------
def policy_dro(P, phat, eps, **kw):
    m = Model(P, **kw)
    st = m.solve_dro(phat, eps)
    return m, st


def policy_saa(P, phat, **kw):
    m = Model(P, **kw)
    st = m.solve_saa(phat)
    return m, st


def conditional_median(phat, k):
    """Median scenario index T of phat restricted to T > k (T = k+1..Q+1). phat index i <-> T = i + 1."""
    p = np.array(phat, dtype=float).copy()
    p[:k] = 0.0                                  # T <= k removed (indices 0..k-1)
    s = p.sum()
    if s <= 0:
        return len(p)                             # T = Q+1
    cdf = np.cumsum(p / s)
    return int(np.searchsorted(cdf, 0.5)) + 1


def policy_rolling_det(P, phat, **kw):
    """B1: at each spine node m_k re-solve deterministic at the conditional median of phat given T > k, commit m_k."""
    fixed = {}
    stats = []
    for k in range(0, P.Q + 1):
        T_hat = max(conditional_median(phat, k), k + 1)
        m = Model(P, fix=fixed, **kw)
        st = m.solve_det(T_hat)
        stats.append(dict(k=k, T_hat=T_hat, **st))
        if not st["ok"]:
            return None, stats
        vals = m.spine_values()
        fixed.update(_fix_dict_for(vals, node_names(P, k)))
    # final model with the whole spine fixed (objective irrelevant; branches polished later)
    m = Model(P, fix=fixed, **kw)
    st = m.solve_det(P.Q + 1)
    stats.append(dict(k="final", **st))
    return m, stats


def policy_det(P, T_hat, **kw):
    m = Model(P, **kw)
    st = m.solve_det(T_hat)
    return m, st


# ----------------------------------------------------------------------------------------------------------------
# Branch polishing and independent cost recomputation
# ----------------------------------------------------------------------------------------------------------------
def polish(P: Params, spine: dict, **kw):
    """Fix the spine; re-optimize each branch T <= Q on its own (post-connection decisions are taken with T known).
    Returns the polished cost vector C (length Q+1) computed by the MILP bookkeeping of a fully fixed-spine model."""
    fix = {k: v for k, v in spine.items()}
    m = Model(P, fix=fix, **kw)
    # With the spine fixed, the branches are independent, so minimizing sum_T C_T optimizes every branch separately.
    m.h.minimize(sum(m.C[T] for T in range(1, P.Q + 2)))
    st = m._status()
    return m, st


def rsb_spine_cost(P: Params, m: Model, k: int):
    """Rented-standby cost booked at spine node k (quarter k + 1 rent, installation, residual beyond Q)."""
    if not P.rsb_on or k + 1 > P.Q:
        return 0.0
    q1 = k + 1
    r, s_ = m.val(m.v["RSB"][q1]), m.val(m.v["RSBs"][q1])
    return P.disc(q1) * (P.rent_rsb * r + P.inst_rsb * s_) + P.rsb_residual(q1) * s_


def rsb_branch_cost(P: Params, m: Model, T: int):
    """Rented-standby cost of branch T (quarters T + 1..Q; quarter T is paid on the spine)."""
    if not P.rsb_on or T > P.Q:
        return 0.0
    rb, sb = m.v["RSBb"][T], m.v["RSBbs"][T]
    return sum(P.disc(q) * (P.rent_rsb * m.val(rb[q]) + P.inst_rsb * m.val(sb[q])) + P.rsb_residual(q) * m.val(sb[q])
               for q in range(T + 1, P.Q + 1))


def rsb_level(P: Params, m: Model, T: int, q: int):
    """Mobilized rented standby (counted MW) in quarter q of scenario T (q >= T)."""
    if not P.rsb_on:
        return 0.0
    return m.val(m.v["RSB"][T]) if q == T else m.val(m.v["RSBb"][T][q])


def gr_spine_cost(P: Params, m: Model, k: int):
    """Gas-rental cost booked at spine node k (Phase 1g G2): quarter k + 1 rent and quarter k fuel."""
    if not P.gr_on:
        return 0.0
    c = P.disc(k + 1) * P.rent_gr * m.val(m.v["RGR"][k + 1]) if k + 1 <= P.Q else 0.0
    if k >= 1:
        c += P.disc(k) * sum(P.H * P.fuel_ge * m.val(m.v["gGR"][k, b]) / 1e6 for b in P.blocks)
    return c


def gr_tail_cost(P: Params, m: Model, T: int):
    """Gas-rental rent still owed in scenario T for starts whose minimum commitment runs past quarter min(T, Q)."""
    if not P.gr_on:
        return 0.0
    after = min(T, P.Q)
    return sum(P.gr_tail(q, after) * m.val(m.v["RGRs"][q]) for q in range(max(1, after - P.gr_min_q + 2), after + 1))


def staged_rent_paid(P: Params, T: int, q: int):
    """Whether branch T pays a staged rental in quarter q (Phase 1g G1: quarter T's is the spine's, paid at T - 1)."""
    return not (P.stage_rent_carry and q == T)


def recompute_costs(P: Params, m: Model):
    """Independent numpy recomputation of C_T from the solved decision values (V2 bookkeeping check)."""
    Q = P.Q
    val = m.val
    assets = P.assets
    x = {j: np.array([val(m.v["x"][j][k]) for k in range(Q + 1)]) for j in assets}

    def Fv(j, q):
        return sum(x[j][k] for k in range(Q + 1) if k <= q - P.lead[j])

    node = np.zeros(Q + 1)
    for k in range(Q + 1):
        c = sum(P.capex[j] * x[j][k] for j in assets) * P.disc(k)
        if k + 1 <= Q:
            c += P.disc(k + 1) * (P.rent_nr * val(m.v["RNR"][k + 1]) + P.rent_st * val(m.v["RST"][k + 1]))
        c += rsb_spine_cost(P, m, k) + gr_spine_cost(P, m, k)
        if k < Q:
            c += P.c_file * P.disc(k) * round(val(m.v["f"][k]))
        if k >= 1:
            q = k
            op = 0.0
            for b in P.blocks:
                op += P.H * ((P.fuel_ge + P.vom["GE"]) * val(m.v["gGE"][q, b])
                             + (P.fuel_dg + P.vom["DG"]) * val(m.v["gDG"][q, b])
                             + P.fuel_dg * (val(m.v["gNR"][q, b]) + val(m.v["gST"][q, b]))
                             + P.abs_maint * val(m.v["a"][q, b])) / 1e6
            op += sum(P.fom_q[j] * Fv(j, q) for j in assets)
            op += P.vod * (P.L(q) - val(m.v["S"][q]))
            c += P.disc(k) * op
        node[k] = c
    C = np.zeros(Q + 1)
    for T in range(1, Q + 2):
        spine = node[:T].sum()
        if T <= Q:
            yk = {j: np.array([val(m.v["y"][T][j][k]) for k in range(T)]) for j in assets}

            def keptv(j, q):
                return sum(yk[j][k] for k in range(T) if k + P.lead[j] <= q)

            bu = [(qo, qd, val(var)) for (qo, qd, var) in m.bu_orders[T]]
            conn = sum(P.disc(qo) * P.c_bu * b for (qo, _, b) in bu)
            conn -= sum(P.disc(T) * P.sig_ret(j, T - k - P.lead[j]) * P.capex[j] * (x[j][k] - yk[j][k])
                        for j in assets for k in range(T))
            conn += refurb_cost(P, T, yk["DG"]) + rsb_branch_cost(P, m, T)
            post = 0.0
            for q in range(T, Q + 1):
                opq = 0.0
                for b in P.blocks:
                    opq += P.H * (P.grid * val(m.v["m"][T, q, b]) + (P.fuel_ge + P.vom["GE"]) * val(m.v["bg"][T, q, b])
                                  + P.abs_maint * val(m.v["ba"][T, q, b])) / 1e6
                if q < T + P.stage_len:                                   # staged connection: residual bridge supply
                    if staged_rent_paid(P, T, q):
                        opq += P.rent_st * val(m.v["brst"][T, q])
                    for b in P.blocks:
                        opq += P.H * ((P.fuel_dg + P.vom["DG"]) * val(m.v["bgd"][T, q, b])
                                      + P.fuel_dg * val(m.v["bgs"][T, q, b])) / 1e6
                opq += sum(P.fom_q[j] * keptv(j, q) for j in assets)
                if P.backup_sym:
                    opq += sum(P.fom_bu * b for (_, qd, b) in bu if qd <= q)
                post += P.disc(q) * opq
            term = sum(P.disc(Q + 1) * P.sig_term(j, Q + 1 - k - P.lead[j]) * P.capex[j] * yk[j][k]
                       for j in assets for k in range(T))
            if P.backup_sym:
                term += sum(P.disc(Q + 1) * P.sig_term("DG", Q + 1 - qd) * P.c_bu * b for (_, qd, b) in bu)
            C[T - 1] = spine + conn + post - term
        else:
            term = sum(P.disc(Q + 1) * P.sig_term(j, Q + 1 - k - P.lead[j]) * P.capex[j] * x[j][k]
                       for j in assets for k in range(Q + 1))
            C[T - 1] = spine - term
        C[T - 1] += gr_tail_cost(P, m, T)
    return C


def policy_detail(P: Params, m: Model):
    """Decisions that show how a policy uses information.
    On the spine: orders by node and asset, rentals by quarter, filings, the NR regime. In each branch T: retained MW by
    asset, purchased backup, and the staged rental schedule."""
    Q, val = P.Q, m.val
    out = dict(orders={j: [round(val(m.v["x"][j][k]), 6) for k in range(Q + 1)] for j in P.assets},
               RST=[round(val(m.v["RST"][q]), 6) for q in range(1, Q + 1)],
               RNR=[round(val(m.v["RNR"][q]), 6) for q in range(1, Q + 1)],
               filings=[k for k in range(Q) if round(val(m.v["f"][k])) == 1],
               z=(round(val(m.v["z"])) if P.residence_on else None), branches={})
    if P.rsb_on:
        out["RSB"] = [round(val(m.v["RSB"][q]), 6) for q in range(1, Q + 1)]
    if P.gr_on:
        out["RGR"] = [round(val(m.v["RGR"][q]), 6) for q in range(1, Q + 1)]
    for T in range(1, Q + 1):
        b = dict(rsb=([round(rsb_level(P, m, T, q), 6) for q in range(T, Q + 1)] if P.rsb_on else None),
                 retained={j: round(sum(val(m.v["y"][T][j][k]) for k in range(T)), 6) for j in P.assets},
                 backup_bought=round(sum(val(var) for (_, _, var) in m.bu_orders[T]), 6))
        if P.stage_len > 0:
            b["staged_rental"] = [round(val(m.v["brst"][T, q]), 6) for q in range(T, min(T + P.stage_len, Q + 1))]
        out["branches"][T] = b
    return out


COMPONENTS = ("capex", "rental", "filing", "bridge_fuel_om", "fixed_om", "delay", "grid", "post_chp", "backup",
              "retirement_value", "terminal_value")


def cost_components(P: Params, m: Model):
    """Decomposition of C_T into additive components. Each value is
    an array over T = 1..Q+1, and the components sum to recompute_costs(P, m). Values (retirement and terminal) enter
    with a negative sign."""
    Q = P.Q
    val = m.val
    assets = P.assets
    x = {j: np.array([val(m.v["x"][j][k]) for k in range(Q + 1)]) for j in assets}

    def Fv(j, q):
        return sum(x[j][k] for k in range(Q + 1) if k <= q - P.lead[j])

    node = {c: np.zeros(Q + 1) for c in ("capex", "rental", "filing", "bridge_fuel_om", "fixed_om", "delay",
                                         "backup")}
    for k in range(Q + 1):
        node["capex"][k] = sum(P.capex[j] * x[j][k] for j in assets) * P.disc(k)
        if k + 1 <= Q:
            node["rental"][k] = P.disc(k + 1) * (P.rent_nr * val(m.v["RNR"][k + 1]) + P.rent_st * val(m.v["RST"][k + 1]))
            if P.gr_on:
                node["rental"][k] += P.disc(k + 1) * P.rent_gr * val(m.v["RGR"][k + 1])
        node["backup"][k] = rsb_spine_cost(P, m, k)
        if k < Q:
            node["filing"][k] = P.c_file * P.disc(k) * round(val(m.v["f"][k]))
        if k >= 1:
            q = k
            fuel = 0.0
            for b in P.blocks:
                fuel += P.H * ((P.fuel_ge + P.vom["GE"]) * val(m.v["gGE"][q, b])
                               + (P.fuel_dg + P.vom["DG"]) * val(m.v["gDG"][q, b])
                               + P.fuel_dg * (val(m.v["gNR"][q, b]) + val(m.v["gST"][q, b]))
                               + P.abs_maint * val(m.v["a"][q, b])) / 1e6
                if P.gr_on:
                    fuel += P.H * P.fuel_ge * val(m.v["gGR"][q, b]) / 1e6
            node["bridge_fuel_om"][k] = P.disc(k) * fuel
            node["fixed_om"][k] = P.disc(k) * sum(P.fom_q[j] * Fv(j, q) for j in assets)
            node["delay"][k] = P.disc(k) * P.vod * (P.L(q) - val(m.v["S"][q]))
    out = {c: np.zeros(Q + 1) for c in COMPONENTS}
    for T in range(1, Q + 2):
        for c in node:
            out[c][T - 1] = node[c][:T].sum()
        if T <= Q:
            yk = {j: np.array([val(m.v["y"][T][j][k]) for k in range(T)]) for j in assets}

            def keptv(j, q):
                return sum(yk[j][k] for k in range(T) if k + P.lead[j] <= q)

            bu = [(qo, qd, val(var)) for (qo, qd, var) in m.bu_orders[T]]
            out["backup"][T - 1] += sum(P.disc(qo) * P.c_bu * b for (qo, _, b) in bu) + refurb_cost(P, T, yk["DG"]) \
                + rsb_branch_cost(P, m, T)
            out["retirement_value"][T - 1] = -sum(P.disc(T) * P.sig_ret(j, T - k - P.lead[j]) * P.capex[j]
                                                  * (x[j][k] - yk[j][k]) for j in assets for k in range(T))
            for q in range(T, Q + 1):
                g = sum(P.H * P.grid * val(m.v["m"][T, q, b]) / 1e6 for b in P.blocks)
                chp = sum(P.H * ((P.fuel_ge + P.vom["GE"]) * val(m.v["bg"][T, q, b])
                                 + P.abs_maint * val(m.v["ba"][T, q, b])) / 1e6 for b in P.blocks)
                out["grid"][T - 1] += P.disc(q) * g
                out["post_chp"][T - 1] += P.disc(q) * chp
                if q < T + P.stage_len:
                    if staged_rent_paid(P, T, q):
                        out["rental"][T - 1] += P.disc(q) * P.rent_st * val(m.v["brst"][T, q])
                    out["bridge_fuel_om"][T - 1] += P.disc(q) * sum(
                        P.H * ((P.fuel_dg + P.vom["DG"]) * val(m.v["bgd"][T, q, b])
                               + P.fuel_dg * val(m.v["bgs"][T, q, b])) / 1e6 for b in P.blocks)
                fom = sum(P.fom_q[j] * keptv(j, q) for j in assets)
                if P.backup_sym:
                    fom += sum(P.fom_bu * b for (_, qd, b) in bu if qd <= q)
                out["fixed_om"][T - 1] += P.disc(q) * fom
            term = sum(P.disc(Q + 1) * P.sig_term(j, Q + 1 - k - P.lead[j]) * P.capex[j] * yk[j][k]
                       for j in assets for k in range(T))
            if P.backup_sym:
                term += sum(P.disc(Q + 1) * P.sig_term("DG", Q + 1 - qd) * P.c_bu * b for (_, qd, b) in bu)
            out["terminal_value"][T - 1] = -term
        else:
            out["terminal_value"][T - 1] = -sum(P.disc(Q + 1) * P.sig_term(j, Q + 1 - k - P.lead[j]) * P.capex[j]
                                                * x[j][k] for j in assets for k in range(Q + 1))
        out["rental"][T - 1] += gr_tail_cost(P, m, T)
    return out


def backup_shortfall(P: Params, m: Model):
    """Independent check of the emergency-backup requirement on solved values: the largest shortfall (MW) over all
    scenarios T <= Q and post-connection quarters with IT load (legacy structure: at T only)."""
    worst = 0.0
    for T in range(1, P.Q + 1):
        y = {j: [m.val(m.v["y"][T][j][k]) for k in range(T)] for j in P.assets}
        bu = [(qd, m.val(var)) for (_, qd, var) in m.bu_orders[T]]
        legacy = P.backup_mode == "instant" and not P.backup_ramp
        tb = P.dg_bridge_last(T)
        for q in ([T] if legacy else range(T + P.bk_start, P.Q + 1)):
            need = P.B_req if legacy else P.B_need(q)
            if need <= 0:
                continue
            have = sum(b for qd, b in bu if qd <= q) + (rsb_level(P, m, T, q) if not legacy else 0.0)
            if P.refurb_active:
                dg = sum(y["DG"][k] for k in range(T) if tb < k + P.lead["DG"] <= q)
                if q >= tb + 1 + P.refurb_delay:
                    dg += P.refurb_derate * sum(y["DG"][k] for k in range(T) if k + P.lead["DG"] <= tb)
            else:
                dg = sum(y["DG"][k] for k in range(T) if k + P.lead["DG"] <= q)
            have += P.bk_ratio * dg
            if "SB" in P.assets:
                have += sum(y["SB"][k] for k in range(T) if k + P.lead["SB"] <= q)
            worst = max(worst, need - have)
    return worst


def refurb_cost(P: Params, T: int, y_dg):
    """Recommissioning cost of the retained bridge-run DG in scenario T (recompute_costs and cost_components)."""
    if P.dg_refurb <= 0:
        return 0.0
    tb = P.dg_bridge_last(T)
    if tb + 1 > P.Q:
        return 0.0
    return P.disc(tb + 1) * P.dg_refurb * sum(y_dg[k] for k in range(T) if k + P.lead["DG"] <= tb)


def worst_case_expectation(C, phat, eps, allowed=None, later_only=False, surv_upper=None, surv_lower=None):
    """Primal Wasserstein-1 worst case over the finite support (V3 duality check), solved as an LP. Same `allowed`,
    `later_only` and survival-band semantics as Model.solve_dro."""
    n = len(C)
    sup = [i for i in range(n) if phat[i] > 1e-12]
    dest = list(range(n)) if allowed is None else sorted(allowed)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)      # HiGHS's scheduler is process-global: every instance must agree
    g = {(i, j): h.addVariable(lb=0.0) for i in sup for j in dest if not (later_only and j < i)}
    for i in sup:
        h.addConstr(sum(g[i, j] for j in dest if (i, j) in g) == phat[i])
    h.addConstr(sum(abs(i - j) * v for (i, j), v in g.items()) <= eps)
    for q, u in (surv_upper or {}).items():
        h.addConstr(sum(v for (i, j), v in g.items() if j >= q) <= u)
    for q, lo in (surv_lower or {}).items():
        h.addConstr(sum(v for (i, j), v in g.items() if j >= q) >= lo)
    h.maximize(sum(C[j] * v for (i, j), v in g.items()))
    return h.getInfo().objective_function_value


def radius(phat, n_records):
    Qp1 = len(phat)
    t = np.arange(1, Qp1 + 1)
    mu = float(np.dot(t, phat))
    sd = math.sqrt(float(np.dot((t - mu) ** 2, phat)))
    return sd / math.sqrt(n_records)


def percentile_quarter(phat, q):
    """Smallest scenario T (1-based) with CDF >= q."""
    cdf = np.cumsum(np.asarray(phat, dtype=float))
    return int(np.searchsorted(cdf, q - 1e-12)) + 1
