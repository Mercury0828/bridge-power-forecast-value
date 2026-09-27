"""Signal tree: the planner observes a utility forecast y at spine node m_tau if the
campus is not connected by then.

Construction.
- One complete policy copy per signal value y, all in one HiGHS instance (explicit non-anticipativity).
- Every decision at spine nodes k < tau is linked across the copies. From node tau on, each copy decides with y known.
- Scenario costs: C_0[T] for T <= tau (connected before the signal; identical in every copy once the spine is linked),
  and C_y[T] for T > tau.
- A joint law Q over (T, y) is given as `pre` (P(T = i + 1) for i < tau) and `post[y][i]` (P(T = i + 1, Y = y) for
  i >= tau). Its expected cost is  sum_{i<tau} pre[i] C_0[i+1] + sum_y sum_{i>=tau} post[y][i] C_y[i+1].
- Survival consistency: the signal concerns the same eventual connection event, and is drawn once.
"""
from __future__ import annotations

import highspy
import numpy as np

from model.bridge import Model, node_names, polish


class SignalModel:
    def __init__(self, P, n_signals, tau, time_limit=600.0, gap=1e-4, threads=1):
        self.P, self.tau, self.n = P, int(tau), int(n_signals)
        self.h = h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("mip_rel_gap", gap)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", int(threads))
        self.copies = [Model(P, h=h, prefix=f"s{y}_") for y in range(self.n)]
        c0 = self.copies[0]
        for k in range(min(self.tau, P.Q + 1)):
            for name in node_names(P, k):
                v0 = c0._lookup(name)
                for c in self.copies[1:]:
                    h.addConstr(c._lookup(name) - v0 == 0)

    def expected(self, law):
        """Linear expression of the expected cost under a joint law {"pre": [...], "post": [[...] per y]}."""
        c0, tau, Q = self.copies[0], self.tau, self.P.Q
        expr = 0.0
        for i in range(min(tau, Q + 1)):
            if law["pre"][i] > 1e-12:
                expr = expr + law["pre"][i] * c0.C[i + 1]
        for y, c in enumerate(self.copies):
            for i in range(tau, Q + 1):
                if law["post"][y][i] > 1e-12:
                    expr = expr + law["post"][y][i] * c.C[i + 1]
        return expr

    def solve_saa(self, law, tiebreak=False, tiebreak_mode="lp", tiebreak_presolve=True):
        """With `tiebreak`, a second lexicographic stage minimizes the uniform sum of every copy's scenario costs among
        primary-optimal policies. This is the common off-support continuation rule of Model._lexi (D-008). In "lp" mode,
        the default here, the first stage's integer decisions are fixed. A MILP second stage over several copies cannot
        finish within the time limit (Phase-1c smoke test). A failed tie-break restores a primary-optimal solution and is
        flagged."""
        if tiebreak:
            vals = np.concatenate([np.asarray(law["pre"], dtype=float)] + [np.asarray(q, dtype=float) for q in law["post"]])
            if np.any((vals > 1e-12) & (vals < LAW_FLOOR)):
                raise ValueError("joint law has masses in (1e-12, 1e-9); clean it first (model.signal.clean_law)")
        primary = self.expected(law)
        self.h.minimize(primary)
        st = self._status()
        # the primary stage's solution quality (Phase 1f; the tie-break stage's own gap is not a quality measure)
        pq = dict(primary_obj=st["obj"], primary_gap=st["gap"], primary_dual_bound=st["dual_bound"])
        st.update(pq, tiebreak_ok=None)
        if not st["ok"] or not tiebreak:
            return st
        z = st["obj"]
        if tiebreak_mode not in ("milp", "lp", "lp_pure"):
            raise ValueError(f"unknown tiebreak_mode {tiebreak_mode!r}")
        if tiebreak_mode in ("lp", "lp_pure"):
            for c in self.copies:
                c.fix_integers(relax=(tiebreak_mode == "lp_pure"))
        self.h.addConstr(primary <= z + 1e-7 * abs(z) + 1e-6)
        Q = self.P.Q
        if not tiebreak_presolve:                                   # D-022 as amended: second stage only
            self.h.setOptionValue("presolve", "off")
        self.h.minimize(sum(c.C[T] for c in self.copies for T in range(1, Q + 2)))
        st2 = self._status()
        if not tiebreak_presolve:
            self.h.setOptionValue("presolve", "choose")
        if st2["ok"]:
            st2.update(pq, tiebreak_ok=True)
            return st2
        self.h.minimize(primary)
        st3 = self._status()
        st3.update(pq, tiebreak_ok=False)
        return st3

    def solve_robust(self, laws):
        """min over signal-aware policies of max over the finite set `laws` of the expected cost.
        The set must contain the true joint laws the guarantee is meant to cover."""
        eta = self.h.addVariable(lb=-highspy.kHighsInf, ub=highspy.kHighsInf, name="eta")
        for law in laws:
            self.h.addConstr(eta - self.expected(law) >= 0)
        self.h.minimize(eta + 0.0)
        return self._status()

    def _status(self):
        st = self.h.getModelStatus()
        info = self.h.getInfo()
        return dict(status=str(st), obj=info.objective_function_value, gap=getattr(info, "mip_gap", float("nan")),
                    dual_bound=getattr(info, "mip_dual_bound", float("nan")),
                    ok=st == highspy.HighsModelStatus.kOptimal)

    def polished_costs(self, time_limit=600.0):
        """Branch-polished cost vector of each copy (the spine of copy y fixed, branches re-optimized)."""
        out = []
        for c in self.copies:
            pm, st = polish(self.P, c.spine_values(), time_limit=time_limit)
            if not st["ok"]:
                return None
            out.append(pm.C_values())
        return out


def evaluate(costs, law, tau):
    """Expected cost of a signal policy (polished cost vectors per y) under a joint law."""
    J = sum(law["pre"][i] * costs[0][i] for i in range(min(tau, len(costs[0]))))
    for y, C in enumerate(costs):
        J += sum(law["post"][y][i] * C[i] for i in range(tau, len(C)))
    return float(J)


LAW_FLOOR = 1e-9


def clean_law(law, floor=LAW_FLOOR):
    """Declared numerical convention (as model.evidence.clean): joint-law masses below `floor` are set to zero and the
    law is renormalized over (pre, post). Masses in (1e-12, 1e-9) become coefficients below HiGHS's small_matrix_value,
    and the tie-break row cannot then be added."""
    pre = np.asarray(law["pre"], dtype=float)
    post = [np.asarray(q, dtype=float) for q in law["post"]]
    pre = np.where(pre < floor, 0.0, pre)
    post = [np.where(q < floor, 0.0, q) for q in post]
    tot = pre.sum() + sum(q.sum() for q in post)
    return {"pre": (pre / tot).tolist(), "post": [(q / tot).tolist() for q in post]}


def joint_law(pT, kernel, tau):
    """pT: marginal of T (length Q+1). kernel[y][i] = K(y | T = i + 1), a column-stochastic signal kernel. The marginal
    is preserved: pre[i] = pT[i] for i < tau; post[y][i] = pT[i] K(y | i) for i >= tau."""
    pT = np.asarray(pT, dtype=float)
    K = np.asarray(kernel, dtype=float)
    pre = np.where(np.arange(len(pT)) < tau, pT, 0.0)
    post = [np.where(np.arange(len(pT)) >= tau, pT * K[y], 0.0) for y in range(K.shape[0])]
    return {"pre": pre.tolist(), "post": [p.tolist() for p in post]}
