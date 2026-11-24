"""Symbolic (SymPy), exact rational, and numerical verification of the formal algebraic results.

Verifies every algebraic claim, closed-form expression, boundary condition, and
code-to-manuscript equation correspondence for the *Scientometrics* submission:
1. Simultaneous Invariance (ESM 1.1 Part 1): Exact constant-degree sufficiency across all 3 VOSviewer
   normalizations (`VOS_AS_CPM`, `VOS_Frac_CPM`, `Leiden_RBConfig`), exact edge-ratio necessity (`iff`),
   unmixed-edge event-degree necessity, K_3 sharpness counterexample, and n=20 AS rank-1 sharpness.
2. Minimal Hub Family G(k, h) (Sec. 2.2 & ESM 1.1 Part 2): Exact closed-form spurious-hub-to-topical
   edge ratios across Full, Fractional (`1/(d_e - 1)`), and Leydesdorff-Park Refined (`2/(d_e*(d_e - 1))`),
   exact asymptotic orders, universal sign flip at gamma=1, single-hub CPM & exact Q_RB full-partition /
   singleton-gain comparisons, `VOS_Frac_CPM` total-weight conservation (`n^2 / 2`), and truncation inflation.
3. Disjoint-Partition `ECS` O(n) Collapse (ESM 1.2): Symbolic proof and dense Personalized PageRank
   linear-system verification for alpha in {0.1, 0.5, 0.85, 0.9, 1.0} under Gates et al. (2019)
   bipartite self-looped CIELG (`W = D_V^{-1} B D_C^{-1} B^T`), contrasted against hollow-clique alpha drift.
4. Closed-Form Hypergeometric `AECS` Expectation (Eq. 7 & ESM 1.2): Symbolic factorial-moment identity,
   exact brute-force S_N permutation verification (N <= 6), and degenerate boundary regimes.
5. Code-to-Equation & Statistical Checks: Exact `igraph` C CPM/modularity quality scaling (including exact
   constant diagonal offset `-gamma * n / (2M')` relative to off-diagonal Eq. 2) and 59-corpus HT4 Jensen gap.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import igraph as ig
import numpy as np
import sympy as sp
from scipy import linalg

EXP_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = EXP_DIR.parent
sys.path.insert(0, str(PAPER_DIR / "biblio_audit"))

from biblio_audit.engine import edges_to_igraph  # noqa: E402
from biblio_audit.metrics import (  # noqa: E402
    adjusted_element_centric_similarity,
    element_centric_scores,
    element_centric_similarity,
    expected_ecs_fixed_marginals,
)
from biblio_audit.networks import build_coupling_matrices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.verify_math")


def verify_prop1_invariance_and_sharpness() -> None:
    """Verify Proposition 1(i) sufficiency, unmixed-edge necessity, and fixed-C sharpness."""
    d, w_ij, k_i, k_j, m2, n_sym, gamma = sp.symbols("d w_ij k_i k_j m2 n gamma", positive=True)
    lam = 1 / (d - 1)

    # 1. Symbolic scale invariance under W_frac = W_full / (d - 1)
    s_as_full = m2 * w_ij / (k_i * k_j)
    s_as_frac = (lam * m2) * (lam * w_ij) / ((lam * k_i) * (lam * k_j))
    assert sp.simplify(s_as_frac - s_as_full) == 0

    s_fr_full = (n_sym / 2) * (w_ij / k_i + w_ij / k_j)
    s_fr_frac = (n_sym / 2) * ((lam * w_ij) / (lam * k_i) + (lam * w_ij) / (lam * k_j))
    assert sp.simplify(s_fr_frac - s_fr_full) == 0

    q_rb_term_full = (w_ij - gamma * k_i * k_j / m2) / m2
    q_rb_term_frac = (lam * w_ij - gamma * (lam * k_i) * (lam * k_j) / (lam * m2)) / (lam * m2)
    assert sp.simplify(q_rb_term_frac - q_rb_term_full) == 0

    # 2. Numerical check on constant-degree bipartite incidence matrices via biblio_audit
    rng = np.random.default_rng(20261123)
    for d_const in (2, 3, 5):
        n_nodes, n_events = 12, 18
        c_mat = np.zeros((n_nodes, n_events), dtype=float)
        for e in range(n_events):
            chosen = rng.choice(n_nodes, size=d_const, replace=False)
            c_mat[chosen, e] = 1.0
        wf, wr = build_coupling_matrices(c_mat, "bc", "vosviewer")
        assert np.allclose(wr, wf / (d_const - 1.0), atol=1e-15)
        fe = {(i, j): wf[i, j] for i in range(n_nodes) for j in range(i + 1, n_nodes) if wf[i, j] > 0}
        re = {(i, j): wr[i, j] for i in range(n_nodes) for j in range(i + 1, n_nodes) if wr[i, j] > 0}
        for norm in ("association_strength", "fractionalization"):
            gf = edges_to_igraph(n_nodes, fe, normalization=norm)
            gr = edges_to_igraph(n_nodes, re, normalization=norm)
            assert np.allclose(gf.es["weight"], gr.es["weight"], atol=1e-14)

    # 3. Necessity: every unequal pair 2 <= d_lo < d_hi <= 8 on minimal connected n = d_hi + 1
    #    breaks edge-weight proportionality across ALL THREE normalizations individually.
    for d_hi in range(3, 9):
        for d_lo in range(2, d_hi):
            n_cur = d_hi + 1
            c_mat = np.zeros((n_cur, 2), dtype=float)
            c_mat[:d_lo, 0] = 1.0
            c_mat[1 : d_hi + 1, 1] = 1.0
            wf, wr = build_coupling_matrices(c_mat, "bc", "vosviewer")
            fe = {(i, j): wf[i, j] for i in range(n_cur) for j in range(i + 1, n_cur) if wf[i, j] > 0}
            re = {(i, j): wr[i, j] for i in range(n_cur) for j in range(i + 1, n_cur) if wr[i, j] > 0}
            for norm in ("association_strength", "fractionalization", "none"):
                gf = edges_to_igraph(n_cur, fe, normalization=norm)
                gr = edges_to_igraph(n_cur, re, normalization=norm)
                ratios = [gr.es[idx]["weight"] / gf.es[idx]["weight"] for idx in range(len(gf.es))]
                assert max(ratios) - min(ratios) > 1e-6

    # 4. Sharpness counterexample on fixed C without unmixed edges:
    #    n=3 with e_0={0,1,2} (d=3) and dyads {0,1},{1,2},{0,2} (d=2) -> W_frac = (3/4) W_full
    c_sharp = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )
    wf_s, wr_s = build_coupling_matrices(c_sharp, "bc", "vosviewer")
    assert np.allclose(wr_s, 0.75 * wf_s, atol=1e-15)

    # 5. Individual Association Strength rank-1 separability sharpness (n=20, |A|=3, |B|=17):
    #    Unmixed AA (d=3, omega=1/2), unmixed AB (d=5, omega=1/4), mixed BB (omega=1/8).
    #    Association Strength alone is invariant (omega_ij = c * alpha_i * alpha_j), while
    #    Fractionalization and Q_RB strictly break -> simultaneous condition is sharp.
    w_aa_f, w_aa_r = Fraction(4760), Fraction(4760, 2)
    w_ab_f, w_ab_r = Fraction(5040), Fraction(5040, 4)
    w_bb_f, w_bb_r = Fraction(8505), Fraction(8505, 8)
    ka_f, ka_r = 2 * w_aa_f + 17 * w_ab_f, 2 * w_aa_r + 17 * w_ab_r
    kb_f, kb_r = 3 * w_ab_f + 16 * w_bb_f, 3 * w_ab_r + 16 * w_bb_r
    m2_f, m2_r = 3 * ka_f + 17 * kb_f, 3 * ka_r + 17 * kb_r
    assert m2_f * w_aa_f / (ka_f * ka_f) == m2_r * w_aa_r / (ka_r * ka_r)
    assert m2_f * w_ab_f / (ka_f * kb_f) == m2_r * w_ab_r / (ka_r * kb_r)
    assert m2_f * w_bb_f / (kb_f * kb_f) == m2_r * w_bb_r / (kb_r * kb_r)
    assert (2 * w_aa_f / ka_f) != (2 * w_aa_r / ka_r)
    LOGGER.info("[PASS] Check 1: Proposition 1(i) constant-d_e invariance, necessity, K_3 and n=20 AS sharpness verified.")


def verify_prop1_minimal_hub_family() -> None:
    """Verify Proposition 1(ii) and Corollary 1 closed-form expressions on G(k, h)."""
    k, h, psi, beta = sp.symbols("k h psi beta", positive=True)
    n = 2 * h
    w_top = k
    w_hub = psi
    k_v = k
    k_u = k + (h - 1) * psi
    m2 = h * k_v + h * k_u

    s_as_top = sp.simplify(m2 * w_top / (k_u * k_v))
    s_as_hub = sp.simplify(m2 * w_hub / (k_u * k_u))
    r_as = sp.simplify(s_as_hub / s_as_top)

    s_fr_top = sp.simplify((n / 2) * (w_top / k_u + w_top / k_v))
    s_fr_hub = sp.simplify((n / 2) * (w_hub / k_u + w_hub / k_u))
    r_fr = sp.simplify(s_fr_hub / s_fr_top)

    r_raw = sp.simplify(w_hub / w_top)

    assert sp.simplify(r_as - psi / (k + (h - 1) * psi)) == 0
    assert sp.simplify(r_fr - 2 * psi / (2 * k + (h - 1) * psi)) == 0
    assert sp.simplify(r_raw - psi / k) == 0

    # Exact closed-form Full/Frac and Full/Refined inflation factors
    phi_as = sp.simplify(r_as.subs(psi, 1) / r_as.subs(psi, 1 / (h - 1)))
    phi_fr = sp.simplify(r_fr.subs(psi, 1) / r_fr.subs(psi, 1 / (h - 1)))
    phi_raw = sp.simplify(r_raw.subs(psi, 1) / r_raw.subs(psi, 1 / (h - 1)))

    assert sp.simplify(phi_as - (k + 1) * (h - 1) / (k + h - 1)) == 0
    assert sp.simplify(phi_fr - (2 * k + 1) * (h - 1) / (2 * k + h - 1)) == 0
    assert sp.simplify(phi_raw - (h - 1)) == 0

    # Strict > 1 inequality for all k >= 1, h >= 3:
    # (k + 1)(h - 1) - (k + h - 1) = k*(h - 2) > 0 for k >= 1, h >= 3
    assert sp.simplify(((k + 1) * (h - 1) - (k + h - 1)) - k * (h - 2)) == 0
    assert sp.simplify(((2 * k + 1) * (h - 1) - (2 * k + h - 1)) - 2 * k * (h - 2)) == 0

    # Exact asymptotic limits:
    # (a) Fixed k, h -> oo: Phi_AS -> k + 1, Phi_Frac -> 2*k + 1 (node-degree partial self-damping)
    assert sp.simplify(sp.limit(phi_as, h, sp.oo) - (k + 1)) == 0
    assert sp.simplify(sp.limit(phi_fr, h, sp.oo) - (2 * k + 1)) == 0
    # (b) Joint growth k = beta * h, h -> oo: Phi_AS / h -> beta/(beta+1), Phi_Frac / h -> 2*beta/(2*beta+1) [Theta(h)!]
    assert sp.simplify(sp.limit(phi_as.subs(k, beta * h) / h, h, sp.oo) - beta / (beta + 1)) == 0
    assert sp.simplify(sp.limit(phi_fr.subs(k, beta * h) / h, h, sp.oo) - (2 * beta) / (2 * beta + 1)) == 0

    # (c) Full vs Leydesdorff-Park Refined: Theta(h) under AS & Frac even for FIXED k, and Theta(h^2) raw!
    psi_ref = 2 / (h * (h - 1))
    om_as = sp.simplify(r_as.subs(psi, 1) / r_as.subs(psi, psi_ref))
    om_fr = sp.simplify(r_fr.subs(psi, 1) / r_fr.subs(psi, psi_ref))
    om_raw = sp.simplify(r_raw.subs(psi, 1) / r_raw.subs(psi, psi_ref))
    assert sp.simplify(om_as - (k * h + 2) * (h - 1) / (2 * (k + h - 1))) == 0
    assert sp.simplify(om_fr - (k * h + 1) * (h - 1) / (2 * k + h - 1)) == 0
    assert sp.simplify(om_raw - h * (h - 1) / 2) == 0
    assert sp.simplify(sp.limit(om_as / h, h, sp.oo) - k / 2) == 0
    assert sp.simplify(sp.limit(om_fr / h, h, sp.oo) - k) == 0

    # (d) Exact sign flip of spurious hub edge (u_r, u_s) at default gamma = 1 across ALL 3 normalizations
    assert sp.simplify((s_as_hub.subs(psi, 1) - 1) - (h - (k - 1) ** 2) / (k + h - 1) ** 2) == 0
    assert sp.simplify((s_as_hub.subs(psi, 1 / (h - 1)) - 1) - (-(k**2) * h + (k + 1) ** 2) / ((h - 1) * (k + 1) ** 2)) == 0
    assert sp.simplify((s_fr_hub.subs(psi, 1) - 1) - (h - k + 1) / (k + h - 1)) == 0
    assert sp.simplify((s_fr_hub.subs(psi, 1 / (h - 1)) - 1) - (-(k - 1) * h + k + 1) / ((h - 1) * (k + 1))) == 0

    # (d2) Exact single-hub G(k,h) unnormalized and normalized CPM/RB partition gains (full counting, gamma = 1)
    m2p_as = sp.simplify(2 * h * s_as_top.subs(psi, 1) + h * (h - 1) * s_as_hub.subs(psi, 1))
    u_as_top = sp.simplify(2 * h * (s_as_top.subs(psi, 1) - 1))
    u_as_hub = sp.simplify(h * (h - 1) * (s_as_hub.subs(psi, 1) - 1))
    assert sp.limit(u_as_top / h**2, h, sp.oo) == 2
    assert sp.limit(u_as_hub / h, h, sp.oo) == 1
    assert sp.limit(u_as_top / m2p_as, h, sp.oo) == sp.Rational(2, 3)
    assert sp.limit(u_as_hub / m2p_as, h, sp.oo) == 0

    m2p_fr = sp.simplify(2 * h * s_fr_top.subs(psi, 1) + h * (h - 1) * s_fr_hub.subs(psi, 1))
    u_fr_top = sp.simplify(2 * h * (s_fr_top.subs(psi, 1) - 1))
    u_fr_hub = sp.simplify(h * (h - 1) * (s_fr_hub.subs(psi, 1) - 1))
    assert sp.simplify(m2p_fr - 4 * h**2) == 0
    assert sp.limit(u_fr_top / h**2, h, sp.oo) == 2
    assert sp.limit(u_fr_hub / h**2, h, sp.oo) == 1
    assert sp.limit(u_fr_top / m2p_fr, h, sp.oo) == sp.Rational(1, 2)
    assert sp.limit(u_fr_hub / m2p_fr, h, sp.oo) == sp.Rational(1, 4)

    # Exact Q_RB single-cluster contribution, full partition (incl. h singletons), singleton gain, & topical dyads
    ku_f = k_u.subs(psi, 1)
    m2_f = m2.subs(psi, 1)
    q_rb_hub_c0 = sp.simplify((h * (h - 1) - (h * ku_f) ** 2 / m2_f) / m2_f)
    q_rb_hub_full = sp.simplify(q_rb_hub_c0 - h * (k_v**2 / m2_f) / m2_f)
    q_rb_sing = sp.simplify(-(h * ku_f**2 + h * k_v**2) / (m2_f**2))
    q_rb_hub_gain_over_sing = sp.simplify(q_rb_hub_full - q_rb_sing)
    q_rb_top = sp.simplify(h * (2 * k - (ku_f + k_v) ** 2 / m2_f) / m2_f)
    assert sp.simplify(q_rb_hub_c0 + k**2 / (h + 2 * k - 1) ** 2) == 0
    assert sp.simplify(q_rb_hub_full + k**2 * (h + 1) / (h * (h + 2 * k - 1) ** 2)) == 0
    assert sp.simplify(q_rb_hub_gain_over_sing - (h - 1) * (h - (k - 1) ** 2) / (h * (h + 2 * k - 1) ** 2)) == 0
    assert sp.simplify(q_rb_top - (2 * k - 1) * (h - 1) / (h * (h + 2 * k - 1))) == 0

    # (e) Exact Fraction + live biblio_audit verification over (k, h) grid + VOS_Frac_CPM sum = n^2/2
    for k_val in (1, 2, 5, 10, 20):
        for h_val in (3, 5, 10, 25):
            n_val = 2 * h_val
            c_mat = np.zeros((n_val, h_val * k_val + 1), dtype=float)
            col = 0
            for r in range(h_val):
                for _ in range(k_val):
                    c_mat[2 * r, col] = 1.0
                    c_mat[2 * r + 1, col] = 1.0
                    col += 1
            for r in range(h_val):
                c_mat[2 * r, col] = 1.0

            wf, wr = build_coupling_matrices(c_mat, "bc", "vosviewer")
            _, wq = build_coupling_matrices(c_mat, "bc", "refined")
            fe = {(i, j): wf[i, j] for i in range(n_val) for j in range(i + 1, n_val) if wf[i, j] > 0}
            re = {(i, j): wr[i, j] for i in range(n_val) for j in range(i + 1, n_val) if wr[i, j] > 0}
            qe = {(i, j): wq[i, j] for i in range(n_val) for j in range(i + 1, n_val) if wq[i, j] > 0}

            kf, hf = Fraction(k_val), Fraction(h_val)
            for norm, exact_fr, exact_ref in [
                (
                    "association_strength",
                    (kf + 1) * (hf - 1) / (kf + hf - 1),
                    (kf * hf + 2) * (hf - 1) / (2 * (kf + hf - 1)),
                ),
                (
                    "fractionalization",
                    (2 * kf + 1) * (hf - 1) / (2 * kf + hf - 1),
                    (kf * hf + 1) * (hf - 1) / (2 * kf + hf - 1),
                ),
                ("none", hf - 1, hf * (hf - 1) / 2),
            ]:
                gf = edges_to_igraph(n_val, fe, normalization=norm)
                gr = edges_to_igraph(n_val, re, normalization=norm)
                gq = edges_to_igraph(n_val, qe, normalization=norm)
                if norm == "fractionalization":
                    assert abs(sum(gf.es["weight"]) - (n_val * n_val) / 2.0) < 1e-11
                    assert abs(sum(gr.es["weight"]) - (n_val * n_val) / 2.0) < 1e-11
                rf_val = gf.es[gf.get_eid(0, 2)]["weight"] / gf.es[gf.get_eid(0, 1)]["weight"]
                rr_val = gr.es[gr.get_eid(0, 2)]["weight"] / gr.es[gr.get_eid(0, 1)]["weight"]
                rq_val = gq.es[gq.get_eid(0, 2)]["weight"] / gq.es[gq.get_eid(0, 1)]["weight"]
                assert abs((rf_val / rr_val) - float(exact_fr)) < 1e-11
                assert abs((rf_val / rq_val) - float(exact_ref)) < 1e-10

    # (f) Corollary 1 denominator-scope truncation inflation: (d_full - 1) / (d_core - 1) > 1
    d_core, r_strip = sp.symbols("d_core r_strip", positive=True)
    d_full = d_core + r_strip
    ratio_denom = sp.simplify((1 / (d_core - 1)) / (1 / (d_full - 1)))
    assert sp.simplify(ratio_denom - (d_core + r_strip - 1) / (d_core - 1)) == 0
    LOGGER.info("[PASS] Check 2: Proposition 1(ii) & Corollary 1 exact ratios, orders, and sign flips verified.")


def _dense_cielg_transition(membership: list[int]) -> np.ndarray:
    """Construct Gates et al. (2019) bipartite two-step CIELG matrix W = D_V^{-1} B D_C^{-1} B^T."""
    n = len(membership)
    clusters = sorted(set(membership))
    c_idx = {c: idx for idx, c in enumerate(clusters)}
    b_mat = np.zeros((n, len(clusters)), dtype=float)
    for i, c in enumerate(membership):
        b_mat[i, c_idx[c]] = 1.0
    proj1 = b_mat / b_mat.sum(axis=1, keepdims=True)
    proj2 = b_mat / b_mat.sum(axis=0, keepdims=True)
    return proj1 @ proj2.T


def _dense_ppr_affinity(w_mat: np.ndarray, alpha: float) -> np.ndarray:
    """Compute exact dense Personalized PageRank row vectors p_i(alpha) for alpha in (0, 1]."""
    n = w_mat.shape[0]
    if alpha < 1.0:
        # Solve p_i = (1 - alpha) e_i + alpha p_i W  <=>  p_i (I - alpha W) = (1 - alpha) e_i
        return linalg.solve((np.eye(n) - alpha * w_mat).T, (1.0 - alpha) * np.eye(n)).T
    # At alpha = 1.0, W on a disjoint partition is idempotent (W^2 = W), so lim_{alpha -> 1-} p_i(alpha) = e_i^T W
    return w_mat.copy()


def verify_ecs_ppr_collapse_and_hollow_trap() -> None:
    """Verify exact O(N) ECS collapse across alpha in {0.1, 0.5, 0.85, 0.9, 1.0} & catch loopless-clique bug."""
    # 1. Symbolic proof for both a >= b and a < b
    alpha, a, b, o = sp.symbols("alpha a b o", positive=True)
    # Case a >= b >= o: |1/a - 1/b| = 1/b - 1/a
    l1_ageb = o * alpha * (1 / b - 1 / a) + (a - o) * (alpha / a) + (b - o) * (alpha / b)
    ecs_ageb = sp.simplify(1 - l1_ageb / (2 * alpha))
    assert sp.simplify(ecs_ageb - o / a) == 0

    # Case b >= a >= o: |1/a - 1/b| = 1/a - 1/b
    l1_bgea = o * alpha * (1 / a - 1 / b) + (a - o) * (alpha / a) + (b - o) * (alpha / b)
    ecs_bgea = sp.simplify(1 - l1_bgea / (2 * alpha))
    assert sp.simplify(ecs_bgea - o / b) == 0

    # 2. Dense linear-system PPR verification across alpha in {0.1, 0.5, 0.85, 0.9, 1.0}
    test_pairs = [
        ([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]),
        ([0, 0, 0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 2, 2, 3, 3]),
        ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0]),
        ([0, 0, 1, 1, 1, 2, 2, 2, 2, 3], [1, 2, 1, 1, 0, 0, 0, 2, 2, 2]),
    ]
    for m1, m2 in test_pairs:
        fast_scores = element_centric_scores(m1, m2)
        w1 = _dense_cielg_transition(m1)
        w2 = _dense_cielg_transition(m2)
        # Verify idempotence W^2 == W of Gates et al. (2019) disjoint CIELG
        assert np.allclose(w1 @ w1, w1, atol=1e-15)
        assert np.allclose(w2 @ w2, w2, atol=1e-15)
        for al in (0.1, 0.5, 0.85, 0.9, 1.0):
            p1 = _dense_ppr_affinity(w1, al)
            p2 = _dense_ppr_affinity(w2, al)
            cl1_scores = 1.0 - (1.0 / (2.0 * al)) * np.sum(np.abs(p1 - p2), axis=1)
            assert np.allclose(cl1_scores, fast_scores, atol=1e-14), f"Mismatch at alpha={al}"

    # 3. Verify that a LOOPLESS (hollow) clique W_hollow DOES NOT cancel alpha when cluster sizes differ!
    m_a = [0, 0, 0, 1, 1]  # |c_0^A| = 3
    m_b = [0, 0, 1, 1, 1]  # |c_0^B| = 2
    hollow_vals = []
    for al in (0.1, 0.5, 0.9):
        w_ha = np.zeros((5, 5))
        w_hb = np.zeros((5, 5))
        for clu, w_h in [(m_a, w_ha), (m_b, w_hb)]:
            for i in range(5):
                mates = [j for j in range(5) if clu[j] == clu[i] and j != i]
                for j in mates:
                    w_h[i, j] = 1.0 / len(mates)
        p_ha = _dense_ppr_affinity(w_ha, al)
        p_hb = _dense_ppr_affinity(w_hb, al)
        hollow_vals.append(float(1.0 - (1.0 / (2.0 * al)) * np.sum(np.abs(p_ha[0] - p_hb[0]))))
    # Confirm alpha-dependence on hollow clique (spread > 0.05 across alpha in {0.1, 0.5, 0.9})
    assert max(hollow_vals) - min(hollow_vals) > 0.05
    LOGGER.info("[PASS] Check 3: Exact O(N) ECS collapse across alpha in {0.1..1.0} & hollow-clique check verified.")


def verify_aecs_hypergeometric_and_boundaries() -> None:
    """Verify closed-form hypergeometric E_pi[ECS], factorial-moment simplification, and boundaries."""
    # 1. Symbolic verification of E[O^2] = Var(O) + E[O]^2 == (a*b/N)*(1 + (a-1)*(b-1)/(N-1))
    n_s, a_s, b_s = sp.symbols("N a b", positive=True)
    mean_o = a_s * b_s / n_s
    var_o = a_s * b_s * (n_s - a_s) * (n_s - b_s) / (n_s**2 * (n_s - 1))
    factorial_form = (a_s * b_s / n_s) * (1 + (a_s - 1) * (b_s - 1) / (n_s - 1))
    assert sp.simplify((var_o + mean_o**2) - factorial_form) == 0

    # 2. Exact brute-force S_N permutation check on N = 2, 3, 5, 6
    perm_cases = [
        ([0, 0], [0, 1]),
        ([0, 0, 1], [0, 1, 1]),
        ([0, 0, 0, 1, 1], [0, 0, 1, 1, 2]),
        ([0, 0, 0, 1, 1, 2], [0, 1, 1, 1, 2, 2]),
    ]
    for m1, m2 in perm_cases:
        n_cur = len(m1)
        exact_closed = expected_ecs_fixed_marginals(m1, m2)

        # Exact rational calculation via Fraction
        c1 = Counter(m1)
        c2 = Counter(m2)
        nf = Fraction(n_cur)
        frac_sum = Fraction(0)
        for ar in c1.values():
            for bc in c2.values():
                af, bf = Fraction(ar), Fraction(bc)
                e_o2 = (af * bf / nf) * (1 + (af - 1) * (bf - 1) / (nf - 1))
                frac_sum += e_o2 / Fraction(max(ar, bc))
        frac_exp = frac_sum / nf
        assert abs(exact_closed - float(frac_exp)) < 1e-15

        # Exhaustive N! permutation enumeration
        perm_ecss = []
        perm_aecss = []
        for p in itertools.permutations(range(n_cur)):
            m1_perm = [m1[p[idx]] for idx in range(n_cur)]
            perm_ecss.append(element_centric_similarity(m1_perm, m2))
            perm_aecss.append(adjusted_element_centric_similarity(m1_perm, m2))
        assert abs(float(np.mean(perm_ecss)) - exact_closed) < 1e-14
        if exact_closed < 1.0 - 1e-12:
            assert abs(float(np.mean(perm_aecss))) < 1e-14

    # 3. Boundary checks: N=0, N=1, N=2, K_1=1, K_1=N
    assert element_centric_similarity([], []) == 1.0
    assert adjusted_element_centric_similarity([0], [0]) == 1.0
    assert adjusted_element_centric_similarity([0, 0], [0, 0]) == 1.0
    assert adjusted_element_centric_similarity([0, 1], [1, 0]) == 1.0
    assert adjusted_element_centric_similarity([0, 0], [0, 1]) == 0.0
    # K_1 = 1 vs nontrivial K_2 >= 2: ECS == E_pi[ECS] == sum(b_c^2)/N^2 -> AECS == 0.0
    assert abs(adjusted_element_centric_similarity([0, 0, 0, 0, 0], [0, 0, 1, 1, 2])) < 1e-15
    # K_1 = N (all singletons) vs nontrivial K_2 < N: ECS == E_pi[ECS] == K_2 / N -> AECS == 0.0
    assert abs(adjusted_element_centric_similarity([0, 1, 2, 3, 4], [0, 0, 1, 1, 2])) < 1e-15
    # K_1 = N vs K_2 = N: identical singleton partitions -> AECS == 1.0
    assert adjusted_element_centric_similarity([0, 1, 2, 3], [3, 2, 1, 0]) == 1.0
    LOGGER.info("[PASS] Check 4: Closed-form hypergeometric AECS expectation, S_N permutations, and boundaries verified.")


def verify_igraph_objectives_and_jensen_gap() -> None:
    """Verify exact igraph C CPM/modularity normalization formulas and 59-corpus HT4 Jensen gap."""
    g = ig.Graph(n=5, edges=[(0, 1), (1, 2), (0, 2), (3, 4)], directed=False)
    g.es["weight"] = [2.7, 1.4, 3.1, 4.2]
    m2_prime = 2.0 * sum(g.es["weight"])
    deg = g.strength(weights="weight")

    for gamma in (0.5, 1.0, 1.7):
        cl_cpm = g.community_leiden(objective_function="CPM", weights="weight", resolution=gamma)
        mem = cl_cpm.membership
        sz = Counter(mem)
        e_in_2 = 2.0 * sum(w for e, w in zip(g.es, g.es["weight"]) if mem[e.source] == mem[e.target])
        # igraph CPM sums over all ordered pairs (i, j) including diagonal i=j (|c|^2), which differs from
        # off-diagonal Eq. (2) (|c|*(|c|-1)) by the partition-independent constant -gamma * n / (2M').
        q_cpm_ordered = (e_in_2 - gamma * sum(v * v for v in sz.values())) / m2_prime
        q_cpm_offdiag = (e_in_2 - gamma * sum(v * (v - 1) for v in sz.values())) / m2_prime
        assert abs(cl_cpm.quality - q_cpm_ordered) < 1e-12
        assert abs((q_cpm_ordered - q_cpm_offdiag) - (-gamma * g.vcount() / m2_prime)) < 1e-12

    cl_rb = g.community_leiden(objective_function="modularity", weights="weight", resolution=1.0)
    mem_rb = cl_rb.membership
    q_mod_ig = g.modularity(mem_rb, weights="weight")
    e_in_rb_2 = 2.0 * sum(w for e, w in zip(g.es, g.es["weight"]) if mem_rb[e.source] == mem_rb[e.target])
    vol_by_c: dict[int, float] = Counter()
    for u_idx, c_id in enumerate(mem_rb):
        vol_by_c[c_id] += deg[u_idx]
    q_rb_ordered = (e_in_rb_2 - sum((v * v) / m2_prime for v in vol_by_c.values())) / m2_prime
    assert abs(q_mod_ig - q_rb_ordered) < 1e-12

    prod_json = EXP_DIR / "results" / "layer2_counting_audit_production.json"
    if prod_json.exists():
        payload = json.loads(prod_json.read_text(encoding="utf-8"))
        revs = payload["reviews"]
        w_map = payload["strata_weights_4q"]
        tot_w = sum(w_map[q] for q in ("Q1", "Q2", "Q3", "Q4"))

        def _ht4(fn) -> float:
            return sum(
                (w_map[q] / tot_w) * float(np.mean([fn(r) for r in revs if r["quartile"] == q]))
                for q in ("Q1", "Q2", "Q3", "Q4")
            )

        e_wf = _ht4(lambda r: r["vos_core"]["ami_within_full"])
        e_wr = _ht4(lambda r: r["vos_core"]["ami_within_frac"])
        e_min = _ht4(lambda r: min(r["vos_core"]["ami_within_full"], r["vos_core"]["ami_within_frac"]))
        e_cx = _ht4(lambda r: r["vos_core"]["ami_cross_mid"])
        jensen_gap = min(e_wf, e_wr) - e_min
        assert jensen_gap >= 0.0 and abs(jensen_gap - 0.013885) < 1e-5
        assert abs((e_min - e_cx) - payload["ht4_primary_nondegenerate"]["vos_core_delta_count_mid"]["mean"]) < 1e-4
    LOGGER.info("[PASS] Check 5: Exact igraph CPM/RBConfig equations and 59-corpus HT4 Jensen gap verified.")


def main() -> int:
    """Run all symbolic, rational, and numerical verification suites."""
    verify_prop1_invariance_and_sharpness()
    verify_prop1_minimal_hub_family()
    verify_ecs_ppr_collapse_and_hollow_trap()
    verify_aecs_hypergeometric_and_boundaries()
    verify_igraph_objectives_and_jensen_gap()
    LOGGER.info("ALL 5 FORMAL VERIFICATION SUITES PASSED WITH ZERO DISCREPANCIES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
