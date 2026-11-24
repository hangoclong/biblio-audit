"""Unit tests for this study counting_engine."""

from __future__ import annotations

import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src_audit"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from counting_engine import (  # noqa: E402
    build_coupling_matrices,
    build_sparse_edges_from_works,
    edges_to_igraph,
    evaluate_counting_sensitivity,
    extract_shared_lcc_edges,
    hungarian_core_match,
    landmark_and_tfidf_retention,
    merge_small_clusters_cwts,
    temporal_bipartite_rewire,
)


def test_perianes_rodriguez_2016_table4_table5_exact() -> None:
    """Verify Eq. (7)-(10) verbatim against Perianes-Rodriguez et al. (2016, pp. 13-14) Table 4-5."""
    c_mat = np.array(
        [
            [3, 1, 2, 0],
            [2, 0, 1, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=float,
    )
    v_full, v_frac = build_coupling_matrices(c_mat, "bibliographic_coupling")

    expected_full = np.array(
        [
            [0, 8, 5, 0, 1],
            [8, 0, 2, 0, 0],
            [5, 2, 0, 0, 2],
            [0, 0, 0, 0, 1],
            [1, 0, 2, 1, 0],
        ],
        dtype=float,
    )
    expected_frac = np.array(
        [
            [0.00, 2.20, 1.26666667, 0.00, 0.33333333],
            [2.20, 0.00, 0.40, 0.00, 0.00],
            [1.26666667, 0.40, 0.00, 0.00, 0.66666667],
            [0.00, 0.00, 0.00, 0.00, 1.00],
            [0.33333333, 0.00, 0.66666667, 1.00, 0.00],
        ],
        dtype=float,
    )
    assert np.allclose(v_full, expected_full, atol=1e-9)
    assert np.allclose(v_frac, expected_frac, atol=1e-6)
    assert np.all(np.diag(v_full) == 0.0)
    assert np.all(np.diag(v_frac) == 0.0)


def test_singleton_references_do_not_divide_by_zero() -> None:
    """References cited by only 1 paper (n_k = 1) must be filtered without NaN/Inf."""
    c_mat = np.array([[1, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    v_full, v_frac = build_coupling_matrices(c_mat, "bibliographic_coupling")
    assert np.all(np.isfinite(v_full)) and np.all(np.isfinite(v_frac))
    assert v_frac[0, 1] == pytest.approx(1.0)
    assert v_frac[0, 2] == pytest.approx(1.0)
    assert v_frac[1, 2] == pytest.approx(0.0)


def test_cocitation_symmetry() -> None:
    """Co-citation projects columns C^T C symmetrically with zero diagonal."""
    c_mat = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=float)
    v_full, v_frac = build_coupling_matrices(c_mat, "co_citation")
    assert v_full.shape == (3, 3)
    assert v_full[0, 1] == pytest.approx(2.0)
    assert v_frac[0, 1] == pytest.approx(1.5)
    assert v_frac[0, 2] == pytest.approx(0.5)


def test_tls_within_network_ignores_future_global_citations() -> None:
    """BC core_selection='tls_within_network' picks tightly coupled docs, ignoring global cited_by_count."""
    works = [
        {"id": "W_ISOLATED_FAMOUS", "cited_by_count": 99999, "referenced_works": ["UNIQUE_REF"]},
        {"id": "W_CORE_1", "cited_by_count": 1, "referenced_works": ["R1", "R2", "R3"]},
        {"id": "W_CORE_2", "cited_by_count": 1, "referenced_works": ["R1", "R2", "R3"]},
        {"id": "W_CORE_3", "cited_by_count": 1, "referenced_works": ["R1", "R2", "R3"]},
    ]
    selected, _, _ = build_sparse_edges_from_works(
        works, "bibliographic_coupling", core_top_n=3, core_selection="tls_within_network"
    )
    ids = {w["id"] for w in selected}
    assert ids == {"W_CORE_1", "W_CORE_2", "W_CORE_3"}


def test_cwts_small_cluster_merge_maximizes_average_link_strength() -> None:
    """CWTS removeSmallClustersBasedOnNNodes merges tiny cluster into highest avg-link neighbor."""
    # Cluster 0 (size 5: nodes 0..4), Cluster 1 (size 5: nodes 5..9), tiny Cluster 2 (size 2: nodes 10..11)
    # Cluster 2 connects weakly (w=1) to Cluster 0, strongly (w=10) to Cluster 1 -> MUST merge into Cluster 1!
    edges: dict[tuple[int, int], float] = {}
    for i in range(5):
        for j in range(i + 1, 5):
            edges[(i, j)] = 5.0
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges[(i, j)] = 5.0
    edges[(10, 11)] = 5.0
    edges[(10, 0)] = 1.0
    edges[(10, 5)] = 10.0
    edges[(11, 6)] = 10.0
    g = edges_to_igraph(12, edges, assoc_strength=False)
    raw_mem = [0] * 5 + [1] * 5 + [2, 2]
    merged = merge_small_clusters_cwts(g, raw_mem, min_cluster_size=5)
    assert len(set(merged)) == 2
    assert merged[10] == merged[5] and merged[11] == merged[6]
    assert merged[10] != merged[0]


def test_edges_to_igraph_canonical_dedup() -> None:
    """Unsorted duplicate keys (u,v) and (v,u) must sum cleanly into one undirected edge."""
    g = edges_to_igraph(3, {(0, 1): 2.0, (1, 0): 3.0, (1, 2): 1.0}, assoc_strength=False)
    assert g.ecount() == 2
    assert sorted(g.es["weight"]) == pytest.approx([1.0, 5.0])


def test_hungarian_invariance_under_rank_swap() -> None:
    """Hungarian alignment fixes naive top-cluster rank-swap bug."""
    m1 = [0] * 102 + [1] * 98
    m2 = [1] * 99 + [0] * 101
    score, mapping = hungarian_core_match(m1, m2)
    assert score == pytest.approx(197 / 200)
    assert mapping[0] == 1 and mapping[1] == 0


def test_temporal_bipartite_rewire_preserves_degrees_causality_and_returns_acceptance() -> None:
    """Year-stratified bipartite rewiring strictly preserves degrees, t_ref <= t_paper, and returns acceptance rate."""
    paper_years = {p: 2015 + p for p in range(10)}
    ref_years = {r: 2010 + r for r in range(12)}
    edges = [(p, r) for p in range(10) for r in range(12) if ref_years[r] <= paper_years[p] and (p + r) % 2 == 0]
    rewired, acc_rate = temporal_bipartite_rewire(edges, paper_years, ref_years, swaps_per_edge=15, seed=20261122)

    assert len(rewired) == len(edges)
    assert len(set(rewired)) == len(edges)
    assert Counter(p for p, _ in rewired) == Counter(p for p, _ in edges)
    assert Counter(r for _, r in rewired) == Counter(r for _, r in edges)
    assert 0.0 < acc_rate <= 1.0
    for p, r in rewired:
        assert ref_years[r] <= paper_years[p]
    assert set(rewired) != set(edges)


def test_synthetic_planted_communities_low_distortion_vosviewer_parity() -> None:
    """Planted clean 2-community graph has Delta_count ~ 0 under VOS_AS_CPM + CWTS merge."""
    works = []
    for i in range(50):
        cl = 0 if i < 25 else 1
        refs = [f"C{cl}_R{r}" for r in range(6)]
        if i % 5 == 0:
            refs.append("SHARED_R0")
        works.append({"id": f"W{i}", "cited_by_count": 50 - i, "title": f"cluster {cl} topic analysis", "referenced_works": refs})

    nodes, ef, er = build_sparse_edges_from_works(works, "bibliographic_coupling")
    res = evaluate_counting_sensitivity(
        nodes, ef, er, objective="VOS_AS_CPM", n_seeds=6, n_random_starts=3, merge_small_clusters=True, grid_points=20, min_cluster_size=5
    )
    assert res.n_lcc == 50
    assert res.merge_small_clusters is True
    assert res.exact_k_matched is True
    assert res.hungarian_match >= 0.95
    assert abs(res.delta_count_mid) <= 0.05


def test_pre_lcc_association_strength_invariance_and_nonzero_tfidf() -> None:
    """Disconnected components must not shift LCC AS weights; TF-IDF ignores zero-weight terms."""
    full_e = {(0, 1): 2.0, (1, 2): 2.0, (2, 3): 1.0, (3, 4): 2.0, (4, 5): 2.0, (6, 7): 1000.0}
    frac_e = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 0.5, (3, 4): 1.0, (4, 5): 1.0, (6, 7): 500.0}
    lcc_idx, f_sub, _ = extract_shared_lcc_edges(8, full_e, frac_e)
    assert lcc_idx == [0, 1, 2, 3, 4, 5]
    assert (6, 7) not in f_sub
    g_as = edges_to_igraph(len(lcc_idx), f_sub, assoc_strength=True)
    assert max(g_as.es["weight"]) < 10.0

    recs = (
        [{"cited_by_count": 10, "title": "supply chain logistics inventory warehouse"} for _ in range(10)]
        + [{"cited_by_count": 10, "title": "neural transformer attention gradient descent"} for _ in range(10)]
    )
    m1 = [0] * 10 + [1] * 10
    m2 = [0] * 10 + [1] * 10
    _, tf_same = landmark_and_tfidf_retention(m1, m2, recs, min_cluster_size=5)
    assert tf_same == pytest.approx(1.0)


def test_4stratum_ht_bootstrap_and_total_partition_rope() -> None:
    """4-stratum HT bootstrap keeps exact Q1..Q4 weights and ROPE has zero dead-zone."""
    mod = import_module("04_run_counting_audit")
    vals = np.array([0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40])
    journals = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"]
    strata = ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4"]
    wh = {"Q1": 0.50, "Q2": 0.20, "Q3": 0.20, "Q4": 0.10}
    pt, lo, hi = mod.stratified_journal_cluster_bootstrap_ci(vals, journals, strata=strata, strata_weights=wh, n_reps=200)
    expected = 0.50 * 0.10 + 0.20 * 0.20 + 0.20 * 0.30 + 0.10 * 0.40
    assert pt == pytest.approx(expected)
    assert lo == pytest.approx(expected)
    assert hi == pytest.approx(expected)

    # Zero dead-zone check across boundary cases (exact protocol specification parity)
    assert mod.classify_rope_zone(0.08, 0.04, 0.12) == "Zone_1_Empirical_Certification"
    assert mod.classify_rope_zone(0.10, 0.05, 0.149) == "Zone_1_Empirical_Certification"
    assert mod.classify_rope_zone(0.10, 0.05, 0.15) == "Zone_3_Conditional_Regime_Dependent"
    assert mod.classify_rope_zone(0.14, 0.10, 0.18) == "Zone_3_Conditional_Regime_Dependent"
    assert mod.classify_rope_zone(0.21, 0.15, 0.27) == "Zone_3_Conditional_Regime_Dependent"
    assert mod.classify_rope_zone(0.22, 0.16, 0.28) == "Zone_2_Mandatory_Verification_Gate"

    # Overlapping-journal BC vs CC bootstrap test + centered p-value floor
    va = np.array([0.25, 0.26, 0.24, 0.27, 0.25])
    vb = np.array([0.10, 0.11, 0.09, 0.12, 0.10])
    res_ov = mod.journal_clustered_group_diff_test(va, ["J1", "J2", "J3", "J4", "J5"], vb, ["J1", "J2", "J3", "J6", "J7"], n_reps=300)
    assert res_ov["unified_cluster_resample"] is True
    assert res_ov["boot_centered_p"] >= (1.0 / 301.0) - 1e-4
