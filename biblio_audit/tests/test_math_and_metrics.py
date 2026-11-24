"""First-principles mathematical verification tests for counting algebra, VOS normalizations, ECS/AECS, VI, and Medoid."""

from __future__ import annotations

import numpy as np
import pytest

from biblio_audit.engine import edges_to_igraph, merge_small_clusters_cwts
from biblio_audit.metrics import (
    adjusted_element_centric_similarity,
    align_fractional_to_full,
    element_centric_scores,
    element_centric_similarity,
    expected_ecs_fixed_marginals,
    select_medoid_partition,
    variation_of_information,
)
from biblio_audit.networks import build_coupling_matrices, build_network_artifact


def test_perianes_rodriguez_2016_table4_and_table5_exact() -> None:
    # Exact 4-doc x 3-ref toy matrix from Perianes-Rodriguez, Waltman & van Eck (2016)
    c_mat = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    vf, vr = build_coupling_matrices(c_mat, "bc", frac_scheme="vosviewer")
    # Ref 0: n=3 -> 1/2; Ref 1: n=2 -> 1/1; Ref 2: n=3 -> 1/2
    assert vf[0, 1] == pytest.approx(2.0)
    assert vr[0, 1] == pytest.approx(1.5)
    assert vr[1, 2] == pytest.approx(1.0)
    assert np.allclose(np.diag(vr), 0.0)
    # Refined Leydesdorff-Park: Ref 0 -> 2/6=1/3, Ref 1 -> 2/2=1, Ref 2 -> 2/6=1/3 => sum(upper tri) = 3 events
    _, vref = build_coupling_matrices(c_mat, "bc", frac_scheme="refined")
    assert float(np.triu(vref, 1).sum()) == pytest.approx(3.0)


def test_ca_pre_truncation_denominator_prevents_consortium_inflation() -> None:
    # 15-author consortium paper where only A0 and A1 are prolific enough to enter top_n=2
    works = [
        {"id": f"w{i}", "authors": [{"id": "A0", "name": "A0"}, {"id": "A1", "name": "A1"}], "cited_by_count": 5, "publication_year": 2022}
        for i in range(4)
    ]
    consortium = [{"id": f"A{i}", "name": f"A{i}"} for i in range(15)]
    works.append({"id": "w_big", "authors": consortium, "cited_by_count": 50, "publication_year": 2023})

    art = build_network_artifact(works, network_type="ca", core_top_n=2, denom_scope="auto", max_authors=25)
    assert art.denom_scope == "full-corpus"
    # 4 two-author papers (weight 1.0 each) + 1 fifteen-author paper (weight 1/14, NOT 1.0!)
    assert art.full_edges[(0, 1)] == pytest.approx(5.0)
    assert art.frac_edges[(0, 1)] == pytest.approx(4.0 + 1.0 / 14.0, abs=1e-9)


def test_vos_fractionalization_exact_total_weight_identity() -> None:
    # Exact theorem: sum_{i<j} s_ij = (n/2) * sum_i (1/k_i * k_i) + ... = n^2 / 2 for any connected graph!
    rng = np.random.default_rng(42)
    n = 40
    edges: dict[tuple[int, int], float] = {}
    for i in range(n - 1):
        edges[(i, i + 1)] = float(rng.uniform(0.5, 5.0))
    for _ in range(120):
        u, v = sorted(rng.choice(n, size=2, replace=False).tolist())
        edges[(u, v)] = float(rng.uniform(0.2, 8.0))

    g = edges_to_igraph(n, edges, normalization="fractionalization")
    assert sum(g.es["weight"]) == pytest.approx(0.5 * n * n, rel=1e-12)


def test_ecs_bit_exact_against_dense_clusim_ppr_formula() -> None:
    rng = np.random.default_rng(7)
    n = 60
    m1 = rng.integers(0, 5, size=n).tolist()
    m2 = rng.integers(0, 6, size=n).tolist()
    alpha = 0.9

    # Explicit dense clusim/clusimelement.py lines 304-311 + line 219 `cL1`
    def _dense_ppr(mem: list[int]) -> np.ndarray:
        ppr = np.zeros((n, n), dtype=float)
        for c in set(mem):
            idxs = [i for i, v in enumerate(mem) if v == c]
            sz = len(idxs)
            block = (alpha / sz) * np.ones((sz, sz)) + np.eye(sz) * (1.0 - alpha)
            for a_i, u in enumerate(idxs):
                for b_i, v in enumerate(idxs):
                    ppr[u, v] = block[a_i, b_i]
        return ppr

    p1 = _dense_ppr(m1)
    p2 = _dense_ppr(m2)
    expected = 1.0 - (1.0 / (2.0 * alpha)) * np.sum(np.abs(p1 - p2), axis=1)
    fast = element_centric_scores(m1, m2)
    assert np.allclose(fast, expected, atol=1e-13)


def test_hypergeometric_expected_ecs_matches_monte_carlo() -> None:
    rng = np.random.default_rng(2026)
    m1 = [0] * 35 + [1] * 25 + [2] * 15
    m2 = [0] * 40 + [1] * 20 + [2] * 10 + [3] * 5
    exact = expected_ecs_fixed_marginals(m1, m2)
    arr2 = np.array(m2)
    mc = np.mean([element_centric_similarity(m1, rng.permutation(arr2).tolist()) for _ in range(4000)])
    assert abs(exact - float(mc)) < 0.005
    # Identical partitions have AECS = 1.0; independent random partitions center around 0.0
    assert adjusted_element_centric_similarity(m1, m1) == pytest.approx(1.0)


def test_vi_metric_axioms_and_medoid_and_surplus_alignment() -> None:
    a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    b = [0, 0, 1, 1, 1, 2, 2, 2, 0]
    c = [2, 2, 2, 0, 0, 1, 1, 1, 1]
    vi_aa, nvi_aa = variation_of_information(a, a)
    assert vi_aa == pytest.approx(0.0) and nvi_aa == pytest.approx(0.0)
    vi_ab, _ = variation_of_information(a, b)
    vi_ba, _ = variation_of_information(b, a)
    vi_bc, _ = variation_of_information(b, c)
    vi_ac, _ = variation_of_information(a, c)
    assert vi_ab == pytest.approx(vi_ba, abs=1e-12)
    assert vi_ac <= vi_ab + vi_bc + 1e-12  # triangle inequality

    # Medoid selects consensus partition over outlier seed 0
    parts = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2],  # outlier seed 0
        a,
        a,
        a,
    ]
    assert select_medoid_partition(parts) in {1, 2, 3}

    # Surplus fractional clusters (K_frac=4 > K_full=2) map deterministically without KeyError
    m_full = [0, 0, 0, 0, 1, 1, 1, 1]
    m_frac = [0, 0, 1, 1, 2, 2, 3, 3]
    aligned, mp = align_fractional_to_full(m_full, m_frac)
    assert len(aligned) == len(m_full)
    assert len(set(aligned)) == 4
    assert len(mp) == 4


def test_cwts_merge_supports_degree_volume_for_rbconfig() -> None:
    edges = {(0, 1): 10.0, (1, 2): 10.0, (0, 2): 10.0, (3, 4): 1.0, (2, 3): 1.0, (2, 4): 1.0, (4, 5): 1.0}
    g = edges_to_igraph(6, edges, normalization="none")
    merged = merge_small_clusters_cwts(g, [0, 0, 0, 1, 1, 2], min_cluster_size=3, use_degree_volume=True)
    assert len(set(merged)) <= 2


def test_nonfinite_inputs_keyword_thresholds_length_checks_and_ca_thesaurus() -> None:
    """Verify non-finite guards, partition length checks, KW min_occurrences, and CA display-name thesaurus."""
    from biblio_audit.metrics import hungarian_core_match

    # Non-finite values and mismatched partition lengths raise ValueError
    with pytest.raises(ValueError):
        edges_to_igraph(3, {(0, 1): float("nan")}, normalization="association_strength")
    with pytest.raises(ValueError):
        build_coupling_matrices(np.array([[1.0, float("inf")], [1.0, 1.0]]), "bc")
    with pytest.raises(ValueError):
        expected_ecs_fixed_marginals([0, 1, 1], [0, 1])
    with pytest.raises(ValueError):
        hungarian_core_match([0, 1, 1], [0, 1])

    # KW min_occurrences=1 keeps singleton keywords while min_occurrences=2 filters them
    works = [
        {"id": f"w{i}", "title": f"T{i}", "publication_year": 2023, "cited_by_count": 5, "authors": [{"id": "scopus_auid:1", "name": "Van Eck, N.J."}, {"id": "scopus_auid:2", "name": "Waltman, L."}], "author_keywords": ["alpha", "beta", f"unique_{i}"], "index_keywords": [], "referenced_works": ["r1", "r2"]}
        for i in range(4)
    ]
    art_kw1 = build_network_artifact(works, network_type="kw", min_occurrences=1, core_top_n=50)
    assert "unique_0" in {r["id"] for r in art_kw1.node_records}
    art_kw2 = build_network_artifact(works, network_type="kw", min_occurrences=2, core_top_n=50)
    kw2_ids = {r["id"] for r in art_kw2.node_records}
    assert "unique_0" not in kw2_ids and "alpha" in kw2_ids

    # Thesaurus matches CA display name even when author ID is Scopus AUID
    art_ca = build_network_artifact(works, network_type="ca", min_occurrences=1, thesaurus={"van eck, n.j.": "n. j. van eck"})
    assert "n. j. van eck" in {r["id"] for r in art_ca.node_records}
