"""Verification tests for Curveball, causal temporal MCMC rewiring, and exchangeable bipartite null evaluation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from biblio_audit.engine import edges_to_igraph
from biblio_audit.networks import build_network_artifact
from biblio_audit.nulls import (
    causal_temporal_bipartite_rewire,
    curveball_rewire,
    evaluate_bipartite_null,
)


def test_curveball_preserves_exact_row_and_column_degrees() -> None:
    rng = np.random.default_rng(101)
    bip = [(i, j) for i in range(15) for j in range(30) if rng.random() < 0.25]
    rw, rate = curveball_rewire(bip, n_steps=4000, rng=rng)
    assert rate > 0.05
    assert Counter(u for u, _ in rw) == Counter(u for u, _ in bip)
    assert Counter(v for _, v in rw) == Counter(v for _, v in bip)
    assert len(set(rw)) == len(rw)  # strictly simple bipartite graph


def test_causal_temporal_rewire_preserves_degrees_and_chronology() -> None:
    rng = np.random.default_rng(202)
    node_years = {i: 2018 + (i % 6) for i in range(18)}
    event_years = {e: 2015 + (e % 6) for e in range(25)}
    bip = [
        (i, e)
        for i in range(18)
        for e in range(25)
        if event_years[e] <= node_years[i] and rng.random() < 0.30
    ]
    rw, rate = causal_temporal_bipartite_rewire(bip, node_years, event_years, "bibliographic_coupling", swaps_per_edge=10, rng=rng)
    assert rate > 0.0
    assert Counter(u for u, _ in rw) == Counter(u for u, _ in bip)
    assert Counter(v for _, v in rw) == Counter(v for _, v in bip)
    for n_i, e_i in rw:
        assert event_years[e_i] <= node_years[n_i]


def test_exchangeable_bipartite_null_detects_planted_community_structure() -> None:
    # Two strong modular communities of 12 docs each sharing community-specific refs + light bridge
    works = []
    for i in range(24):
        grp = 0 if i < 12 else 1
        refs = [f"r_{grp}_{k}" for k in range(6)] + (["r_bridge"] if i in {0, 12} else [])
        works.append({
            "id": f"w{i}",
            "title": f"Group {grp} paper {i}",
            "publication_year": 2022 + (i % 3),
            "cited_by_count": 10,
            "referenced_works": refs,
            "authors": [{"id": f"a_{grp}_{i % 4}", "name": f"Auth {grp}_{i % 4}"}, {"id": f"a_{grp}_{(i + 1) % 4}", "name": f"Auth {grp}_{(i + 1) % 4}"}],
            "author_keywords": [f"kw_{grp}_{k}" for k in range(4)] + (["kw_shared"] if i in {0, 12} else []),
            "index_keywords": [],
        })
    art_bc = build_network_artifact(works, network_type="bc", core_top_n=24)
    res_bc = evaluate_bipartite_null(art_bc, edges_to_igraph, n_rewires=19, n_starts=2, seed=42)
    assert res_bc.method == "causal_temporal_mcmc"
    assert res_bc.q_obs_rb_fixed_n > res_bc.q_null_rb_mean
    assert res_bc.p_value_rb <= 0.10

    art_kw = build_network_artifact(works, network_type="kw", core_top_n=24)
    res_kw = evaluate_bipartite_null(art_kw, edges_to_igraph, n_rewires=19, n_starts=2, seed=42)
    assert res_kw.method == "curveball"
    assert res_kw.p_value_rb <= 0.10


def test_full_corpus_null_event_degrees_and_undated_curveball_fallback() -> None:
    """Verify full-corpus event_degrees denominator parity in nulls and Curveball fallback on undated citation corpora."""
    # 12-author consortium where only A0..A5 enter top_n=6 -> event_degrees retains 12
    works_ca = []
    for i in range(10):
        grp = 0 if i < 5 else 1
        auths = [{"id": f"A{grp * 3 + k}", "name": f"A{grp * 3 + k}"} for k in range(3)]
        works_ca.append({"id": f"w{i}", "authors": auths, "publication_year": 2023, "cited_by_count": 5})
    works_ca.append({"id": "w_mega", "authors": [{"id": f"A{k}", "name": f"A{k}"} for k in range(12)], "publication_year": 2024, "cited_by_count": 20})
    art_ca = build_network_artifact(works_ca, network_type="ca", core_top_n=6, denom_scope="auto", min_occurrences=1)
    assert art_ca.denom_scope == "full-corpus"
    assert max(art_ca.event_degrees.values()) == 12
    res_ca = evaluate_bipartite_null(art_ca, edges_to_igraph, n_rewires=5, n_starts=1, seed=7)
    assert res_ca.method == "curveball"
    assert res_ca.q_obs_vos_norm == res_ca.q_obs_vos_norm  # finite

    # Undated BC corpus has empty node_years and falls back cleanly to Curveball with warning
    undated = [
        {"id": f"u{i}", "title": f"Undated {i}", "publication_year": None, "cited_by_count": 1, "referenced_works": [f"r_{i // 5}_{k}" for k in range(4)] + ["r_hub"]}
        for i in range(10)
    ]
    art_u = build_network_artifact(undated, network_type="bc", core_top_n=10)
    assert len(art_u.node_years) == 0
    res_u = evaluate_bipartite_null(art_u, edges_to_igraph, n_rewires=5, n_starts=1, seed=9)
    assert res_u.method == "curveball"
    assert any("MissingYearsFallbackWarning" in w for w in res_u.warnings)
