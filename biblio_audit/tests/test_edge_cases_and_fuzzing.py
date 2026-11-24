"""Exhaustive edge-case, degenerate-topology, and Hypothesis property-based fuzzing suite."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import igraph as ig
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from biblio_audit.cli import run_cli
from biblio_audit.engine import (
    SUPPORTED_OBJECTIVES,
    _leiden_partition,
    audit_artifact,
    edges_to_igraph,
    evaluate_counting_sensitivity,
)
from biblio_audit.exceptions import FormatDetectionError, InsufficientNetworkError
from biblio_audit.metrics import (
    adjusted_element_centric_similarity,
    element_centric_scores,
    element_centric_similarity,
    expected_ecs_fixed_marginals,
    hungarian_core_match,
    landmark_and_tfidf_retention,
    variation_of_information,
)
from biblio_audit.networks import NetworkBuildArtifact, build_coupling_matrices, build_network_artifact
from biblio_audit.nulls import (
    _leiden_null_partition,
    causal_temporal_bipartite_rewire,
    curveball_rewire,
    evaluate_bipartite_null,
)
from biblio_audit.parsers import (
    load_corpus_files,
    normalize_reference_key,
    split_scopus_references,
)
from biblio_audit.reporting import export_vosviewer_files, render_latex_table, render_markdown_table

DEMO_SCOPUS = Path(__file__).resolve().parents[1] / "examples" / "demo_scopus.csv"


# ---------------------------------------------------------------------------
# 1. Hypothesis Property-Based Mathematical Invariants
# ---------------------------------------------------------------------------


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n_rows=st.integers(min_value=2, max_value=10),
    n_cols=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=100_000),
    net=st.sampled_from(["bc", "cc"]),
    scheme=st.sampled_from(["vosviewer", "refined"]),
)
def test_hyp_build_coupling_matrices_invariants(
    n_rows: int, n_cols: int, seed: int, net: str, scheme: str
) -> None:
    rng = np.random.default_rng(seed)
    c_mat = (rng.random((n_rows, n_cols)) < 0.45).astype(float)
    w_full, w_frac = build_coupling_matrices(c_mat, network_type=net, frac_scheme=scheme)

    expected_n = n_cols if net == "cc" else n_rows
    assert w_full.shape == (expected_n, expected_n)
    assert w_frac.shape == (expected_n, expected_n)
    assert np.allclose(w_full, w_full.T, atol=1e-12)
    assert np.allclose(w_frac, w_frac.T, atol=1e-12)
    assert np.allclose(np.diag(w_full), 0.0)
    assert np.allclose(np.diag(w_frac), 0.0)
    assert np.all(w_frac >= -1e-12)
    assert np.all(w_frac <= w_full + 1e-12)

    if scheme == "refined":
        arr = c_mat.T if net == "cc" else c_mat
        n_active_events = int(np.sum(arr.sum(axis=0) > 1.0))
        assert float(np.triu(w_frac, 1).sum()) == pytest.approx(float(n_active_events), abs=1e-10)


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n_nodes=st.integers(min_value=4, max_value=12),
    n_events=st.integers(min_value=2, max_value=10),
    deg_const=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hyp_constant_degree_proportionality_and_scale_invariance(
    n_nodes: int, n_events: int, deg_const: int, seed: int
) -> None:
    d = min(deg_const, n_nodes)
    rng = np.random.default_rng(seed)
    c_mat = np.zeros((n_nodes, n_events), dtype=float)
    for e in range(n_events):
        chosen = rng.choice(n_nodes, size=d, replace=False)
        c_mat[chosen, e] = 1.0

    w_full, w_frac = build_coupling_matrices(c_mat, network_type="bc", frac_scheme="vosviewer")
    assert np.allclose(w_frac, w_full / float(d - 1), atol=1e-12)

    f_edges = {(i, j): float(w_full[i, j]) for i in range(n_nodes) for j in range(i + 1, n_nodes) if w_full[i, j] > 0}
    r_edges = {(i, j): float(w_frac[i, j]) for i in range(n_nodes) for j in range(i + 1, n_nodes) if w_frac[i, j] > 0}

    for norm in ("association_strength", "fractionalization"):
        gf = edges_to_igraph(n_nodes, f_edges, normalization=norm)
        gr = edges_to_igraph(n_nodes, r_edges, normalization=norm)
        assert np.allclose(gf.es["weight"], gr.es["weight"], rtol=1e-11, atol=1e-12)

    # Leiden_RBConfig (modularity) is strictly scale-invariant in edge weights
    gf_raw = edges_to_igraph(n_nodes, f_edges, normalization="none")
    gr_raw = edges_to_igraph(n_nodes, r_edges, normalization="none")
    mem = [i % 2 for i in range(n_nodes)]
    assert gf_raw.modularity(mem, weights="weight") == pytest.approx(
        gr_raw.modularity(mem, weights="weight"), abs=1e-12
    )


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n_nodes=st.integers(min_value=3, max_value=16),
    extra_edges=st.integers(min_value=0, max_value=25),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hyp_vos_frac_cpm_total_weight_conservation(
    n_nodes: int, extra_edges: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    # Build random spanning tree so graph is guaranteed connected (zero isolated nodes)
    perm = rng.permutation(n_nodes).tolist()
    edges: dict[tuple[int, int], float] = {}
    for k in range(1, n_nodes):
        u = perm[k]
        v = perm[int(rng.integers(0, k))]
        key = (u, v) if u < v else (v, u)
        edges[key] = float(rng.uniform(0.1, 10.0))
    for _ in range(extra_edges):
        u, v = sorted(rng.choice(n_nodes, size=2, replace=False).tolist())
        edges[(u, v)] = float(rng.uniform(0.1, 10.0))

    g = edges_to_igraph(n_nodes, edges, normalization="fractionalization")
    assert sum(g.es["weight"]) == pytest.approx(0.5 * n_nodes * n_nodes, rel=1e-11)


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hyp_ecs_aecs_and_bruteforce_permutation_mean(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m1 = rng.integers(0, n, size=n).tolist()
    m2 = rng.integers(0, n, size=n).tolist()

    ecs_12 = element_centric_similarity(m1, m2)
    ecs_21 = element_centric_similarity(m2, m1)
    assert ecs_12 == pytest.approx(ecs_21, abs=1e-12)
    assert 0.0 <= ecs_12 <= 1.0 + 1e-12
    assert element_centric_similarity(m1, m1) == pytest.approx(1.0, abs=1e-12)
    assert adjusted_element_centric_similarity(m1, m1) == pytest.approx(1.0, abs=1e-12)
    assert adjusted_element_centric_similarity(m1, m2) == pytest.approx(
        adjusted_element_centric_similarity(m2, m1), abs=1e-12
    )

    # Exact verification against brute-force mean over all N! permutations
    exact_e = expected_ecs_fixed_marginals(m1, m2)
    perms = list(itertools.permutations(m2))
    brute_e = float(np.mean([element_centric_similarity(m1, list(p)) for p in perms]))
    assert exact_e == pytest.approx(brute_e, abs=1e-12)


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hyp_vi_metric_axioms_and_triangle_inequality(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, max(1, n), size=n).tolist()
    b = rng.integers(0, max(1, n), size=n).tolist()
    c = rng.integers(0, max(1, n), size=n).tolist()

    vi_aa, nvi_aa = variation_of_information(a, a)
    assert vi_aa == pytest.approx(0.0, abs=1e-12)
    assert nvi_aa == pytest.approx(0.0, abs=1e-12)

    vi_ab, nvi_ab = variation_of_information(a, b)
    vi_ba, nvi_ba = variation_of_information(b, a)
    vi_bc, _ = variation_of_information(b, c)
    vi_ac, _ = variation_of_information(a, c)

    assert vi_ab >= 0.0 and 0.0 <= nvi_ab <= 1.0
    assert vi_ab == pytest.approx(vi_ba, abs=1e-12)
    assert nvi_ab == pytest.approx(nvi_ba, abs=1e-12)
    assert vi_ac <= vi_ab + vi_bc + 1e-12


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n_r=st.integers(min_value=2, max_value=10),
    n_c=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=100_000),
)
def test_hyp_curveball_exact_degree_conservation(n_r: int, n_c: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bip = [(i, j) for i in range(n_r) for j in range(n_c) if rng.random() < 0.35]
    rw, rate = curveball_rewire(bip, n_steps=200, rng=rng)
    assert 0.0 <= rate <= 1.0
    assert len(set(rw)) == len(rw)
    assert Counter(u for u, _ in rw) == Counter(u for u, _ in bip)
    assert Counter(v for _, v in rw) == Counter(v for _, v in bip)


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    n_r=st.integers(min_value=2, max_value=10),
    n_c=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=100_000),
    nt=st.sampled_from(["bibliographic_coupling", "co_citation"]),
)
def test_hyp_causal_temporal_rewire_degrees_and_chronology(
    n_r: int, n_c: int, seed: int, nt: str
) -> None:
    rng = np.random.default_rng(seed)
    node_years = {i: int(rng.integers(2015, 2026)) for i in range(n_r)}
    event_years = {e: int(rng.integers(2015, 2026)) for e in range(n_c)}
    if nt == "bibliographic_coupling":
        bip = [(i, e) for i in range(n_r) for e in range(n_c) if event_years[e] <= node_years[i] and rng.random() < 0.4]
    else:
        bip = [(i, e) for i in range(n_r) for e in range(n_c) if node_years[i] <= event_years[e] and rng.random() < 0.4]

    rw, rate = causal_temporal_bipartite_rewire(bip, node_years, event_years, nt, swaps_per_edge=6, rng=rng)
    assert 0.0 <= rate <= 1.0
    assert len(set(rw)) == len(rw)
    assert Counter(u for u, _ in rw) == Counter(u for u, _ in bip)
    assert Counter(v for _, v in rw) == Counter(v for _, v in bip)
    for n_i, e_i in rw:
        if nt == "bibliographic_coupling":
            assert event_years[e_i] <= node_years[n_i]
        else:
            assert node_years[n_i] <= event_years[e_i]


# ---------------------------------------------------------------------------
# 2. Full 48-Combination Matrix Test (4 nets x 3 objectives x 2 schemes x 2 scopes)
# ---------------------------------------------------------------------------


def test_full_48_combination_matrix_on_demo_scopus() -> None:
    works, _ = load_corpus_files([DEMO_SCOPUS])
    combos = list(
        itertools.product(
            ("bc", "cc", "ca", "kw"),
            SUPPORTED_OBJECTIVES,
            ("vosviewer", "refined"),
            ("full-corpus", "selected-core"),
        )
    )
    assert len(combos) == 48

    # Cache NetworkBuildArtifact per (net, scheme, scope) so 16 builds feed all 48 evaluations cleanly
    for net, obj, scheme, scope in combos:
        art = build_network_artifact(
            works,
            network_type=net,
            core_top_n=150,
            min_occurrences=1,
            denom_scope=scope,
            frac_scheme=scheme,
        )
        if net in {"ca", "kw"} and scope == "selected-core":
            assert any("DenomScopeWarning" in w for w in art.warnings)
        res = audit_artifact(art, objective=obj, n_seeds=2, n_random_starts=2, min_cluster_size=3)
        assert res.n_lcc >= 3
        assert math.isfinite(res.delta_count_mid) and -1.0 <= res.delta_count_mid <= 1.0
        assert math.isfinite(res.delta_aecs_mid) and -1.0 <= res.delta_aecs_mid <= 1.0
        assert math.isfinite(res.nvi_cross) and 0.0 <= res.nvi_cross <= 1.0
        assert 0.0 <= res.core_reassignment <= 1.0
        assert 0.0 <= res.landmark_top10_retention <= 1.0


# ---------------------------------------------------------------------------
# 3. Hostile Parser Inputs, Reference Stub Guards, Encodings & Richer Dedup
# ---------------------------------------------------------------------------


def test_scopus_bare_years_split_initials_fullnames_mononyms_and_stub_warning(tmp_path: Path) -> None:
    refs_cell = (
        "(2026); (2024); (2015); "
        "Barendregt; W.; Biorn-Hansen A.; Andersson D., Carbon calculator study, J. Clean. Prod., (2020); "
        "Zhang A.; Danish; Chen R., AI paradox and environment, Environ. Res., (2025); "
        "Devlin Jacob; Chang Ming-Wei; Lee Kenton; Toutanova Kristina, BERT: Pre-training transformers, NAACL, (2019); "
        "Madsen D.&#X.008;.; Berg T.; Di Nardo M., Industry 5.0 overview, Appl. Syst. Innov., (2023); "
        "SEC, Climate disclosure guidance (Release Nos. 33-9106; 34-61469; FR-82), (2010); "
        "w; a; 1; pp. 92-93; et al."
    )
    chunks = split_scopus_references(refs_cell)
    assert any("Barendregt W." in c and "Biorn-Hansen A." in c for c in chunks)
    assert any("Zhang A.; Danish; Chen R." in c for c in chunks)
    assert any("Devlin Jacob; Chang Ming-Wei; Lee Kenton; Toutanova Kristina" in c for c in chunks)
    assert any("Madsen D." in c and "Berg T." in c for c in chunks)
    assert any("33-9106; 34-61469; FR-82" in c for c in chunks)

    # Bare years, single letters, bare numbers, and page stubs must NEVER survive normalization
    for bad in ("(2026)", "(2024)", "2015", "2022a", "w", "a", "W.", "f. k.", "1", "90", "pp. 92-93", "et al."):
        assert normalize_reference_key(bad) == ""

    csv_p = tmp_path / "noisy_scopus.csv"
    csv_p.write_text(
        "Authors,Author(s) ID,Title,Year,Cited by,DOI,EID,References\n"
        f'"Smith A.","1001;","Study One",2025,5,10.1000/n1,2-s2.0-1,"{refs_cell}"\n',
        encoding="utf-8",
    )
    wks, warns = load_corpus_files([csv_p])
    assert len(wks) == 1
    refs = wks[0]["referenced_works"]
    assert len(refs) == 5
    assert not any(len(r) < 4 for r in refs)
    assert not any(r.isdigit() for r in refs)
    assert any("UnidentifiableReferenceStubWarning" in w for w in warns)


def test_duplicate_doi_retains_richer_record(tmp_path: Path) -> None:
    csv_p = tmp_path / "dup_scopus.csv"
    csv_p.write_text(
        "Authors,Author(s) ID,Title,Year,Cited by,DOI,EID,Author Keywords,References\n"
        '"Smith A.","1001;","Sparse First Row",2025,1,10.1000/dup1,2-s2.0-1,"",""\n'
        '"Smith A.; Jones B.","1001; 1002;","Rich Second Row",2025,12,10.1000/dup1,2-s2.0-2,"ai; ml","Author X., Core Ref A, (2020); Author Y., Core Ref B, (2021)"\n'
        '"Doe C.","1003;","Rich First Row",2025,9,10.1000/dup2,2-s2.0-3,"esg","Author Z., Core Ref C, (2022)"\n'
        '"Doe C.","1003;","Sparse Second Row",2025,0,10.1000/dup2,2-s2.0-4,"",""\n',
        encoding="utf-8",
    )
    wks, warns = load_corpus_files([csv_p])
    assert len(wks) == 2
    by_doi = {w["doi"]: w for w in wks}
    assert len(by_doi["10.1000/dup1"]["referenced_works"]) == 2
    assert by_doi["10.1000/dup1"]["cited_by_count"] == 12
    assert len(by_doi["10.1000/dup2"]["referenced_works"]) == 1
    assert any("Deduplicated 2" in w for w in warns)


def test_latin1_cp1252_blank_rows_and_empty_file_guards_all_formats(tmp_path: Path) -> None:
    # Windows-1252 / Latin-1 CSV with accented characters, curly quotes, and trailing blank line
    latin_csv = tmp_path / "latin1_scopus.csv"
    content = (
        'Authors,Title,Year,Cited by,DOI,EID,Author Keywords,References\n'
        '"Müller J.; García L.","Étude sur l’IA — résumé",2024,7,10.1000/lat1,2-s2.0-9,"ia; éthique","Müller J., Arbeit über KI, (2020); García L., Estudio de IA, (2021)"\n'
        ',,,,,,,\n'
    )
    latin_csv.write_bytes(content.encode("cp1252"))
    wks, _ = load_corpus_files([latin_csv])
    assert len(wks) == 1
    assert "Müller" in wks[0]["authors"][0]["name"]
    assert "Étude" in wks[0]["title"]

    # Empty files raise FormatDetectionError cleanly across all 7 explicit formats
    empty_f = tmp_path / "empty.txt"
    empty_f.write_text("   \n\n", encoding="utf-8")
    for fmt in ("auto", "scopus", "wos", "wos_plaintext", "dimensions", "lens", "openalex", "openalex_csv"):
        with pytest.raises(FormatDetectionError):
            load_corpus_files([empty_f], fmt=fmt)

    # Malformed OpenAlex JSON authorships with None items do not crash
    oa_bad = tmp_path / "oa_null_auth.json"
    oa_bad.write_text(
        json.dumps([{"id": "W1", "title": "Null Auth Test", "publication_year": 2024, "authorships": [None, {"author": None}], "referenced_works": ["(2024)", "W2"]}]),
        encoding="utf-8",
    )
    wks_oa, warns_oa = load_corpus_files([oa_bad], fmt="openalex")
    assert len(wks_oa) == 1
    assert wks_oa[0]["referenced_works"] == ["https://openalex.org/W2"]
    assert any("UnidentifiableReferenceStubWarning" in w for w in warns_oa)


# ---------------------------------------------------------------------------
# 4. Edge Cases across Networks, Metrics, Engine, Nulls, and Reporting
# ---------------------------------------------------------------------------


def test_edges_to_igraph_isolated_nodes_tiny_weights_and_validation() -> None:
    # Isolated degree-0 nodes (nodes 2, 3, 4) safe under all 3 normalizations
    for norm in ("none", "association_strength", "fractionalization"):
        g = edges_to_igraph(5, {(0, 1): 2.5}, normalization=norm)
        assert g.vcount() == 5 and g.ecount() == 1
        assert math.isfinite(g.es["weight"][0]) and g.es["weight"][0] > 0.0

    # Tiny positive weights (1e-200) must NOT trigger ZeroDivisionError from du * dv underflow
    g_tiny = edges_to_igraph(3, {(0, 1): 1e-200, (1, 2): 1e-200}, normalization="association_strength")
    assert all(math.isfinite(w) and w > 0.0 for w in g_tiny.es["weight"])

    # Empty graph still validates normalization string & bounds
    g0 = edges_to_igraph(0, {}, normalization="association_strength")
    assert g0.vcount() == 0 and g0.ecount() == 0
    with pytest.raises(ValueError):
        edges_to_igraph(3, {}, normalization="bogus_norm")
    with pytest.raises(ValueError):
        edges_to_igraph(3, {(0, 5): 1.0}, normalization="none")
    with pytest.raises(ValueError):
        edges_to_igraph(-1, {}, normalization="none")


def test_metrics_degenerate_partitions_and_non_english_titles() -> None:
    # N=0, N=1, N=2, K=1, K=N
    assert element_centric_scores([], []).size == 0
    assert element_centric_similarity([], []) == 1.0
    assert expected_ecs_fixed_marginals([], []) == 1.0
    assert adjusted_element_centric_similarity([], []) == 1.0
    assert variation_of_information([], []) == (0.0, 0.0)
    assert hungarian_core_match([], []) == (1.0, {})

    assert adjusted_element_centric_similarity([0], [0]) == 1.0
    assert variation_of_information([0], [0]) == (0.0, 0.0)
    assert variation_of_information([0, 0], [0, 1]) == pytest.approx((1.0, 1.0))

    # K=1 vs K=1 and K=N vs K=N (where 1 - E[ECS] == 0) must never produce NaN/ZeroDivisionError
    assert adjusted_element_centric_similarity([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    assert adjusted_element_centric_similarity([0, 1, 2, 3], [3, 2, 1, 0]) == 1.0
    assert adjusted_element_centric_similarity([0, 0, 0, 0], [0, 1, 2, 3]) == pytest.approx(0.0, abs=1e-12)

    # Empty titles, stopword-only titles, and non-English titles in TF-IDF
    m1 = [0] * 6 + [1] * 6
    m2 = [0] * 5 + [1] * 7
    recs_empty = [{"title": "", "cited_by_count": i, "occurrences": 1} for i in range(12)]
    lm_e, tf_e = landmark_and_tfidf_retention(m1, m2, recs_empty, network_type="bibliographic_coupling", min_cluster_size=3)
    assert 0.0 <= lm_e <= 1.0 and tf_e is None

    recs_stop = [{"title": "the and of in to for with on at from by", "cited_by_count": i, "occurrences": 1} for i in range(12)]
    _, tf_s = landmark_and_tfidf_retention(m1, m2, recs_stop, network_type="bibliographic_coupling", min_cluster_size=3)
    assert tf_s is None

    recs_vi = [
        {"title": f"Nghiên cứu trí tuệ nhân tạo bền vững doanh nghiệp {i % 3}", "cited_by_count": i, "occurrences": 1}
        for i in range(12)
    ]
    lm_vi, tf_vi = landmark_and_tfidf_retention(m1, m2, recs_vi, network_type="bibliographic_coupling", min_cluster_size=3)
    assert 0.0 <= lm_vi <= 1.0 and tf_vi is not None and 0.0 <= tf_vi <= 1.0


def test_engine_degenerate_topologies_constant_d2_and_options(tmp_path: Path) -> None:
    # 1. N <= 5 or min_cluster_size > n_lcc raises InsufficientNetworkError
    small_recs = [{"id": f"d{i}", "label": f"d{i}", "title": f"Doc {i}", "cited_by_count": i, "occurrences": 1} for i in range(4)]
    small_e = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    with pytest.raises(InsufficientNetworkError):
        evaluate_counting_sensitivity(small_recs, small_e, small_e, n_seeds=2, n_random_starts=1, min_cluster_size=5)

    # 2. Complete clique K_15 -> K_full = 1 -> DEGENERATE_K_LT_2
    k15_recs = [{"id": f"k{i}", "label": f"k{i}", "title": f"Clique node {i}", "cited_by_count": i, "occurrences": 1} for i in range(15)]
    k15_e = {(i, j): 1.0 for i in range(15) for j in range(i + 1, 15)}
    res_clique = evaluate_counting_sensitivity(
        k15_recs, k15_e, k15_e, objective="VOS_AS_CPM", n_seeds=2, n_random_starts=1, min_cluster_size=3
    )
    assert res_clique.degenerate_k is True
    assert res_clique.triage_band == "DEGENERATE_K_LT_2"

    # 3. Star hub topology runs cleanly without crashes
    star_recs = [{"id": f"s{i}", "label": f"s{i}", "title": f"Star {i}", "cited_by_count": i, "occurrences": 1} for i in range(12)]
    star_e = {(0, i): 1.0 for i in range(1, 12)}
    res_star = evaluate_counting_sensitivity(
        star_recs, star_e, star_e, objective="Leiden_RBConfig", n_seeds=2, n_random_starts=1, min_cluster_size=3, merge_small_clusters=False
    )
    assert res_star.n_lcc == 12 and math.isfinite(res_star.delta_count_mid)

    # 4. Disconnected 3-component graph (10 + 10 + 10 = 30 nodes -> LCC = 10/30 = 33.3% < 50%)
    disc_recs = [{"id": f"c{i}", "label": f"c{i}", "title": f"Comp {i}", "cited_by_count": i, "occurrences": 1} for i in range(30)]
    disc_e: dict[tuple[int, int], float] = {}
    for base in (0, 10, 20):
        # Two dense 5-cliques joined by 1 weak bridge inside each component
        for i in range(base, base + 5):
            for j in range(i + 1, base + 5):
                disc_e[(i, j)] = 5.0
        for i in range(base + 5, base + 10):
            for j in range(i + 1, base + 10):
                disc_e[(i, j)] = 5.0
        disc_e[(base + 4, base + 5)] = 0.5
    res_disc = evaluate_counting_sensitivity(
        disc_recs, disc_e, disc_e, objective="VOS_AS_CPM", n_seeds=2, n_random_starts=1, min_cluster_size=3
    )
    assert res_disc.n_lcc == 10
    assert res_disc.lcc_fraction < 0.50
    assert any("FRAGMENTED_LCC_LT_50PCT" in w for w in res_disc.warnings)
    assert res_disc.triage_band == "LOW_SENSITIVITY_GUARDED"

    # 5. Constant event degree d_e = 2 -> exact zero distortion (delta_count == 0.0)
    const_works = []
    # Two modular 6-doc communities where every shared ref is cited by EXACTLY 2 docs
    pairs = [(i, j) for base in (0, 6) for i in range(base, base + 6) for j in range(i + 1, base + 6)] + [(0, 6)]
    doc_refs: list[list[str]] = [[] for _ in range(12)]
    for e_i, (u, v) in enumerate(pairs):
        doc_refs[u].append(f"ref_pair_{e_i}")
        doc_refs[v].append(f"ref_pair_{e_i}")
    for i in range(12):
        const_works.append({"id": f"w{i}", "title": f"Constant degree doc {i}", "publication_year": 2023, "cited_by_count": 5, "referenced_works": doc_refs[i]})
    art_const = build_network_artifact(const_works, network_type="bc", core_top_n=0)  # top_n=0 unlimited!
    assert art_const.n_core_selected == 12
    assert art_const.full_edges == art_const.frac_edges
    res_const = audit_artifact(art_const, objective="VOS_AS_CPM", n_seeds=2, n_random_starts=2, min_cluster_size=3)
    assert res_const.delta_count_mid == pytest.approx(0.0, abs=1e-9)
    assert res_const.delta_aecs_mid == pytest.approx(0.0, abs=1e-9)

    # 6. CLI options: --top-n 0, --seeds 2, --starts 1, --no-merge
    payload = run_cli([
        "--input", str(DEMO_SCOPUS),
        "--network", "bc",
        "--top-n", "0",
        "--seeds", "2",
        "--starts", "1",
        "--no-merge",
        "--min-cluster-size", "3",
    ])
    assert payload["parameters"]["top_n"] is None
    assert payload["parameters"]["merge_small_clusters"] is False
    assert payload["audit"]["n_core_selected"] == 36


def test_leiden_tie_oscillation_non_hang_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify bounded monotone convergence terminates immediately on tie-oscillating Leiden calls."""
    g = edges_to_igraph(6, {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (3, 4): 1.0, (4, 5): 1.0}, normalization="none")
    calls = {"n": 0}

    class _OscillatingClustering:
        def __init__(self, flip: int) -> None:
            self.membership = [0, 0, 0, 1, 1, 1] if flip == 0 else [1, 1, 1, 0, 0, 0]
            self.quality = 0.5  # exact tie every iteration!

    def _fake_leiden(self: ig.Graph, **kwargs: object) -> _OscillatingClustering:
        calls["n"] += 1
        return _OscillatingClustering(calls["n"] % 2)

    monkeypatch.setattr(ig.Graph, "community_leiden", _fake_leiden)
    mem, q = _leiden_partition(g, "CPM", 1.0, seed=0, max_iterations=10)
    assert calls["n"] <= 2  # stops on 2nd call when Delta Q <= 1e-12
    assert q == 0.5 and len(mem) == 6

    calls["n"] = 0
    mem2, q2 = _leiden_null_partition(g, "CPM", seed=0, max_iterations=10)
    assert calls["n"] <= 2
    assert q2 == 0.5 and len(mem2) == 6


def test_nulls_edge_cases_empty_1year_rigid_and_n_rewires_1() -> None:
    # Empty bipartite edges
    assert curveball_rewire([]) == ([], 0.0)
    assert causal_temporal_bipartite_rewire([], {}, {}, "bibliographic_coupling") == ([], 0.0)

    # Duplicate-edge tiny input (len == 2 raw, len == 1 deduplicated) must NOT crash gen.integers(0, 0)
    rw_dup, rate_dup = causal_temporal_bipartite_rewire([(0, 0), (0, 0)], {0: 2024}, {0: 2024}, "bibliographic_coupling")
    assert rw_dup == [(0, 0)] and rate_dup == 0.0

    # Complete bipartite graph K_{4,4} has zero valid trades -> LowMixingWarning
    k44_bip = [(i, j) for i in range(4) for j in range(4)]
    rw_full, rate_full = curveball_rewire(k44_bip, n_steps=50)
    assert rate_full == 0.0 and len(rw_full) == 16

    # 1-year rigid corpus with n_rewires=1 runs cleanly and emits LowMixingWarning
    art_rigid = NetworkBuildArtifact(
        network_type="bibliographic_coupling",
        frac_scheme="vosviewer",
        denom_scope="selected-core",
        node_records=[{"id": f"w{i}", "label": f"w{i}", "title": f"T{i}", "cited_by_count": 1, "occurrences": 1, "publication_year": 2024} for i in range(4)],
        full_edges={(i, j): 4.0 for i in range(4) for j in range(i + 1, 4)},
        frac_edges={(i, j): 4.0 / 3.0 for i in range(4) for j in range(i + 1, 4)},
        bipartite_edges=k44_bip,
        event_degrees={j: 4 for j in range(4)},
        node_years={i: 2024 for i in range(4)},
        event_years={j: 2024 for j in range(4)},
        n_eligible_nodes=4,
        n_core_selected=4,
        event_degree_gini=0.0,
        tls_core_turnover_jaccard=None,
        warnings=[],
    )
    null_res = evaluate_bipartite_null(art_rigid, edges_to_igraph, n_rewires=1, n_starts=1, seed=7)
    assert null_res.method == "causal_temporal_mcmc"
    assert null_res.n_rewires == 1
    assert any("LowMixingWarning" in w for w in null_res.warnings)

    # Empty bipartite edges inside evaluate_bipartite_null must not crash
    art_empty = NetworkBuildArtifact(
        network_type="co_authorship",
        frac_scheme="vosviewer",
        denom_scope="full-corpus",
        node_records=[{"id": "a0", "label": "a0"}, {"id": "a1", "label": "a1"}],
        full_edges={},
        frac_edges={},
        bipartite_edges=[],
        event_degrees={},
        node_years={},
        event_years={},
        n_eligible_nodes=2,
        n_core_selected=2,
        event_degree_gini=0.0,
        tls_core_turnover_jaccard=None,
        warnings=[],
    )
    empty_null = evaluate_bipartite_null(art_empty, edges_to_igraph, n_rewires=1, n_starts=1, seed=7)
    assert empty_null.p_value_rb == 1.0


def test_reporting_special_chars_and_vosviewer_sanitization(tmp_path: Path) -> None:
    works, _ = load_corpus_files([DEMO_SCOPUS])
    works[0]["title"] = "\t\r\n"  # blank after stripping -> falls back to node_id
    works[1]["title"] = "Line1\nLine2\tCol2\rLine3"
    art = build_network_artifact(works, network_type="bc", core_top_n=20)
    res = audit_artifact(art, n_seeds=2, n_random_starts=1, min_cluster_size=3)
    res.warnings.append(r"Warn & % _ # $ { } ~ ^ \ end")

    # LaTeX escaping of all 10 special characters
    tex = render_latex_table(r"my_file&100%#$a{b}~^c\d.csv", 20, [res])
    for esc in (r"\_", r"\&", r"\%", r"\#", r"\$", r"\{", r"\}", r"\textasciitilde{}", r"\textasciicircum{}", r"\textbackslash{}"):
        assert esc in tex

    # Markdown table pipe/newline sanitization
    md = render_markdown_table("pipe|name\nnewline.csv", 20, res)
    assert r"pipe\|name newline.csv" in md

    # VOSviewer export has strictly 8 tab-separated columns per row and zero embedded newlines/tabs
    vos_files = export_vosviewer_files(tmp_path / "vos", res)
    map_lines = Path(vos_files["vos_map"]).read_text(encoding="utf-8").splitlines()
    assert len(map_lines) == res.n_lcc + 1
    for ln in map_lines[1:]:
        cols = ln.split("\t")
        assert len(cols) == 8
        assert cols[1].strip() != ""  # blank title fell back safely
