"""`biblio-audit`: VOSviewer-parity full-vs.-fractional counting sensitivity & clustering stability auditor."""

from __future__ import annotations

__version__ = "1.0.0"

from .engine import (
    SUPPORTED_OBJECTIVES,
    CountingAuditResult,
    NodeStabilityRow,
    audit_artifact,
    classify_single_map_band,
    edges_to_igraph,
    evaluate_counting_sensitivity,
    extract_shared_lcc_edges,
    merge_small_clusters_cwts,
)
from .exceptions import (
    BiblioAuditError,
    FormatDetectionError,
    InsufficientNetworkError,
    MissingFieldError,
)
from .metrics import (
    adjusted_element_centric_similarity,
    align_fractional_to_full,
    element_centric_scores,
    element_centric_similarity,
    expected_ecs_fixed_marginals,
    hungarian_core_match,
    select_medoid_partition,
    variation_of_information,
)
from .networks import (
    NetworkBuildArtifact,
    build_coupling_matrices,
    build_network_artifact,
    build_sparse_edges_from_works,
    gini_coefficient,
    reference_degree_gini,
)
from .nulls import (
    BipartiteNullResult,
    causal_temporal_bipartite_rewire,
    curveball_rewire,
    evaluate_bipartite_null,
)
from .parsers import (
    SUPPORTED_FORMATS,
    canonicalize_doi,
    detect_format,
    load_corpus,
    load_corpus_files,
    load_thesaurus,
    split_scopus_references,
)

__all__ = [
    "__version__",
    "SUPPORTED_FORMATS",
    "SUPPORTED_OBJECTIVES",
    "BiblioAuditError",
    "FormatDetectionError",
    "InsufficientNetworkError",
    "MissingFieldError",
    "NetworkBuildArtifact",
    "CountingAuditResult",
    "NodeStabilityRow",
    "BipartiteNullResult",
    "load_corpus",
    "load_corpus_files",
    "load_thesaurus",
    "detect_format",
    "canonicalize_doi",
    "split_scopus_references",
    "build_coupling_matrices",
    "build_network_artifact",
    "build_sparse_edges_from_works",
    "gini_coefficient",
    "reference_degree_gini",
    "edges_to_igraph",
    "extract_shared_lcc_edges",
    "merge_small_clusters_cwts",
    "evaluate_counting_sensitivity",
    "audit_artifact",
    "classify_single_map_band",
    "element_centric_scores",
    "element_centric_similarity",
    "expected_ecs_fixed_marginals",
    "adjusted_element_centric_similarity",
    "variation_of_information",
    "hungarian_core_match",
    "align_fractional_to_full",
    "select_medoid_partition",
    "curveball_rewire",
    "causal_temporal_bipartite_rewire",
    "evaluate_bipartite_null",
]
