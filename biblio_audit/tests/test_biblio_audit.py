"""Core smoke & ROPE guardrail tests for `biblio-audit` v1.0."""

from __future__ import annotations

import numpy as np

from biblio_audit.engine import classify_single_map_band
from biblio_audit.networks import build_coupling_matrices, reference_degree_gini


def test_matrix_identity_and_gini() -> None:
    c_mat = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
        ],
        dtype=float,
    )
    vf, vr = build_coupling_matrices(c_mat, "bibliographic_coupling")
    assert vf[0, 1] == 2.0
    assert abs(vr[0, 1] - 1.5) < 1e-12
    g = reference_degree_gini([{"referenced_works": ["r1", "r2"]}, {"referenced_works": ["r1", "r2", "r3"]}, {"referenced_works": ["r1"]}])
    assert 0.0 <= g <= 1.0


def test_rope_band_boundaries_and_guardrails() -> None:
    b0, f0 = classify_single_map_band(0.05, degenerate_k=True)
    assert b0 == "DEGENERATE_K_LT_2" and len(f0) == 1

    b1, f1 = classify_single_map_band(0.10, degenerate_k=False, lcc_fraction=0.95, min_within_ami=0.92, core_reassignment=0.08)
    assert b1 == "LOW_SENSITIVITY_PASS" and f1 == []

    # Fragmented CA LCC (<50%) must NEVER receive an unqualified LOW_SENSITIVITY_PASS!
    b_frag, f_frag = classify_single_map_band(0.05, degenerate_k=False, lcc_fraction=0.22)
    assert b_frag == "LOW_SENSITIVITY_GUARDED"
    assert any("FRAGMENTED_LCC_LT_50PCT" in f for f in f_frag)

    # High KW turnover / depressed within-AMI triggers dispersion warning
    b_kw, f_kw = classify_single_map_band(0.17, degenerate_k=False, lcc_fraction=0.95, min_within_ami=0.78, core_reassignment=0.37)
    assert b_kw == "MODERATE_SENSITIVITY_CHECK_BOTH"
    assert any("HIGH_TURNOVER_OR_SOLVER_DISPERSION" in f for f in f_kw)

    b2, _ = classify_single_map_band(0.20, degenerate_k=False)
    assert b2 == "HIGH_SENSITIVITY_REPORT_FRACTIONAL"
