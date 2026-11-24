"""Replication regression test verifying this study Box 1 (`10.3390/admsci13030075`) parity against biblio-audit v1.0."""

from __future__ import annotations

from pathlib import Path

import pytest

from biblio_audit.engine import audit_artifact
from biblio_audit.networks import build_network_artifact
from biblio_audit.parsers import load_corpus


def test_study_box1_backward_parity_if_corpus_present() -> None:
    """Verify this study Box 1 (`10.3390/admsci13030075`) reproduces unified igraph Delta_count ~= 0.3722."""
    corp_p = Path(__file__).resolve().parents[1] / "data" / "raw" / "repro_corpora"
    matches = list(corp_p.glob("*admsci13030075*.json")) if corp_p.exists() else []
    if not matches:
        pytest.skip("Local this study corpus cache not present")
    wks = load_corpus(matches[0], fmt="openalex")
    art = build_network_artifact(wks, network_type="bc", core_top_n=150, denom_scope="selected-core")
    res = audit_artifact(art, objective="VOS_AS_CPM", n_seeds=20, n_random_starts=10, min_cluster_size=5)
    assert res.k_full_major == 5 and res.k_frac_matched_major == 5
    assert res.exact_k_matched is True
    assert res.gamma_plateau_mid == pytest.approx(0.8979, abs=0.001)
    assert res.delta_count_mid == pytest.approx(0.3722, abs=0.002)
    assert res.delta_aecs_mid == pytest.approx(0.3948, abs=0.002)
    assert res.core_reassignment == pytest.approx(0.2762, abs=0.002)
    assert res.landmark_top10_retention == pytest.approx(0.6743, abs=0.002)
    assert res.triage_band == "HIGH_SENSITIVITY_REPORT_FRACTIONAL"
