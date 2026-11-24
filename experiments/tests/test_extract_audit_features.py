"""Unit tests for hardened Round-1 regex parser, Elsevier .txt support, and flat Excel export."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import openpyxl

SRC_DIR = Path(__file__).resolve().parents[1] / "src_audit"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_audit_features import (  # noqa: E402
    AUTHOR_FRAC_RE,
    DATA_CODE_RE,
    DATA_NEGATION_RE,
    GUIDE_TITLE_RE,
    NETWORK_FRAC_RE,
    REF_HEADING_RE,
    SCOPUS_RE,
    STRICT_QUERY_RE,
    WOS_RE,
    _first_positive_match,
    normalize_pdf_text,
    parse_pdf_features,
)


def test_negation_window_blocks_omitted_database_and_multiline_wrap() -> None:
    """Multi-line soft wrap must NOT sever negation ('Scopus\\nor other repositories were excluded')."""
    pages = [
        (3, "Data were extracted from Web of Science Core Collection. Scopus\nor other databases were excluded from this study."),
    ]
    sc_hit, _, _ = _first_positive_match(pages, SCOPUS_RE, apply_negation_guard=True)
    wos_hit, _, _ = _first_positive_match(pages, WOS_RE, apply_negation_guard=True)
    assert sc_hit is False
    assert wos_hit is True


def test_negation_window_blocks_alternative_wos_false_positive() -> None:
    """'The natural alternatives are Web of Science' must NOT flag uses_wos."""
    pages = [
        (4, "The Scopus database is chosen as the only database. The natural alternatives are Web of Science and Google Scholar."),
    ]
    sc_hit, _, _ = _first_positive_match(pages, SCOPUS_RE, apply_negation_guard=True)
    wos_hit, _, _ = _first_positive_match(pages, WOS_RE, apply_negation_guard=True)
    assert sc_hit is True
    assert wos_hit is False


def test_database_alias_ssci_and_pdf_normalization() -> None:
    """SSCI alias, soft hyphen \\xad, and curly quotes must normalize cleanly."""
    raw = "We queried the SSCI database using the string TITLE-ABS-KEY(\u201cperson\xad-organization\u201d OR \u201cPO fit\u201d)."
    norm = normalize_pdf_text(raw)
    assert "\xad" not in norm
    assert '"person-organization"' in norm
    wos_hit, _, _ = _first_positive_match([(1, norm)], WOS_RE, apply_negation_guard=True)
    q_hit, _, _ = _first_positive_match([(1, norm)], STRICT_QUERY_RE, apply_negation_guard=False)
    assert wos_hit is True
    assert q_hit is True


def test_separates_network_fractional_from_author_articles_fractionalized() -> None:
    """Biblioshiny 'Articles Fractionalized' must NOT trigger network fractional_disclosed."""
    pages = [(5, "Table 3 reports Articles Fractionalized per author. We used full counting rather than fractional counting in VOSviewer.")]
    net_hit, _, _ = _first_positive_match(pages, NETWORK_FRAC_RE, apply_negation_guard=True)
    auth_hit, _, _ = _first_positive_match(pages, AUTHOR_FRAC_RE, apply_negation_guard=False)
    assert net_hit is False
    assert auth_hit is True


def test_data_or_code_boilerplate_blocked_while_repo_url_accepted() -> None:
    """'upon reasonable request' / 'sourced from Scopus' must NOT flag; Zenodo/GitHub MUST flag."""
    bad = [(10, "Data availability: The datasets are available from the corresponding author upon reasonable request and sourced from Scopus.")]
    good = [(10, "Replication data and code are archived at https://github.com/example/biblio-audit and 10.5281/zenodo.123456.")]
    bad_hit, _, _ = _first_positive_match(bad, DATA_CODE_RE, apply_negation_guard=True, neg_pattern=DATA_NEGATION_RE)
    good_hit, _, _ = _first_positive_match(good, DATA_CODE_RE, apply_negation_guard=True, neg_pattern=DATA_NEGATION_RE)
    assert bad_hit is False
    assert good_hit is True


def test_ref_heading_skips_biblioshiny_table1_row() -> None:
    """Biblioshiny Table 1 row '7 References\\n59367' must NOT trigger References truncation."""
    table_text = "Table 1 Main Information\n7 References\n59367\nAuthors 1204\n"
    real_refs = "Discussion ends here.\nReferences\nSmith, J. (2020). Management study.\n"
    assert REF_HEADING_RE.search(table_text) is None
    assert REF_HEADING_RE.search(real_refs) is not None


def test_elsevier_plaintext_file_parsing_and_guide_filter(tmp_path: Path) -> None:
    """parse_pdf_features supports Elsevier Full-Text API .txt files seamlessly."""
    txt_file = tmp_path / "elsevier_sample.txt"
    txt_file.write_text(
        "Introduction\nWe queried Scopus using TITLE-ABS-KEY(\"green finance\" AND \"supply chain\") "
        "following PRISMA guidelines and built a bibliographic coupling map in VOSviewer using fractional counting.\n"
        + ("Filler prose sentence for body length. " * 100)
        + "\nReferences\nVan Eck, N. J., & Waltman, L. (2010).\n",
        encoding="utf-8",
    )
    res = parse_pdf_features(txt_file, title="A bibliometric analysis of green finance")
    assert res["source_format"] == "txt"
    assert res["is_guide"] is False
    assert res["uses_scopus"] is True
    assert res["exact_query_printed"] is True
    assert res["fractional_disclosed"] is True
    assert res["sw_vosviewer"] is True
    assert GUIDE_TITLE_RE.search("How to design bibliometric research: an overview") is not None


def test_author_excel_export_and_roundtrip_rejoin_zero_merged_cells(tmp_path: Path) -> None:
    """: 3 flat zero-merge sheets, blind packets, Excel->03 re-join, & parenthesized Scopus query."""
    mod05 = import_module("05_export_author_excel")
    mod03 = import_module("03_build_repro_corpora")
    mod02b = import_module("02b_prepare_extraction_packets")

    out_xlsx = tmp_path / "author_verification_pack.xlsx"
    cand_json = tmp_path / "layer2_llm_candidates.json"
    l2 = [
        {
            "row_id": 1,
            "quartile": "Q1",
            "doi": "10.1016/j.jbr.2024.001",
            "scopus_id": "85100001",
            "journal_key": "01482963",
            "citescore": 20.5,
            "ipw_weight": 1.85,
            "pub_year": 2024,
            "journal": "Journal of Business Research",
            "title": "Sample Review",
            "source_format": "txt",
            "database_source": "wos",
            "llm_network_type": "bibliographic_coupling",
            "figure_locator": "Fig. 4, p. 8",
            "regex_mentions_bc_cc": "BC=1,CC=0",
            "regex_vs_llm_flag": "MATCH",
            "verbatim_query": '(TITLE-ABS-KEY("supply chain*") AND TITLE-ABS-KEY("resilience"))',
            "query_page": 4,
            "start_year": 2010,
            "cutoff_year": 2023,
            "subjarea_filter": "BUSI",
            "n_orig_printed": 412,
            "n_repro_count": 425,
            "ratio_repro_orig": 1.032,
            "node_threshold": 150,
            "verbatim_network_quote": "Figure 4 presents the bibliographic coupling of documents in VOSviewer.",
            "quote_byte_verified": True,
            "author_verdict_OK_or_FIX": "OK",
            "author_correction_note": "",
        }
    ]
    cand_json.write_text(json.dumps(l2), encoding="utf-8")
    l1_qa = [{"row_id": 1, "quartile": "Q1", "doi": "10.1016/j.jbr.2024.001", "llm_scopus": True, "regex_scopus": True}]
    l1_dis = [{"row_id": 1, "quartile": "Q2", "doi": "10.1007/s11301-024-001", "disagree_fields": "query(LLM=1,R=0)"}]

    mod05.export_author_workbook(l2, l1_qa, l1_dis, out_xlsx)
    wb = openpyxl.load_workbook(out_xlsx)
    assert wb.sheetnames == ["Layer2_Verify_60", "Layer1_QA_50", "Layer1_Disagreements"]
    for name in wb.sheetnames:
        ws = wb[name]
        assert len(ws.merged_cells.ranges) == 0
        assert ws.max_row == 2

    # Round-trip Excel -> 03_build_repro_corpora preserves journal_key, scopus_id, citescore, ipw_weight, database_source!
    rejoined = mod03.load_specs_from_excel(out_xlsx, candidates_json=cand_json)
    assert len(rejoined) == 1
    assert rejoined[0]["journal_key"] == "01482963"
    assert rejoined[0]["database_source"] == "wos"
    assert rejoined[0]["ipw_weight"] == 1.85
    assert rejoined[0]["citescore"] == 20.5

    # Leading '(' in Scopus query must NOT double-wrap TITLE-ABS-KEY
    sq = mod03.build_scopus_api_query(rejoined[0])
    assert "TITLE-ABS-KEY((TITLE-ABS-KEY" not in sq
    assert 'TITLE-ABS-KEY("supply chain*")' in sq

    # Blind packet must NOT leak REGEX_FLAGS and MUST preserve tail Data Availability page
    big_txt = tmp_path / "big_paper.txt"
    big_txt.write_text(
        ("Introduction Scopus VOSviewer Fig. 1. " * 2000) + "\nData availability: https://zenodo.org/record/999999\n",
        encoding="utf-8",
    )
    pkt = mod02b.build_compact_packet(big_txt, {"doi": "10.1/test", "uses_scopus": True}, max_chars=20_000, tail_reserve_chars=5_000)
    assert "REGEX_FLAGS" not in pkt
    assert "zenodo.org/record/999999" in pkt
