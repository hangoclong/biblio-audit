"""Adversarial parser tests covering all 6 database formats and pathological edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biblio_audit.exceptions import MissingFieldError
from biblio_audit.networks import build_network_artifact
from biblio_audit.parsers import (
    canonicalize_doi,
    detect_format,
    load_corpus_files,
    load_thesaurus,
    normalize_natural_author_name,
    normalize_reference_key,
    split_scopus_references,
)


def test_sici_doi_preserved_intact() -> None:
    raw = "Teece, D.J., Dynamic capabilities, Strat Mgmt J, 18(7), 509-533, 10.1002/(sici)1097-0266(199708)18:7<509::aid-smj882>3.0.co;2-z."
    doi = canonicalize_doi(raw)
    assert doi == "10.1002/(sici)1097-0266(199708)18:7<509::aid-smj882>3.0.co;2-z"
    assert normalize_reference_key(raw) == f"doi:{doi}"


def test_scopus_2023_semicolon_authors_html_entities_and_midtitle_year() -> None:
    refs_cell = (
        "Cheng M.; Xue Y.; Cui M., Global supply chains, J. Bus. Res., (2023); "
        "&#x0141;ukasiewicz J.; Kowalski A., Logic foundations, (2017a); "
        "Wang Y.; Li X., Supply chain resilience during COVID-19 (2020); an empirical study, IJPE, 245, (2022); "
        "Teece D.J.; Pisano G., Dynamic capabilities, SMJ, (1997), 10.1002/(sici)1097-0266(199708)18:7<509::aid-smj882>3.0.co;2-z"
    )
    parts = split_scopus_references(refs_cell)
    assert len(parts) == 4
    assert "Cheng M." in parts[0] and "Cui M." in parts[0]
    assert "Łukasiewicz J." in parts[1]
    assert "COVID-19 (2020); an empirical study" in parts[2]
    assert normalize_reference_key(parts[3]) == "doi:10.1002/(sici)1097-0266(199708)18:7<509::aid-smj882>3.0.co;2-z"


def test_scopus_csv_author_ids_trailing_semicolon_and_keywords(tmp_path: Path) -> None:
    csv_p = tmp_path / "scopus.csv"
    csv_p.write_text(
        "\ufeffAuthors,Author full names,Author(s) ID,Title,Year,Source title,Cited by,DOI,EID,Author Keywords,Indexed Keywords,References\n"
        '"Liu F.; Li C.","Liu, Fengyuan (60410297700); Li, Chunyun (58862216200)","60410297700; 58862216200;","AI in HRM",2024,"Adm Sci",12,10.1000/a,2-s2.0-1,"AI; HRM","Machine Learning; Staffing","Smith A.; Jones B., Ref one, (2020); Doe C., Ref two, (2021)"\n',
        encoding="utf-8",
    )
    assert detect_format(csv_p) == "scopus"
    wks, warns = load_corpus_files([csv_p])
    assert len(wks) == 1
    assert [a["id"] for a in wks[0]["authors"]] == ["scopus_auid:60410297700", "scopus_auid:58862216200"]
    assert wks[0]["author_keywords"] == ["ai", "hrm"]
    assert wks[0]["index_keywords"] == ["machine learning", "staffing"]
    assert len(wks[0]["referenced_works"]) == 2
    assert not any("Homonym" in w for w in warns)


def test_wos_tsv_and_plaintext_and_multifile_dedup(tmp_path: Path) -> None:
    tsv_p = tmp_path / "savedrecs_1.txt"
    tsv_p.write_text(
        "PT\tAU\tAF\tTI\tPY\tDI\tTC\tDE\tID\tCR\tUT\n"
        "J\tSmith, J; Doe, A\tSmith, Jane; Doe, Alan\tPaper One\t2023\t10.1000/w1\t10\tAI; Strategy\tManagement\tKessler, 1963, DOI DOI 10.1002/asi.5090140103; Small, 1973, DOI [10.1002/asi.4630240406, 10.9999/dup]\tWOS:001\n",
        encoding="utf-8",
    )
    pt_p = tmp_path / "savedrecs_2.txt"
    pt_p.write_text(
        "FN Clarivate Analytics Web of Science\n"
        "VR 1.0\n"
        "PT J\n"
        "AF Smith, Jane\n"
        "   Doe, Alan\n"
        "TI Paper One Duplicate\n"
        "PY 2023\n"
        "DI 10.1000/w1\n"
        "TC 10\n"
        "UT WOS:001\n"
        "ER\n"
        "PT J\n"
        "AF van Eck, Nees Jan\n"
        "   Waltman, Ludo\n"
        "TI Science Mapping\n"
        "   Revisited\n"
        "DE Bibliometrics; VOSviewer\n"
        "CR Kessler M, 1963, DOI 10.1002/asi.5090140103\n"
        "   Small H, 1973, DOI 10.1002/asi.4630240406\n"
        "PY 2024\n"
        "DI 10.1000/w2\n"
        "TC 25\n"
        "UT WOS:002\n"
        "ER\n"
        "EF\n",
        encoding="utf-8",
    )
    assert detect_format(tsv_p) == "wos"
    assert detect_format(pt_p) == "wos_plaintext"
    wks, warns = load_corpus_files([tsv_p, pt_p])
    assert len(wks) == 2
    assert any("Deduplicated 1" in w for w in warns)
    assert "doi:10.1002/asi.5090140103" in wks[0]["referenced_works"]
    assert "doi:10.1002/asi.4630240406" in wks[0]["referenced_works"]
    assert wks[1]["title"] == "Science Mapping Revisited"
    assert [a["id"] for a in wks[1]["authors"]] == ["van eck, nj", "waltman, l"]


def test_dimensions_preamble_and_title_pipe_shift(tmp_path: Path) -> None:
    dim_p = tmp_path / "dimensions.csv"
    dim_p.write_text(
        '"About the data: exported from Dimensions on 2026-11-23"\n'
        'Publication ID,Title,Pub Year,Times cited,DOI,Authors,Concepts,Cited references\n'
        'pub.1,"Dynamic Firms",2024,15,10.1000/d1,"Barney, J.; Teece, D.","strategy; capabilities","[Barney, J.|Firm resources: A | B perspective|JOM|1991|17|1|99-120|10.1177/014920639101700108];[Small, H.|Co-citation|JASIS|1973|24|4|265-269|10.1002/asi.4630240406]"\n',
        encoding="utf-8",
    )
    assert detect_format(dim_p) == "dimensions"
    wks, _ = load_corpus_files([dim_p])
    assert len(wks) == 1
    assert "doi:10.1177/014920639101700108" in wks[0]["referenced_works"]
    assert "doi:10.1002/asi.4630240406" in wks[0]["referenced_works"]


def test_lens_natural_author_particles_suffixes_and_lens_id_refs(tmp_path: Path) -> None:
    assert normalize_natural_author_name("Nees Jan van Eck") == "van eck, nj"
    assert normalize_natural_author_name("Daron Acemoglu Jr.") == "acemoglu, d"
    assert normalize_natural_author_name("Maria Elena de la Cruz III") == "de la cruz, me"

    lens_p = tmp_path / "lens.csv"
    lens_p.write_text(
        'Lens ID,Title,Publication Year,DOI,Citing Works Count,Author/s,Keywords,References\n'
        '000-360-111-222-333,"Mapping Science",2025,10.1000/l1,8,"Nees Jan van Eck; Daron Acemoglu Jr.","bibliometrics; science mapping","000-928-794-577-264; 008-908-111-222-333"\n',
        encoding="utf-8",
    )
    assert detect_format(lens_p) == "lens"
    wks, warns = load_corpus_files([lens_p])
    assert [a["id"] for a in wks[0]["authors"]] == ["van eck, nj", "acemoglu, d"]
    assert wks[0]["referenced_works"] == ["lens:000-928-794-577-264", "lens:008-908-111-222-333"]
    assert any("HomonymCollapseWarning" in w for w in warns)


def test_openalex_csv_empty_pipe_alignment_and_missing_refs_error(tmp_path: Path) -> None:
    oa_csv = tmp_path / "oa.csv"
    oa_csv.write_text(
        "id,display_name,publication_year,doi,cited_by_count,authorships.author.id,authorships.author.display_name,keywords.display_name\n"
        "https://openalex.org/W1,Paper A,2024,10.1000/o1,5,https://openalex.org/A1||https://openalex.org/A3,Alice One|Bob Two|Carol Three,ai|strategy\n",
        encoding="utf-8",
    )
    assert detect_format(oa_csv) == "openalex_csv"
    wks, _ = load_corpus_files([oa_csv])
    assert [a["id"] for a in wks[0]["authors"]] == ["openalex:A1", "two, b", "openalex:A3"]
    assert [a["name"] for a in wks[0]["authors"]] == ["Alice One", "Bob Two", "Carol Three"]
    with pytest.raises(MissingFieldError):
        build_network_artifact(wks, network_type="bc")


def test_openalex_json_100_author_truncation_warning(tmp_path: Path) -> None:
    oa_json = tmp_path / "oa.json"
    auths = [{"author": {"id": f"https://openalex.org/A{i}", "display_name": f"Author {i}"}} for i in range(100)]
    oa_json.write_text(
        json.dumps([{"id": "https://openalex.org/W99", "title": "Consortium", "publication_year": 2024, "authorships": auths, "referenced_works": []}]),
        encoding="utf-8",
    )
    _, warns = load_corpus_files([oa_json])
    assert any("OpenAlexTruncationWarning" in w for w in warns)


def test_vosviewer_thesaurus_merge_and_ignore(tmp_path: Path) -> None:
    th_p = tmp_path / "thesaurus.txt"
    th_p.write_text("label\treplace by\nai\tartificial intelligence\nsmes\tsme\nliterature review\t\n", encoding="utf-8")
    th = load_thesaurus(th_p)
    assert th["ai"] == "artificial intelligence"
    assert th["smes"] == "sme"
    assert th["literature review"] is None


def test_particles_unicode_multiline_bom_and_nodoi_dedup_edge_cases(tmp_path: Path) -> None:
    """Verify nobiliary particles, curly apostrophes, hyphenated initials, Unicode names, multiline CSV, BOM JSON, and no-DOI dedup."""
    from biblio_audit.exceptions import FormatDetectionError
    from biblio_audit.parsers import normalize_natural_author_name

    # Scopus 2023+ references with particles, curly apostrophes, hyphenated initials
    refs_cell = (
        "van Eck N.J.; Waltman L., Software survey: VOSviewer, Scientometrics, (2010); "
        "De Massis A.; Frattini F.; Kotlar J., Innovation in family firms, JPIM, (2015); "
        "O’Reilly C.A.; Tushman M.L., Organizational ambidexterity, AMP, (2013); "
        "Richard M.-O.; Chebat J.-C., Modeling online consumer behavior, JBR, (2016)"
    )
    parts = split_scopus_references(refs_cell)
    assert len(parts) == 4
    assert "van Eck N.J." in parts[0] and "Waltman L." in parts[0]
    assert "De Massis A." in parts[1] and "Kotlar J." in parts[1]
    assert "O’Reilly C.A." in parts[2] and "Tushman M.L." in parts[2]
    assert "Richard M.-O." in parts[3] and "Chebat J.-C." in parts[3]

    # Full Unicode surnames & given names (Vietnamese & Polish)
    assert normalize_natural_author_name("Nguyễn, Đức Minh") == "nguyễn, đm"
    assert normalize_natural_author_name("Kowalski, Łukasz") == "kowalski, ł"

    # Multiline quoted CSV cells remain intact
    ml_csv = tmp_path / "multiline.csv"
    ml_csv.write_text(
        'Authors,Title,Year,Cited by,DOI,EID,Author Keywords,References\n'
        '"Smith A.; Jones B.","Line one title\nLine two subtitle",2024,5,10.1000/ml1,2-s2.0-ml1,"ai; governance","Ref A (2020); Ref B (2021)"\n',
        encoding="utf-8",
    )
    wks_ml, _ = load_corpus_files([ml_csv])
    assert len(wks_ml) == 1
    assert "Line one title" in wks_ml[0]["title"] and "Line two subtitle" in wks_ml[0]["title"]

    # WoS Plain-Text wrapped DE continuation lines join with space, not semicolon
    pt_wrap = tmp_path / "wos_wrap.txt"
    pt_wrap.write_text(
        "FN Clarivate Analytics Web of Science\n"
        "VR 1.0\n"
        "PT J\n"
        "AF Smith, Jane\n"
        "TI Wrapped Keywords\n"
        "DE artificial\n"
        "   intelligence; supply chain\n"
        "   resilience\n"
        "PY 2024\n"
        "UT WOS:999\n"
        "ER\n"
        "EF\n",
        encoding="utf-8",
    )
    wks_pt, _ = load_corpus_files([pt_wrap])
    assert wks_pt[0]["author_keywords"] == ["artificial intelligence", "supply chain resilience"]

    # OpenAlex JSON with UTF-8 BOM parses cleanly; malformed JSON raises FormatDetectionError
    bom_json = tmp_path / "oa_bom.json"
    bom_json.write_bytes(b"\xef\xbb\xbf" + json.dumps([{"id": "W1", "title": "BOM OK", "publication_year": 2024}]).encode("utf-8"))
    wks_bom, _ = load_corpus_files([bom_json])
    assert len(wks_bom) == 1 and wks_bom[0]["title"] == "BOM OK"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(FormatDetectionError):
        load_corpus_files([bad_json])

    # Multi-file records without DOI/EID do NOT collide when titles differ
    nodoi_1 = tmp_path / "nodoi_1.csv"
    nodoi_2 = tmp_path / "nodoi_2.csv"
    nodoi_1.write_text('Authors,Title,Year,Cited by,DOI,EID\n"Smith A.","Book Alpha",2021,10,,\n', encoding="utf-8")
    nodoi_2.write_text('Authors,Title,Year,Cited by,DOI,EID\n"Jones B.","Book Beta",2022,12,,\n', encoding="utf-8")
    wks_nd, _ = load_corpus_files([nodoi_1, nodoi_2])
    assert len(wks_nd) == 2
