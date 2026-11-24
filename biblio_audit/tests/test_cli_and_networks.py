"""End-to-end CLI, multi-network (`--network all`), normalization, and reporting edge-case tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from biblio_audit.cli import run_cli
from biblio_audit.engine import audit_artifact
from biblio_audit.networks import build_network_artifact
from biblio_audit.parsers import load_corpus


def _write_rich_scopus_fixture(path: Path) -> None:
    """Create a 36-document multi-community Scopus CSV supporting BC, CC, CA, and KW cleanly."""
    rows = []
    for i in range(36):
        grp = i // 12  # 3 communities of 12 papers
        # Refs: 6 cluster-specific refs + 1 ubiquitous generic hub ref cited by everyone
        refs = [f"Author{grp} A., Core work {k}, J. Mgmt, (201{k % 9})" for k in range(6)]
        if i % 3 == 0:
            refs.append("HubAuthor H., Generic Methods Review, J. Methods, (2015)")
        # Authors: 3 overlapping teams + 1 bridging author
        au_ids = [f"{10000 + grp * 10 + (i % 6)}", f"{10000 + grp * 10 + ((i + 1) % 6)}"]
        if i % 4 == 0:
            au_ids.append("99999")
        au_names = "; ".join(f"Auth{aid} A." for aid in au_ids)
        # Keywords: cluster-specific + 1 bridge keyword
        kws = [f"topic_{grp}_a", f"topic_{grp}_b", f"topic_{grp}_c", f"topic_{grp}_{i % 4}"]
        if i % 3 == 0:
            kws.append("ai")
        rows.append({
            "Authors": au_names,
            "Author(s) ID": "; ".join(au_ids) + ";",
            "Title": f"Management study {i} on stream {grp} strategy innovation governance",
            "Year": 2021 + (i % 4),
            "Source title": "Test Management Journal",
            "Cited by": 5 + (i % 10),
            "DOI": f"10.1000/test.{i}",
            "EID": f"2-s2.0-{i}",
            "Author Keywords": "; ".join(kws),
            "Index Keywords": "",
            "References": "; ".join(refs),
        })
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_cli_network_all_thesaurus_latex_nodes_and_vos_export(tmp_path: Path) -> None:
    csv_p = tmp_path / "scopus_rich.csv"
    _write_rich_scopus_fixture(csv_p)
    th_p = tmp_path / "thesaurus.txt"
    th_p.write_text("label\treplace by\nai\tartificial intelligence\n", encoding="utf-8")

    md_out = tmp_path / "out.md"
    tex_out = tmp_path / "out.tex"
    json_out = tmp_path / "out.json"
    nodes_dir = tmp_path / "nodes"
    vos_dir = tmp_path / "vos"

    payload = run_cli([
        "--input", str(csv_p),
        "--network", "all",
        "--thesaurus", str(th_p),
        "--seeds", "4",
        "--starts", "2",
        "--min-cluster-size", "3",
        "--md-out", str(md_out),
        "--latex-out", str(tex_out),
        "--json-out", str(json_out),
        "--nodes-csv", str(nodes_dir),
        "--vos-export-dir", str(vos_dir),
    ])

    assert payload["n_works_loaded"] == 36
    assert set(payload["audits"].keys()) == {"bibliographic_coupling", "co_citation", "co_authorship", "keyword_cooccurrence"}
    assert md_out.exists() and "Multi-Network Counting-Rule Sensitivity Audit" in md_out.read_text(encoding="utf-8")
    assert tex_out.exists() and r"\toprule" in tex_out.read_text(encoding="utf-8")
    assert json_out.exists() and len(json.loads(json_out.read_text(encoding="utf-8"))["inputs"]) == 1
    assert (nodes_dir / "nodes_bibliographic_coupling.csv").exists()
    assert (vos_dir / "keyword_cooccurrence_vos_map.txt").exists()
    assert "artificial intelligence" in (vos_dir / "keyword_cooccurrence_vos_map.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("objective", ["VOS_AS_CPM", "VOS_Frac_CPM", "Leiden_RBConfig"])
def test_all_three_vosviewer_objectives_run_cleanly(tmp_path: Path, objective: str) -> None:
    csv_p = tmp_path / "scopus_rich.csv"
    _write_rich_scopus_fixture(csv_p)
    payload = run_cli([
        "--input", str(csv_p),
        "--network", "bc",
        "--objective", objective,
        "--seeds", "4",
        "--starts", "2",
        "--min-cluster-size", "3",
    ])
    a = payload["audit"]
    assert a["objective"] == objective
    assert -1.0 <= a["delta_count_mid"] <= 1.0
    assert -1.0 <= a["delta_aecs_mid"] <= 1.0
    assert 0.0 <= a["nvi_cross"] <= 1.0


def test_cli_bounds_validation_and_reporting_regressions(tmp_path: Path) -> None:
    """Verify CLI integer validation, full LaTeX escaping/warnings/nulls, VOSviewer sanitization, and SHA-256."""
    from biblio_audit.reporting import compute_files_sha256, export_vosviewer_files, render_latex_table

    # CLI rejects invalid integer parameters upfront
    csv_p = tmp_path / "scopus_rich.csv"
    _write_rich_scopus_fixture(csv_p)
    for bad_args in (
        ["--input", str(csv_p), "--seeds", "1"],
        ["--input", str(csv_p), "--starts", "0"],
        ["--input", str(csv_p), "--top-n", "-1"],
        ["--input", str(csv_p), "--min-occurrences", "0"],
        ["--input", str(csv_p), "--min-cluster-size", "0"],
        ["--input", str(csv_p), "--null-rewires", "-1"],
        ["--input", str(csv_p), "--null-starts", "0"],
    ):
        with pytest.raises(ValueError):
            run_cli(bad_args)

    # Full LaTeX escaping (all 10 special chars), Gini/TF-IDF/null/warnings parity, and VOSviewer CR/tab stripping
    wks = load_corpus(csv_p)
    wks[0]["title"] = "Broken\r\nTitle\tWith Tabs"
    art = build_network_artifact(wks, network_type="bc", core_top_n=20)
    res = audit_artifact(art, n_seeds=2, n_random_starts=1, min_cluster_size=3, null_rewires=5, null_starts=1)
    res.warnings.append("GUARDRAIL_SAMPLE_100%_CHECK")
    tex = render_latex_table("study#1_$100_{a}^~.csv", 20, [res])
    assert r"\#" in tex and r"\$" in tex and r"\{a\}" in tex and r"\textasciicircum{}" in tex and r"\textasciitilde{}" in tex
    assert "study#1_$100_{a}^~" not in tex
    assert "Event-degree Gini" in tex and "Bipartite null" in tex and "Guardrail warnings:" in tex and r"100\%" in tex

    vos_dir = tmp_path / "vos_clean"
    vos_files = export_vosviewer_files(vos_dir, res)
    map_p = Path(vos_files["vos_map"])
    raw_bytes = map_p.read_bytes()
    assert b"\r" not in raw_bytes.replace(b"\r\n", b"\n") or b"Broken Title" in raw_bytes
    assert "Broken Title With Tabs" in map_p.read_text(encoding="utf-8")

    # compute_files_sha256 disambiguates identical basenames in separate subdirs
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    f1, f2 = d1 / "export.csv", d2 / "export.csv"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")
    hashes = compute_files_sha256([f1, f2])
    assert len(hashes) == 2
    assert len(set(hashes.values())) == 2

    # --nodes-csv pointing to existing dir under --network all writes named file even if only 1 network succeeds
    sparse_csv = tmp_path / "sparse.csv"
    sparse_csv.write_text(
        "Authors,Title,Year,Cited by,DOI,EID,Author Keywords,References\n"
        + "\n".join(
            f'"A{i} B.","Study {i}",2023,5,10.1000/s.{i},2-s2.0-{i},"","Ref1 (2020); Ref2 (2020); Ref3 (2020)"'
            for i in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "existing_nodes_dir"
    out_dir.mkdir()
    payload = run_cli([
        "--input", str(sparse_csv),
        "--network", "all",
        "--seeds", "2",
        "--starts", "1",
        "--min-cluster-size", "2",
        "--nodes-csv", str(out_dir),
    ])
    assert "bibliographic_coupling" in payload["audits"]
    assert (out_dir / "nodes_bibliographic_coupling.csv").exists()
