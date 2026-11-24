"""Export ONE flat, unmerged-cell Excel verification pack for author review.

Follows verification specification + - Single .xlsx file (`experiments/results/author_verification_pack.xlsx`).
- Simple flat columns, STRICTLY ZERO merged cells (`len(ws.merged_cells.ranges) == 0`).
- Sheet 1 (`Layer2_Verify_60`): Candidate/selected Layer-2 studies with `database_source`, verbatim
  query, Figure/Table locator, Regex-vs-LLM flag, live Scopus/OpenAlex N_repro / N_orig, and
  author verdict columns (`OK` / `FIX`).
- Sheet 2 (`Layer1_QA_50`): Unbiased stratified random 50-article Layer-1 QA sample measuring
  human-verified precision/recall/F1/kappa with per-field verbatim quotes.
- Sheet 3 (`Layer1_Disagreements`): All remaining Layer-1 papers outside QA-50 where Round-1
  Regex and Round-2 Blind LLM disagreed, so zero automated conflicts go unadjudicated.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.excel")

EXP_DIR = Path(__file__).resolve().parents[1]
RES_DIR = EXP_DIR / "results"

L2_COLUMNS = [
    "row_id",
    "quartile",
    "doi",
    "pub_year",
    "journal",
    "title",
    "source_format",
    "database_source",
    "llm_network_type",
    "figure_locator",
    "regex_mentions_bc_cc",
    "regex_vs_llm_flag",
    "verbatim_query",
    "query_page",
    "start_year",
    "cutoff_year",
    "subjarea_filter",
    "n_orig_printed",
    "n_repro_count",
    "ratio_repro_orig",
    "node_threshold",
    "verbatim_network_quote",
    "quote_byte_verified",
    "author_verdict_OK_or_FIX",
    "author_correction_note",
]

L1_COLUMNS = [
    "row_id",
    "quartile",
    "doi",
    "pub_year",
    "journal",
    "title",
    "llm_scopus",
    "regex_scopus",
    "llm_wos",
    "regex_wos",
    "llm_query_printed",
    "regex_query_printed",
    "llm_net_fractional",
    "regex_net_fractional",
    "llm_null_model",
    "regex_null_model",
    "llm_stability",
    "regex_stability",
    "llm_data_code",
    "regex_data_code",
    "disagree_fields",
    "verbatim_evidence_quotes",
    "quote_byte_verified",
    "author_verdict_OK_or_FIX",
    "author_correction_note",
]


def _write_flat_sheet(wb: openpyxl.Workbook, title: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a flat tabular worksheet with zero merged cells and clean auto-filter."""
    ws = wb.create_sheet(title=title)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    warn_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    for r_idx, row_data in enumerate(rows, start=2):
        has_flag = bool(row_data.get("disagree_fields")) or (str(row_data.get("regex_vs_llm_flag", "")) not in ("", "MATCH"))
        for c_idx, col_name in enumerate(columns, start=1):
            val = row_data.get(col_name, "")
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if has_flag and col_name in ("disagree_fields", "regex_vs_llm_flag"):
                cell.fill = warn_fill

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{max(1, len(rows) + 1)}"

    wide_cols = {"title", "verbatim_query", "verbatim_network_quote", "verbatim_evidence_quotes", "author_correction_note"}
    for c_idx, col_name in enumerate(columns, start=1):
        width = 45 if col_name in wide_cols else (24 if col_name in ("doi", "journal", "disagree_fields") else 15)
        ws.column_dimensions[get_column_letter(c_idx)].width = width

    assert len(ws.merged_cells.ranges) == 0, f"Merged cells detected in {title}!"


def export_author_workbook(
    l2_rows: list[dict[str, Any]],
    l1_qa_rows: list[dict[str, Any]],
    l1_disagree_rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    """Create the single flat unmerged Excel workbook with 3 clean sheets for author verification."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_flat_sheet(wb, "Layer2_Verify_60", L2_COLUMNS, l2_rows)
    _write_flat_sheet(wb, "Layer1_QA_50", L1_COLUMNS, l1_qa_rows)
    _write_flat_sheet(wb, "Layer1_Disagreements", L1_COLUMNS, l1_disagree_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


def main() -> None:
    """Build author_verification_pack.xlsx from structured extraction JSONs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-json", type=str, default=str(RES_DIR / "layer2_llm_candidates.json"))
    ap.add_argument("--l1-qa-json", type=str, default=str(RES_DIR / "layer1_qa_50_merged.json"))
    ap.add_argument("--l1-disagree-json", type=str, default=str(RES_DIR / "layer1_disagreements_merged.json"))
    ap.add_argument("--out", type=str, default=str(RES_DIR / "author_verification_pack.xlsx"))
    args = ap.parse_args()

    l2_rows = json.loads(Path(args.l2_json).read_text(encoding="utf-8")) if Path(args.l2_json).exists() else []
    l1_qa = json.loads(Path(args.l1_qa_json).read_text(encoding="utf-8")) if Path(args.l1_qa_json).exists() else []
    l1_dis = json.loads(Path(args.l1_disagree_json).read_text(encoding="utf-8")) if Path(args.l1_disagree_json).exists() else []

    out = export_author_workbook(l2_rows, l1_qa, l1_dis, Path(args.out))
    LOGGER.info("Exported flat unmerged Excel pack: %s (L2=%d, L1_QA=%d, L1_Disagree=%d)", out, len(l2_rows), len(l1_qa), len(l1_dis))


if __name__ == "__main__":
    main()
