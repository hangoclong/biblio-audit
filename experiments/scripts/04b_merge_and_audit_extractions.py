"""Deterministic Merger, Per-Field Verbatim Byte-Verifier & Stage-1 Regex vs Stage-2 Blind LLM Audit.

Verification specification:
- Byte-verifies EVERY positive field's verbatim quote (`evidence_quotes` dict) against the source file.
- Applies `llm.is_guide_or_editorial` filter to BOTH Layer 1 and Layer 2.
- Exports 3 clean flat tables for `05_export_author_excel.py`:
  1. `layer2_llm_candidates.json` (`Layer2_Verify_60`, locked Q1 + 1:1:1 Q2/Q3/Q4 interleave, cap <= 3/journal),
  2. `layer1_qa_50_merged.json` (`Layer1_QA_50`, unbiased random 50-article sample for precision/recall/F1/kappa),
  3. `layer1_disagreements_merged.json` (`Layer1_Disagreements`, ALL remaining Regex-vs-LLM conflicts across ~200 articles).
- Mode `--finalize-l1-excel <xlsx>`: ingests author-signed `author_verification_pack.xlsx`, applies `FIX`
  adjudications, and outputs final author-adjudicated Layer-1 HT (capped + cap-adjusted + Two-Stage IPW) 95% CIs
  using the exact canonical bootstrap seed schedule shared with `04d_compute_layer1_adjudicated_census.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

EXP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXP_DIR / "src_audit"
SCRIPTS_DIR = EXP_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_audit_features import extract_body_pages, normalize_pdf_text, unwrap_soft_newlines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.merge_audit")

PROC_DIR = EXP_DIR / "data" / "processed"
SHARD_DIR = PROC_DIR / "llm_shards"
RES_DIR = EXP_DIR / "results"
MASTER_SEED = 20261122

COMPARE_FIELDS = [
    ("uses_scopus", "uses_scopus", "scopus"),
    ("uses_wos", "uses_wos", "wos"),
    ("exact_query_printed", "exact_query_printed", "query"),
    ("network_fractional_disclosed", "fractional_disclosed", "net_frac"),
    ("null_model_reported", "null_model_reported", "null"),
    ("stability_check_reported", "stability_check_reported", "stability"),
    ("data_or_code_shared", "data_or_code_shared", "data_code"),
]


def _norm_ws(text: str) -> str:
    t = normalize_pdf_text(str(text or ""))
    t = unwrap_soft_newlines(t)
    return re.sub(r"\s+", " ", t).strip().lower()


def verify_quote_in_fulltext(norm_fulltext: str, quote: str) -> bool:
    """Check whether a verbatim quote (or ellipsis-split spans >= 12 chars) occurs verbatim in source."""
    q = _norm_ws(quote)
    if len(q) < 12:
        return False
    if q in norm_fulltext:
        return True
    parts = [p.strip() for p in re.split(r"\.{3,}|\[…\]|…", q) if len(p.strip()) >= 12]
    return bool(parts) and all(p in norm_fulltext for p in parts)


def verify_per_field_quotes(doc_path: Path, llm: dict[str, Any]) -> tuple[bool, dict[str, bool], str]:
    """Verify every positive field's verbatim quote in source text."""
    pages = extract_body_pages(doc_path)
    full = _norm_ws(" ".join(txt for _, txt in pages))

    q_dict = llm.get("evidence_quotes") if isinstance(llm.get("evidence_quotes"), dict) else {}
    per_field: dict[str, bool] = {}
    quote_lines: list[str] = []

    for l_key, _, lbl in COMPARE_FIELDS:
        if bool(llm.get(l_key)):
            q_str = str(q_dict.get(lbl) or q_dict.get(l_key) or "").strip()
            if not q_str and lbl == "query":
                q_str = str(llm.get("verbatim_query") or llm.get("verbatim_query_quote") or "").strip()
            if not q_str:
                q_str = str(llm.get("verbatim_evidence_quote") or "").strip()
            ok = verify_quote_in_fulltext(full, q_str) if q_str else False
            per_field[lbl] = ok
            quote_lines.append(f"[{lbl}:{'OK' if ok else 'MISS'}] {q_str[:180]}")

    net_q = str(llm.get("verbatim_network_quote") or "").strip()
    if net_q:
        per_field["network"] = verify_quote_in_fulltext(full, net_q)

    all_ok = all(per_field.values()) if per_field else True
    return all_ok, per_field, " | ".join(quote_lines)


def _format_l1_excel_row(row_id: int, r: dict[str, Any]) -> dict[str, Any]:
    llm = r["llm"]
    return {
        "row_id": row_id,
        "quartile": r["quartile"],
        "doi": r["doi"],
        "pub_year": r["pub_year"],
        "journal": r["publication_name"],
        "title": r["title"],
        "llm_scopus": bool(llm.get("uses_scopus")),
        "regex_scopus": bool(r.get("uses_scopus")),
        "llm_wos": bool(llm.get("uses_wos")),
        "regex_wos": bool(r.get("uses_wos")),
        "llm_query_printed": bool(llm.get("exact_query_printed")),
        "regex_query_printed": bool(r.get("exact_query_printed")),
        "llm_net_fractional": bool(llm.get("network_fractional_disclosed")),
        "regex_net_fractional": bool(r.get("fractional_disclosed")),
        "llm_null_model": bool(llm.get("null_model_reported")),
        "regex_null_model": bool(r.get("null_model_reported")),
        "llm_stability": bool(llm.get("stability_check_reported")),
        "regex_stability": bool(r.get("stability_check_reported")),
        "llm_data_code": bool(llm.get("data_or_code_shared")),
        "regex_data_code": bool(r.get("data_or_code_shared")),
        "disagree_fields": r["disagree_fields"],
        "verbatim_evidence_quotes": r["formatted_evidence_quotes"],
        "quote_byte_verified": bool(r["quote_byte_verified"]),
        "author_verdict_OK_or_FIX": "",
        "author_correction_note": "",
    }


def finalize_layer1_from_excel(xlsx_path: Path) -> None:
    """Ingest author-signed Excel (`Layer1_QA_50` + `Layer1_Disagreements`) & compute final adjudicated L1 HT/IPW CIs."""
    from importlib import import_module

    mod02 = import_module("02_download_and_parse_oa")
    merged_all = json.loads((RES_DIR / "layer1_llm_merged.json").read_text(encoding="utf-8"))
    bal = json.loads((RES_DIR / "layer0_population_balance.json").read_text(encoding="utf-8"))
    wh = {q: float(bal["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    excel_overrides: dict[str, dict[str, Any]] = {}
    qa_sheet_rows: list[dict[str, Any]] = []
    for sname in ("Layer1_QA_50", "Layer1_Disagreements"):
        if sname not in wb.sheetnames:
            continue
        ws = wb[sname]
        headers = [str(ws.cell(1, c).value or "").strip() for c in range(1, ws.max_column + 1)]
        for r_idx in range(2, ws.max_row + 1):
            row = {headers[c - 1]: ws.cell(r_idx, c).value for c in range(1, len(headers) + 1)}
            doi_k = str(row.get("doi") or "").lower().strip()
            if not doi_k:
                continue
            if sname == "Layer1_QA_50":
                qa_sheet_rows.append(row)
            excel_overrides[doi_k] = row

    # Build final adjudicated binary labels (if author edited any llm_* column or wrote FIX, Excel cell wins)
    col_map = {
        "uses_scopus": "llm_scopus",
        "uses_wos": "llm_wos",
        "exact_query_printed": "llm_query_printed",
        "network_fractional_disclosed": "llm_net_fractional",
        "null_model_reported": "llm_null_model",
        "stability_check_reported": "llm_stability",
        "data_or_code_shared": "llm_data_code",
    }
    adjudicated: list[dict[str, Any]] = []
    for r in merged_all:
        doi_k = str(r.get("doi") or "").lower().strip()
        ov = excel_overrides.get(doi_k)
        final_rec = {
            "doi": r["doi"],
            "quartile": r["quartile"],
            "journal_key": r["journal_key"],
            "cap_ht_weight": float(r.get("cap_ht_weight") or 1.0),
            "two_stage_ipw_weight": float(r.get("two_stage_ipw_weight") or r.get("ipw_weight") or 1.0),
        }
        for target_f, xl_col in col_map.items():
            if ov is not None and ov.get(xl_col) is not None and str(ov.get(xl_col)).strip() != "":
                val = str(ov[xl_col]).strip().lower() in ("true", "1", "yes")
            else:
                val = bool(r["llm"].get(target_f))
            final_rec[target_f] = val
        adjudicated.append(final_rec)

    ht_capped = {}
    ht_cap_adj = {}
    ht_ipw2 = {}
    for idx, field in enumerate(col_map.keys()):
        ht_capped[field] = mod02.journal_clustered_ht_bootstrap_prop(adjudicated, field, wh, weight_field=None, seed=MASTER_SEED + 1100 + idx)
        ht_cap_adj[field] = mod02.journal_clustered_ht_bootstrap_prop(adjudicated, field, wh, weight_field="cap_ht_weight", seed=MASTER_SEED + 1200 + idx)
        ht_ipw2[field] = mod02.journal_clustered_ht_bootstrap_prop(adjudicated, field, wh, weight_field="two_stage_ipw_weight", seed=MASTER_SEED + 1300 + idx)

    out = {
        "schema_version": "1.2",
        "n_adjudicated_empirical": len(adjudicated),
        "n_qa50_verified": len(qa_sheet_rows),
        "n_excel_rows_applied": len(excel_overrides),
        "ht_capped_clustered": ht_capped,
        "ht_article_weighted_cap_adjusted": ht_cap_adj,
        "ht_two_stage_ipw_sensitivity": ht_ipw2,
    }
    out_path = RES_DIR / "layer1_final_adjudicated_census.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Saved final author-adjudicated Layer-1 census to %s", out_path)


def main() -> None:
    """Merge blind LLM shards, verify per-field quotes, compute Regex parity, emit 3-tab Excel JSONs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", type=str, default="shard_*.json")
    ap.add_argument("--cont-records", type=str, default="", help="Optional Layer-2 continuation records JSON (does NOT alter frozen Layer 1)")
    ap.add_argument("--cont-shard-dir", type=str, default="", help="Optional directory of continuation LLM shards for Layer-2 candidate expansion")
    ap.add_argument("--l2-max-candidates", type=int, default=150)
    ap.add_argument("--finalize-l1-excel", type=str, default="", help="Path to author-signed Excel to compute final L1 HT/IPW CIs")
    args = ap.parse_args()

    if args.finalize_l1_excel:
        finalize_layer1_from_excel(Path(args.finalize_l1_excel))
        return

    parsed_list = json.loads((RES_DIR / "layer1_parsed_records.json").read_text(encoding="utf-8"))
    llm_items: list[dict[str, Any]] = []
    for p in sorted(SHARD_DIR.glob(args.shard_glob)):
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        llm_items.extend(data)
        LOGGER.info("Loaded %d LLM extractions from %s", len(data), p.name)

    if not llm_items:
        raise FileNotFoundError(f"No LLM shard JSON files found in {SHARD_DIR} matching {args.shard_glob}")

    dedup: dict[str, dict[str, Any]] = {}
    for item in llm_items:
        key = str(item.get("doi") or item.get("scopus_id") or "").lower().strip()
        if key:
            dedup[key] = item

    merged_all: list[dict[str, Any]] = []
    n_llm_guides_excluded = 0
    y_llm: dict[str, list[int]] = {lbl: [] for _, _, lbl in COMPARE_FIELDS}
    y_reg: dict[str, list[int]] = {lbl: [] for _, _, lbl in COMPARE_FIELDS}

    for rec in parsed_list:
        if rec.get("is_guide") or int(rec.get("n_body_pages") or 0) == 0:
            continue
        doi_k = str(rec.get("doi") or "").lower().strip()
        sid_k = str(rec.get("scopus_id") or "").strip()
        llm = dedup.get(doi_k) or dedup.get(sid_k)
        if not llm:
            continue
        # Exclude tutorials/editorials identified by LLM from BOTH Layer 1 and Layer 2
        if bool(llm.get("is_guide_or_editorial")):
            n_llm_guides_excluded += 1
            continue

        doc_path = EXP_DIR / rec["doc_rel_path"]
        all_q_ok, per_field_ok, formatted_quotes = verify_per_field_quotes(doc_path, llm)

        disagree: list[str] = []
        for l_key, r_key, lbl in COMPARE_FIELDS:
            lv = bool(llm.get(l_key))
            rv = bool(rec.get(r_key))
            y_llm[lbl].append(int(lv))
            y_reg[lbl].append(int(rv))
            if lv != rv:
                disagree.append(f"{lbl}(LLM={int(lv)},R={int(rv)})")

        merged_all.append(
            {
                **rec,
                "llm": llm,
                "quote_byte_verified": bool(all_q_ok),
                "per_field_quote_verified": per_field_ok,
                "network_quote_byte_verified": bool(per_field_ok.get("network", False)),
                "formatted_evidence_quotes": formatted_quotes,
                "disagree_fields": "; ".join(disagree),
            }
        )

    field_metrics: dict[str, Any] = {}
    for _, _, lbl in COMPARE_FIELDS:
        yl = np.array(y_llm[lbl], dtype=int)
        yr = np.array(y_reg[lbl], dtype=int)
        if len(yl) == 0:
            continue
        agree = float(np.mean(yl == yr))
        kappa = float(cohen_kappa_score(yl, yr)) if len(set(yl)) > 1 or len(set(yr)) > 1 else 1.0
        if np.isnan(kappa):
            kappa = 1.0
        field_metrics[lbl] = {
            "n": int(len(yl)),
            "llm_pos": int(np.sum(yl)),
            "regex_pos": int(np.sum(yr)),
            "raw_agreement": round(agree, 4),
            "cohen_kappa": round(kappa, 4),
            "regex_precision_vs_llm": round(float(precision_score(yl, yr, zero_division=0)), 4),
            "regex_recall_vs_llm": round(float(recall_score(yl, yr, zero_division=0)), 4),
            "regex_f1_vs_llm": round(float(f1_score(yl, yr, zero_division=0)), 4),
        }

    audit_report = {
        "lock_version": "1.2",
        "n_merged_empirical": len(merged_all),
        "n_llm_guides_excluded": n_llm_guides_excluded,
        "all_positive_quotes_verified_rate": round(sum(1 for r in merged_all if r["quote_byte_verified"]) / max(1, len(merged_all)), 4),
        "any_regex_llm_disagreement_count": sum(1 for r in merged_all if r["disagree_fields"]),
        "field_agreement_regex_vs_llm": field_metrics,
    }
    (RES_DIR / "layer1_llm_merged.json").write_text(json.dumps(merged_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (RES_DIR / "layer1_regex_vs_llm_audit.json").write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    # Build Layer-2 candidate pool: Layer 1 (frozen) + optional locked-order continuation pool
    l2_pool = list(merged_all)
    if args.cont_records and args.cont_shard_dir:
        cont_recs = json.loads(Path(args.cont_records).read_text(encoding="utf-8"))
        cont_llm: dict[str, dict[str, Any]] = {}
        for p in sorted(Path(args.cont_shard_dir).glob("shard_*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "records" in data:
                data = data["records"]
            for item in data:
                k = str(item.get("doi") or item.get("scopus_id") or "").lower().strip()
                if k:
                    cont_llm[k] = item
        seen_l1_dois = {str(r.get("doi") or "").lower() for r in merged_all}
        n_cont_added = 0
        for rec in cont_recs:
            doi_k = str(rec.get("doi") or "").lower().strip()
            if not doi_k or doi_k in seen_l1_dois:
                continue
            llm = cont_llm.get(doi_k) or cont_llm.get(str(rec.get("scopus_id") or "").strip())
            if not llm or bool(llm.get("is_guide_or_editorial")):
                continue
            doc_path = EXP_DIR / rec["doc_rel_path"]
            all_q_ok, per_field_ok, _ = verify_per_field_quotes(doc_path, llm)
            l2_pool.append(
                {
                    **rec,
                    "llm": llm,
                    "from_continuation": True,
                    "quote_byte_verified": bool(all_q_ok),
                    "network_quote_byte_verified": bool(per_field_ok.get("network", False)),
                }
            )
            n_cont_added += 1
        LOGGER.info("Appended %d continuation empirical records to Layer-2 candidate pool (Layer 1 stays frozen at %d)", n_cont_added, len(merged_all))

    # Layer-2 candidates: Q1 first + deterministic 1:1:1 Q2/Q3/Q4 interleave, cap <= 3/journal
    q1_m = sorted([r for r in l2_pool if r["quartile"] == "Q1"], key=lambda r: int(r.get("perm_rank", 0)))
    by_q24 = {
        q: sorted([r for r in l2_pool if r["quartile"] == q], key=lambda r: int(r.get("perm_rank", 0)))
        for q in ("Q2", "Q3", "Q4")
    }
    q24_m: list[dict[str, Any]] = []
    for idx in range(max((len(v) for v in by_q24.values()), default=0)):
        for q in ("Q2", "Q3", "Q4"):
            if idx < len(by_q24[q]):
                q24_m.append(by_q24[q][idx])

    l2_candidates: list[dict[str, Any]] = []
    j_cnt: Counter[str] = Counter()
    row_id = 0
    for r in q1_m + q24_m:
        llm = r["llm"]
        net_type = str(llm.get("primary_citation_network") or "none").strip().lower()
        if net_type not in ("bibliographic_coupling", "co_citation"):
            continue
        v_query = str(llm.get("verbatim_query") or "").strip()
        if not v_query or len(v_query) < 6:
            continue
        if j_cnt[r["journal_key"]] >= 3:
            continue
        j_cnt[r["journal_key"]] += 1
        row_id += 1
        r_bc = int(bool(r.get("mentions_bc")))
        r_cc = int(bool(r.get("mentions_cc")))
        flag = "MATCH"
        if net_type == "bibliographic_coupling" and not r_bc:
            flag = "CHECK_REGEX_NO_BC"
        elif net_type == "co_citation" and not r_cc:
            flag = "CHECK_REGEX_NO_CC"
        elif not r.get("exact_query_printed"):
            flag = "CHECK_REGEX_NO_QUERY"
        elif not r["network_quote_byte_verified"]:
            flag = "CHECK_QUOTE_VERIFY"

        l2_candidates.append(
            {
                "row_id": row_id,
                "quartile": r["quartile"],
                "doi": r["doi"],
                "scopus_id": r["scopus_id"],
                "journal_key": r["journal_key"],
                "citescore": r["citescore"],
                "ipw_weight": r.get("two_stage_ipw_weight", r.get("ipw_weight", 1.0)),
                "pub_year": r["pub_year"],
                "journal": r["publication_name"],
                "title": r["title"],
                "source_format": r["source_format"],
                "database_source": str(llm.get("primary_database") or "scopus"),
                "llm_network_type": net_type,
                "figure_locator": str(llm.get("figure_locator") or ""),
                "regex_mentions_bc_cc": f"BC={r_bc},CC={r_cc}",
                "regex_vs_llm_flag": flag,
                "verbatim_query": v_query,
                "query_page": llm.get("query_page"),
                "start_year": llm.get("start_year"),
                "cutoff_year": llm.get("cutoff_year") or (int(r["pub_year"]) - 1 if r.get("pub_year") else 2024),
                "subjarea_filter": str(llm.get("subjarea_filter") or ""),
                "n_orig_printed": llm.get("n_orig_printed"),
                "n_repro_count": None,
                "ratio_repro_orig": None,
                "node_threshold": llm.get("node_threshold") or 150,
                "verbatim_network_quote": str(llm.get("verbatim_network_quote") or ""),
                "quote_byte_verified": bool(r["network_quote_byte_verified"]),
                "author_verdict_OK_or_FIX": "",
                "author_correction_note": "",
            }
        )
        if len(l2_candidates) >= args.l2_max_candidates:
            break

    (RES_DIR / "layer2_llm_candidates.json").write_text(json.dumps(l2_candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sheet 2: Unbiased stratified random Layer1_QA_50
    l1_qa_rows: list[dict[str, Any]] = []
    qa_dois: set[str] = set()
    qa_id = 0
    for q_idx, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        q_rng = np.random.default_rng(MASTER_SEED + 10 + q_idx)
        sub = [r for r in merged_all if r["quartile"] == q]
        take = min(len(sub), 13 if q in ("Q1", "Q2") else 12)
        idxs = q_rng.choice(len(sub), size=take, replace=False) if take > 0 else []
        for idx in idxs:
            r = sub[int(idx)]
            qa_id += 1
            qa_dois.add(str(r["doi"]).lower())
            l1_qa_rows.append(_format_l1_excel_row(qa_id, r))
    (RES_DIR / "layer1_qa_50_merged.json").write_text(json.dumps(l1_qa_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sheet 3: ALL remaining Regex-vs-LLM conflicts outside QA-50
    l1_disagree_rows: list[dict[str, Any]] = []
    dis_id = 0
    for r in merged_all:
        if r["disagree_fields"] and str(r["doi"]).lower() not in qa_dois:
            dis_id += 1
            l1_disagree_rows.append(_format_l1_excel_row(dis_id, r))
    (RES_DIR / "layer1_disagreements_merged.json").write_text(json.dumps(l1_disagree_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info(
        "Merge & Audit complete: merged=%d (guides_excl=%d), L2=%d, L1_QA=%d, L1_Disagree=%d",
        len(merged_all), n_llm_guides_excluded, len(l2_candidates), len(l1_qa_rows), len(l1_disagree_rows),
    )


if __name__ == "__main__":
    main()
