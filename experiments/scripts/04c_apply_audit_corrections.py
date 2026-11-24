"""Apply verified independent extraction-audit corrections reproducibly.

Reads `layer1_independent_extraction_audit.json` (plus optional continuation audit JSON),
applies verified fixes cleanly onto `layer1_llm_merged.json`, `layer1_qa_50_merged.json`,
`layer1_disagreements_merged.json`, and `layer2_llm_candidates.json` without mutating raw
LLM shards, preserving a complete provenance trail.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.apply_audit")

EXP_DIR = Path(__file__).resolve().parents[1]
RES_DIR = EXP_DIR / "results"


def apply_corrections(audit_paths: list[Path]) -> None:
    """Apply all verified exclusions and field-level corrections from independent audit JSONs."""
    l2_corrs: list[dict[str, Any]] = []
    l1_corrs: list[dict[str, Any]] = []
    for p in audit_paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        l2_corrs.extend(data.get("l2_corrections") or [])
        l1_corrs.extend(data.get("l1_rare_field_corrections") or [])

    excluded_dois = {str(c["doi"]).lower() for c in l2_corrs if c.get("exclude")}
    LOGGER.info("Applying %d L2 corrections (%d exclusions) & %d L1 corrections", len(l2_corrs), len(excluded_dois), len(l1_corrs))

    # 1. Layer 2 candidates
    l2_path = RES_DIR / "layer2_llm_candidates.json"
    if l2_path.exists():
        l2 = json.loads(l2_path.read_text(encoding="utf-8"))
        l2_by_doi = {str(r["doi"]).lower(): r for r in l2}
        for c in l2_corrs:
            d = str(c["doi"]).lower()
            if d not in l2_by_doi or c.get("exclude"):
                continue
            rec = l2_by_doi[d]
            fld = str(c["field"])
            val = c["corrected_val"]
            if fld == "llm_network_type" and isinstance(val, str) and val not in ("bibliographic_coupling", "co_citation"):
                # Flag for human author adjudication instead of overwriting valid enum
                rec["regex_vs_llm_flag"] = f"AUDIT_FLAG:{val}"
                rec["author_correction_note"] = f"[Auditor]: {c.get('reason', '')[:220]}"
            else:
                rec[fld] = val
                if rec.get("regex_vs_llm_flag") == "MATCH":
                    rec["regex_vs_llm_flag"] = f"AUDITOR_FIXED:{fld}"
        kept_l2 = [r for r in l2 if str(r["doi"]).lower() not in excluded_dois]
        for idx, r in enumerate(kept_l2, start=1):
            r["row_id"] = idx
        l2_path.write_text(json.dumps(kept_l2, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Updated %s: %d -> %d rows", l2_path.name, len(l2), len(kept_l2))

    # 2. Layer 1 merged + QA-50 + Disagreements (also drop retracted articles from empirical Layer 1)
    l1_map = {(str(c["doi"]).lower(), str(c["field"])): c["corrected_val"] for c in l1_corrs}
    m_path = RES_DIR / "layer1_llm_merged.json"
    if m_path.exists():
        merged = json.loads(m_path.read_text(encoding="utf-8"))
        kept_m = []
        for r in merged:
            d = str(r["doi"]).lower()
            if d in excluded_dois:
                continue
            for (cd, fld), cval in l1_map.items():
                if cd == d:
                    r["llm"][fld] = cval
            kept_m.append(r)
        m_path.write_text(json.dumps(kept_m, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Updated %s: %d -> %d empirical rows", m_path.name, len(merged), len(kept_m))

    xl_col_for_field = {
        "uses_scopus": "llm_scopus",
        "uses_wos": "llm_wos",
        "exact_query_printed": "llm_query_printed",
        "network_fractional_disclosed": "llm_net_fractional",
        "null_model_reported": "llm_null_model",
        "stability_check_reported": "llm_stability",
        "data_or_code_shared": "llm_data_code",
    }
    for fname in ("layer1_qa_50_merged.json", "layer1_disagreements_merged.json"):
        fp = RES_DIR / fname
        if not fp.exists():
            continue
        rows = json.loads(fp.read_text(encoding="utf-8"))
        kept = []
        for r in rows:
            d = str(r["doi"]).lower()
            if d in excluded_dois:
                continue
            for (cd, fld), cval in l1_map.items():
                if cd == d and fld in xl_col_for_field:
                    r[xl_col_for_field[fld]] = cval
            kept.append(r)
        for idx, r in enumerate(kept, start=1):
            r["row_id"] = idx
        fp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Updated %s: %d -> %d rows", fname, len(rows), len(kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="+", default=[str(RES_DIR / "layer1_independent_extraction_audit.json")])
    args = ap.parse_args()
    apply_corrections([Path(p) for p in args.audits])


if __name__ == "__main__":
    main()
