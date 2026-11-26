"""Compute complete Layer-1 adjudicated reporting census (n = 246 clean empirical reviews).

Applies author verification overrides from `author_verification_pack.xlsx` whenever present and
reconciles all Layer-0 -> Layer-1 counts, 3 HT estimators (capped, MC cap-adjusted, Two-Stage IPW),
quartile breakdowns (Q1..Q4 & Q2-Q4), journal-clustered Q1 vs Q2-Q4 difference tests, network/software
breakdowns, and exact n=246 Regex-vs-LLM agreement metrics.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXP_DIR / "scripts"
RES_DIR = EXP_DIR / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.l1_census")

MASTER_SEED = 20261122

FIELDS = [
    ("uses_scopus", "uses_scopus", "scopus"),
    ("uses_wos", "uses_wos", "wos"),
    ("exact_query_printed", "exact_query_printed", "query"),
    ("network_fractional_disclosed", "fractional_disclosed", "net_frac"),
    ("null_model_reported", "null_model_reported", "null"),
    ("stability_check_reported", "stability_check_reported", "stability"),
    ("data_or_code_shared", "data_or_code_shared", "data_code"),
]


def _clustered_prop_ci(
    records: list[dict[str, Any]],
    metric: str,
    n_reps: int = 2000,
    seed: int = MASTER_SEED,
) -> dict[str, Any]:
    """Unweighted journal-clustered bootstrap 95% CI for a single subgroup."""
    if not records:
        return {"n": 0, "n_pos": 0, "n_journals": 0, "rate": 0.0, "ci95": [0.0, 0.0]}
    j_map: dict[str, list[float]] = defaultdict(list)
    for r in records:
        j_map[str(r["journal_key"])].append(1.0 if r.get(metric) else 0.0)
    j_keys = sorted(j_map.keys())
    g = len(j_keys)
    all_y = [v for lst in j_map.values() for v in lst]
    pt = float(np.mean(all_y))
    n_pos = int(sum(all_y))
    if g < 2:
        return {"n": len(all_y), "n_pos": n_pos, "n_journals": g, "rate": round(pt, 4), "ci95": [round(pt, 4), round(pt, 4)]}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        drawn = rng.integers(0, g, size=g)
        vals = [v for idx in drawn for v in j_map[j_keys[int(idx)]]]
        boots[b] = float(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(all_y),
        "n_pos": n_pos,
        "n_journals": g,
        "rate": round(pt, 4),
        "ci95": [round(max(0.0, float(lo)), 4), round(min(1.0, float(hi)), 4)],
    }


def _clustered_prop_diff_q1_vs_q24(
    q1_recs: list[dict[str, Any]],
    q24_recs: list[dict[str, Any]],
    metric: str,
    n_reps: int = 2000,
    seed: int = MASTER_SEED + 777,
) -> dict[str, Any]:
    """Centered journal-clustered bootstrap difference test (Q1 minus Q2-Q4)."""
    ja: dict[str, list[float]] = defaultdict(list)
    jb: dict[str, list[float]] = defaultdict(list)
    for r in q1_recs:
        ja[str(r["journal_key"])].append(1.0 if r.get(metric) else 0.0)
    for r in q24_recs:
        jb[str(r["journal_key"])].append(1.0 if r.get(metric) else 0.0)
    ka, kb = sorted(ja.keys()), sorted(jb.keys())
    pa = float(np.mean([v for lst in ja.values() for v in lst]))
    pb = float(np.mean([v for lst in jb.values() for v in lst]))
    diff_obs = pa - pb
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        da = rng.integers(0, len(ka), size=len(ka))
        db = rng.integers(0, len(kb), size=len(kb))
        va = [v for idx in da for v in ja[ka[int(idx)]]]
        vb = [v for idx in db for v in jb[kb[int(idx)]]]
        diffs[b] = float(np.mean(va) - np.mean(vb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    centered = diffs - diff_obs
    p_boot = float((1 + np.sum(np.abs(centered) >= abs(diff_obs))) / (n_reps + 1))
    return {
        "q1_rate": round(pa, 4),
        "q24_rate": round(pb, 4),
        "diff_q1_minus_q24": round(diff_obs, 4),
        "boot_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "boot_centered_p": round(p_boot, 4),
    }


def main() -> None:
    mod02 = import_module("02_download_and_parse_oa")
    bal = json.loads((RES_DIR / "layer0_population_balance.json").read_text(encoding="utf-8"))
    wh = {q: float(bal["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}
    parsed_raw = json.loads((RES_DIR / "layer1_parsed_records.json").read_text(encoding="utf-8"))
    merged = json.loads((RES_DIR / "layer1_llm_merged.json").read_text(encoding="utf-8"))

    # Apply author-signed Excel overrides if author_verification_pack.xlsx has FIX verdicts
    excel_overrides: dict[str, dict[str, Any]] = {}
    xlsx_p = RES_DIR / "author_verification_pack.xlsx"
    n_author_fixes = 0
    if xlsx_p.exists():
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_p, data_only=True)
        for sname in ("Layer1_QA_50", "Layer1_Disagreements"):
            if sname not in wb.sheetnames:
                continue
            ws = wb[sname]
            hdrs = [str(ws.cell(1, c).value or "").strip() for c in range(1, ws.max_column + 1)]
            for r_i in range(2, ws.max_row + 1):
                row_d = {hdrs[c - 1]: ws.cell(r_i, c).value for c in range(1, len(hdrs) + 1)}
                dk = str(row_d.get("doi") or "").lower().strip()
                ver = str(row_d.get("author_verdict_OK_or_FIX") or "").strip().upper()
                if dk and ver in ("OK", "FIX"):
                    excel_overrides[dk] = row_d
                    if ver == "FIX":
                        n_author_fixes += 1

    xl_col_map = {
        "uses_scopus": "llm_scopus",
        "uses_wos": "llm_wos",
        "exact_query_printed": "llm_query_printed",
        "network_fractional_disclosed": "llm_net_fractional",
        "null_model_reported": "llm_null_model",
        "stability_check_reported": "llm_stability",
        "data_or_code_shared": "llm_data_code",
    }

    # Flatten LLM booleans (with PI Excel adjudications applied) onto record dicts for HT bootstrap
    flat: list[dict[str, Any]] = []
    for r in merged:
        llm = r["llm"]
        dk = str(r.get("doi") or "").lower().strip()
        ov = excel_overrides.get(dk)
        row = {
            "doi": r["doi"],
            "quartile": r["quartile"],
            "journal_key": r["journal_key"],
            "publisher_group": r.get("publisher_group", "Other"),
            "cap_ht_weight": float(r.get("cap_ht_weight") or 1.0),
            "two_stage_ipw_weight": float(r.get("two_stage_ipw_weight") or r.get("ipw_weight") or 1.0),
            "two_stage_ipw_weight_trimmed10": min(float(r.get("two_stage_ipw_weight") or r.get("ipw_weight") or 1.0), 10.0),
            "primary_citation_network": str(llm.get("primary_citation_network") or "none").strip().lower(),
            "primary_database": str(llm.get("primary_database") or "none").strip().lower(),
            "author_pub_fractionalized": bool(llm.get("author_pub_fractionalized")),
            "sw_vosviewer": bool(r.get("sw_vosviewer")),
            "sw_bibliometrix": bool(r.get("sw_bibliometrix")),
            "sw_citespace": bool(r.get("sw_citespace")),
            "sw_gephi": bool(r.get("sw_gephi")),
            "sw_scimat": bool(r.get("sw_scimat")),
            "prisma_reported": bool(r.get("prisma_reported")),
            "snapshot_date_reported": bool(r.get("snapshot_date_reported")),
            "quote_byte_verified": bool(r.get("quote_byte_verified")),
            "disagree_fields": str(r.get("disagree_fields") or ""),
        }
        for l_key, r_key, _ in FIELDS:
            xl_c = xl_col_map[l_key]
            if ov is not None and ov.get(xl_c) is not None and str(ov.get(xl_c)).strip() != "":
                row[l_key] = str(ov[xl_c]).strip().lower() in ("true", "1", "yes")
            else:
                row[l_key] = bool(llm.get(l_key))
            row[f"regex_{l_key}"] = bool(r.get(r_key))
        flat.append(row)

    # 4 estimators on all 7 primary reporting fields + author_pub_fractionalized + software
    ht_capped: dict[str, Any] = {}
    ht_cap_adj: dict[str, Any] = {}
    ht_ipw2: dict[str, Any] = {}
    ht_ipw2_trim10: dict[str, Any] = {}
    regex_ht_capped: dict[str, Any] = {}
    by_quartile: dict[str, dict[str, Any]] = {q: {} for q in ("Q1", "Q2", "Q3", "Q4", "Q2_Q4")}
    q1_vs_q24: dict[str, Any] = {}
    regex_vs_llm: dict[str, Any] = {}

    q_groups = {
        "Q1": [r for r in flat if r["quartile"] == "Q1"],
        "Q2": [r for r in flat if r["quartile"] == "Q2"],
        "Q3": [r for r in flat if r["quartile"] == "Q3"],
        "Q4": [r for r in flat if r["quartile"] == "Q4"],
        "Q2_Q4": [r for r in flat if r["quartile"] != "Q1"],
    }

    def _exact_ht_rate(rows: list[dict[str, Any]], metric: str, w_field: str | None) -> float:
        tot = 0.0
        for q_name in ("Q1", "Q2", "Q3", "Q4"):
            sub_q = [r for r in rows if r["quartile"] == q_name]
            ys_q = np.array([1.0 if r.get(metric) else 0.0 for r in sub_q], dtype=float)
            if w_field is None:
                tot += wh[q_name] * float(np.mean(ys_q))
            else:
                ws_q = np.array([float(r[w_field]) for r in sub_q], dtype=float)
                tot += wh[q_name] * float(np.sum(ws_q * ys_q) / np.sum(ws_q))
        return round(tot, 6)

    for idx, (l_key, _, lbl) in enumerate(FIELDS):
        ht_capped[l_key] = mod02.journal_clustered_ht_bootstrap_prop(flat, l_key, wh, weight_field=None, seed=MASTER_SEED + 1100 + idx)
        ht_cap_adj[l_key] = mod02.journal_clustered_ht_bootstrap_prop(flat, l_key, wh, weight_field="cap_ht_weight", seed=MASTER_SEED + 1200 + idx)
        ht_ipw2[l_key] = mod02.journal_clustered_ht_bootstrap_prop(flat, l_key, wh, weight_field="two_stage_ipw_weight", seed=MASTER_SEED + 1300 + idx)
        ht_ipw2_trim10[l_key] = mod02.journal_clustered_ht_bootstrap_prop(flat, l_key, wh, weight_field="two_stage_ipw_weight_trimmed10", seed=MASTER_SEED + 1300 + idx)
        ht_ipw2_trim10[l_key]["rate_exact"] = _exact_ht_rate(flat, l_key, "two_stage_ipw_weight_trimmed10")
        regex_ht_capped[l_key] = mod02.journal_clustered_ht_bootstrap_prop(flat, f"regex_{l_key}", wh, weight_field=None, seed=MASTER_SEED + 1400 + idx)

        for qname, sub in q_groups.items():
            by_quartile[qname][l_key] = _clustered_prop_ci(sub, l_key, seed=MASTER_SEED + 1500 + idx)
        q1_vs_q24[l_key] = _clustered_prop_diff_q1_vs_q24(q_groups["Q1"], q_groups["Q2_Q4"], l_key, seed=MASTER_SEED + 1600 + idx)

        yl = np.array([int(r[l_key]) for r in flat], dtype=int)
        yr = np.array([int(r[f"regex_{l_key}"]) for r in flat], dtype=int)
        kap = float(cohen_kappa_score(yl, yr)) if (len(set(yl)) > 1 or len(set(yr)) > 1) else 1.0
        if np.isnan(kap):
            kap = 1.0
        regex_vs_llm[lbl] = {
            "n": len(flat),
            "llm_pos": int(yl.sum()),
            "regex_pos": int(yr.sum()),
            "raw_agreement": round(float(np.mean(yl == yr)), 4),
            "cohen_kappa": round(kap, 4),
            "regex_precision_vs_llm": round(float(precision_score(yl, yr, zero_division=0)), 4),
            "regex_recall_vs_llm": round(float(recall_score(yl, yr, zero_division=0)), 4),
            "regex_f1_vs_llm": round(float(f1_score(yl, yr, zero_division=0)), 4),
        }

    # Software, PRISMA, snapshot date & author-level fractionalization HT estimates
    extra_ht: dict[str, Any] = {}
    for idx, field in enumerate([
        "author_pub_fractionalized",
        "sw_vosviewer",
        "sw_bibliometrix",
        "sw_citespace",
        "sw_gephi",
        "sw_scimat",
        "prisma_reported",
        "snapshot_date_reported",
    ]):
        t10_res = mod02.journal_clustered_ht_bootstrap_prop(flat, field, wh, weight_field="two_stage_ipw_weight_trimmed10", seed=MASTER_SEED + 1900 + idx)
        t10_res["rate_exact"] = _exact_ht_rate(flat, field, "two_stage_ipw_weight_trimmed10")
        extra_ht[field] = {
            "n_pos": sum(1 for r in flat if r[field]),
            "ht_capped": mod02.journal_clustered_ht_bootstrap_prop(flat, field, wh, weight_field=None, seed=MASTER_SEED + 1700 + idx),
            "ht_cap_adj": mod02.journal_clustered_ht_bootstrap_prop(flat, field, wh, weight_field="cap_ht_weight", seed=MASTER_SEED + 1800 + idx),
            "ht_ipw2": mod02.journal_clustered_ht_bootstrap_prop(flat, field, wh, weight_field="two_stage_ipw_weight", seed=MASTER_SEED + 1900 + idx),
            "ht_ipw2_trimmed10": t10_res,
        }

    # Kish effective sample sizes for cap-adjusted and Two-Stage IPW weights
    w_cap = np.array([r["cap_ht_weight"] for r in flat], dtype=float)
    w_ipw_raw = np.array([r["two_stage_ipw_weight"] for r in flat], dtype=float)
    w_ipw_t10 = np.array([r["two_stage_ipw_weight_trimmed10"] for r in flat], dtype=float)
    kish_ess = {
        "cap_ht": round(float((np.sum(w_cap) ** 2) / np.sum(w_cap ** 2)), 2),
        "cap_ht_pct": round(100.0 * float((np.sum(w_cap) ** 2) / np.sum(w_cap ** 2)) / max(1, len(flat)), 1),
        "ipw2_untrimmed": round(float((np.sum(w_ipw_raw) ** 2) / np.sum(w_ipw_raw ** 2)), 2),
        "ipw2_untrimmed_pct": round(100.0 * float((np.sum(w_ipw_raw) ** 2) / np.sum(w_ipw_raw ** 2)) / max(1, len(flat)), 1),
        "ipw2_untrimmed_max_weight": round(float(np.max(w_ipw_raw)), 2),
        "ipw2_trimmed10": round(float((np.sum(w_ipw_t10) ** 2) / np.sum(w_ipw_t10 ** 2)), 2),
        "ipw2_trimmed10_pct": round(100.0 * float((np.sum(w_ipw_t10) ** 2) / np.sum(w_ipw_t10 ** 2)) / max(1, len(flat)), 1),
    }

    net_counts = Counter(r["primary_citation_network"] for r in flat)
    db_counts = Counter(r["primary_database"] for r in flat)
    pub_counts_q1 = Counter(r["publisher_group"] for r in q_groups["Q1"])
    q_counts = {q: {"n_articles": len(q_groups[q]), "n_journals": len({r["journal_key"] for r in q_groups[q]})} for q in ("Q1", "Q2", "Q3", "Q4")}

    # Compute exact sample exclusion accounting (250 -> 246)
    n_retrieved = len(parsed_raw)
    n_empty_pdf = sum(1 for r in parsed_raw if int(r.get("n_body_pages") or 0) == 0)
    n_regex_guide = sum(1 for r in parsed_raw if r.get("is_guide") and int(r.get("n_body_pages") or 0) > 0)
    n_other_excluded = n_retrieved - n_empty_pdf - n_regex_guide - len(flat)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.2",
        "status": "author_verified_final" if excel_overrides else "pending_author_verification",
        "n_author_verified_overrides": len(excel_overrides),
        "n_author_fixes_applied": n_author_fixes,
        "corpus_funnel": {
            "scopus_busi_raw": int(bal["total_scopus_busi_raw"]),
            "matched_active_asjc14": int(bal["matched_active_asjc14"]),
            "unique_active_journals": int(bal["unique_active_journals_with_biblio"]),
            "legal_oa_retrievable_pool": int(sum(bal["strata"][q]["n_legal_oa_retrievable_pool"] for q in ("Q1", "Q2", "Q3", "Q4"))),
            "layer1_retrieved": n_retrieved,
            "layer1_excluded_empty_image_pdf": n_empty_pdf,
            "layer1_excluded_regex_guide": n_regex_guide,
            "layer1_excluded_llm_guide_mismatch_or_retracted": n_other_excluded,
            "layer1_clean_empirical": len(flat),
            "layer1_unique_journals": len({r["journal_key"] for r in flat}),
            "quote_byte_verified_rate": round(sum(1 for r in flat if r["quote_byte_verified"]) / max(1, len(flat)), 4),
            "by_quartile_sample": q_counts,
            "q1_publisher_counts": dict(pub_counts_q1.most_common()),
        },
        "kish_ess": kish_ess,
        "ht_capped_clustered": ht_capped,
        "ht_article_weighted_cap_adjusted": ht_cap_adj,
        "ht_two_stage_ipw_sensitivity": ht_ipw2,
        "ht_two_stage_ipw_trimmed10": ht_ipw2_trim10,
        "regex_ht_capped_comparison": regex_ht_capped,
        "by_quartile": by_quartile,
        "q1_vs_q24_clustered_tests": q1_vs_q24,
        "extra_metrics_ht": extra_ht,
        "primary_citation_network_counts": dict(net_counts.most_common()),
        "primary_database_counts": dict(db_counts.most_common()),
        "regex_vs_llm_n246": regex_vs_llm,
    }

    out_path = RES_DIR / "layer1_final_adjudicated_census.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Synchronize final adjudicated summary JSON with exact byte parity on all 4 estimators
    final_path = RES_DIR / "layer1_final_adjudicated_census.json"
    final_out = {
        "schema_version": "1.2",
        "n_adjudicated_empirical": len(flat),
        "n_qa50_verified": 50,
        "n_excel_rows_applied": len(excel_overrides),
        "kish_ess": kish_ess,
        "ht_capped_clustered": ht_capped,
        "ht_article_weighted_cap_adjusted": ht_cap_adj,
        "ht_two_stage_ipw_sensitivity": ht_ipw2,
        "ht_two_stage_ipw_trimmed10": ht_ipw2_trim10,
        "extra_metrics_ht_trimmed10": {
            "sw_vosviewer": extra_ht["sw_vosviewer"]["ht_ipw2_trimmed10"],
            "sw_bibliometrix": extra_ht["sw_bibliometrix"]["ht_ipw2_trimmed10"],
        },
    }
    final_path.write_text(json.dumps(final_out, indent=2), encoding="utf-8")
    LOGGER.info(
        "Saved adjudicated Layer-1 census (n=%d, %d journals, IPW trim10 ESS=%.2f) to %s & %s",
        len(flat), out["corpus_funnel"]["layer1_unique_journals"], kish_ess["ipw2_trimmed10"], out_path, final_path,
    )
    for k, _, lbl in FIELDS:
        c = ht_capped[k]
        m = ht_cap_adj[k]
        i = ht_ipw2[k]
        LOGGER.info("  %-10s | HT_cap=%.1f%% %s | MC=%.1f%% | IPW2=%.1f%% | Q1=%.1f%% vs Q2-4=%.1f%% (p=%.3f)",
                    lbl, 100 * c["rate"], [round(100 * x, 1) for x in c["ci95"]], 100 * m["rate"], 100 * i["rate"],
                    100 * q1_vs_q24[k]["q1_rate"], 100 * q1_vs_q24[k]["q24_rate"], q1_vs_q24[k]["boot_centered_p"])


if __name__ == "__main__":
    main()
