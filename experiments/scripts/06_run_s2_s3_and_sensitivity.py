"""Secondary Diagnostics S2 (Scopus vs OpenAlex S∩O), S3 (Auxiliary AI-HRM Check) & Sensitivity Ladder.

Runs cleanly alongside `04_run_counting_audit.py` on `data/processed/layer2_manifest_60.json`:
1. S2 — Decomposed Database Swap (Scopus vs. OpenAlex):
   On Scopus-harvested Layer-2 reviews, executes the balanced OpenAlex equivalent query,
   computes retrieval Jaccard |S ∩ O| / |S ∪ O|, and runs VOS_AS_CPM clustering sensitivity
   strictly on the shared DOI intersection S ∩ O (`|S ∩ O| >= 50`) using 20 seeds x best-of-10 starts
   and 2,000 journal-clustered bootstrap replicates.
2. S3 — Auxiliary AI & HRM / Digital-Workplace Topic Check:
   Identifies AI / digital-workplace / HRM / gig / algorithmic-management reviews in the
   Layer-2 manifest via a deterministic topic lexicon to inspect theme-share sensitivity.
3. Algorithmic Sensitivity Ladder (20 seeds, 2,000 journal-clustered bootstrap replicates):
   - Core threshold grid: core_top_n in {100, 150, 250} (VOS_AS_CPM, best-of-10 starts)
   - Unmerged ablation: merge_small_clusters=False vs True
   - Single-start ablation: n_random_starts=1 vs 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = EXP_DIR / "src_audit"
SCRIPTS_DIR = EXP_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from importlib import import_module

from counting_engine import (  # noqa: E402
    _run_seeds,
    build_sparse_edges_from_works,
    edges_to_igraph,
    evaluate_counting_sensitivity,
    extract_shared_lcc_edges,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.s2s3")

PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"
MASTER_SEED = 20261122

# Frozen S3 topic filter (selects AI / HRM / workplace / gig / digital-work reviews deterministically)
S3_TOPIC_RE = re.compile(
    r"\b(artificial intelligence|\bai\b|human resource|\bhrm\b|workplace|employee|worker|gig economy|"
    r"algorithmic management|digital labor|digital labour|human capital|organizational behavior|"
    r"leadership|psychological capital|working poor|in-work poverty|future of work|talent management|"
    r"diversity|inclusion|work[- ]life)\b",
    re.I,
)

# Frozen S3 dark-side worker theme lexicon (Pan & Froese 2023; Fang et al. 2026; Wei et al. 2026)
S3_DARK_SIDE_RE = re.compile(
    r"\b(surveillance|monitoring|algorithmic control|burnout|technostress|stress|exhaustion|"
    r"job insecurity|unemployment|displacement|deskilling|precarit|precarious|exploitation|"
    r"autonomy loss|alienation|bias|discrimination|inequality|privacy|ethical|ethics|dark side|"
    r"well[- ]?being|mental health|turnover|layoff|working poor|in-work poverty|harassment|"
    r"exclusion|marginaliz|anxiety|fatigue)\b",
    re.I,
)


def run_s2_database_swap(
    manifest: list[dict[str, Any]], oakey: str | None = None, max_reviews: int = 38, n_seeds: int = 20
) -> dict[str, Any]:
    """Decompose Scopus vs. OpenAlex retrieval Jaccard from S∩O shared-DOI reference-graph AMI."""
    mod03 = import_module("03_build_repro_corpora")
    mod04 = import_module("04_run_counting_audit")
    s2_cache_dir = EXP_DIR / "data" / "raw" / "s2_oa_cache"
    s2_cache_dir.mkdir(parents=True, exist_ok=True)

    scopus_items = [m for m in manifest if m.get("harvest_source") == "scopus_to_openalex"][:max_reviews]
    rows: list[dict[str, Any]] = []
    for item in scopus_items:
        corp = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
        s_works = corp["works"] if isinstance(corp, dict) else corp
        s_by_doi = {mod03.norm_doi(w.get("doi")): w for w in s_works if w.get("doi")}
        if len(s_by_doi) < 50:
            continue

        c_path = s2_cache_dir / item["corpus_file"]
        if c_path.exists():
            cd = json.loads(c_path.read_text(encoding="utf-8"))
            oa_total, o_works, oa_ok = int(cd.get("oa_total", 0)), cd.get("works", []), bool(cd.get("oa_ok", False))
        elif not oakey:
            raise FileNotFoundError(f"Missing cached S2 corpus {c_path} and no OPENALEX_API_KEY provided")
        else:
            try:
                oa_total, o_works, oa_ok = mod03.fetch_complete_openalex_corpus(item, oakey, max_complete_works=10_000)
                c_path.write_text(json.dumps({"oa_total": oa_total, "oa_ok": oa_ok, "works": o_works}, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("S2 OpenAlex fetch halted for %s: %s", item.get("doi"), exc)
                rows.append({"doi": item["doi"], "quartile": item["quartile"], "journal_key": item["journal_key"], "status": f"oa_err:{exc}"})
                if "budget" in str(exc).lower():
                    break
                continue
        if not oa_ok or len(o_works) < 50:
            rows.append(
                {
                    "doi": item["doi"],
                    "quartile": item["quartile"],
                    "journal_key": item["journal_key"],
                    "status": f"oa_skip:{oa_total}",
                }
            )
            continue
        o_by_doi = {mod03.norm_doi(w.get("doi")): w for w in o_works if w.get("doi")}
        s_set, o_set = set(s_by_doi.keys()), set(o_by_doi.keys())
        inter = sorted(s_set & o_set)
        union_sz = len(s_set | o_set)
        jacc = len(inter) / max(1, union_sz)
        if len(inter) < 50:
            rows.append(
                {
                    "doi": item["doi"],
                    "quartile": item["quartile"],
                    "journal_key": item["journal_key"],
                    "n_scopus": len(s_set),
                    "n_openalex": len(o_set),
                    "n_intersection": len(inter),
                    "retrieval_jaccard": round(jacc, 4),
                    "status": "small_intersection",
                }
            )
            continue

        # Compare clustering on full Scopus corpus vs S∩O intersection (measuring database-boundary shift)
        net_type = item.get("network_type", "bibliographic_coupling")
        core_n = int(item.get("core_top_n") or 150)
        inter_works = [s_by_doi[d] for d in inter]
        try:
            inodes, ief, ier = build_sparse_edges_from_works(inter_works, net_type, core_top_n=core_n)
            res_inter = evaluate_counting_sensitivity(
                inodes, ief, ier, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=10, merge_small_clusters=True
            )
            rows.append(
                {
                    "doi": item["doi"],
                    "quartile": item["quartile"],
                    "journal_key": item["journal_key"],
                    "network_type": net_type,
                    "n_scopus": len(s_set),
                    "n_openalex": len(o_set),
                    "n_intersection": len(inter),
                    "retrieval_jaccard": round(jacc, 4),
                    "delta_count_on_intersection": res_inter.delta_count_mid,
                    "core_reassignment_on_intersection": res_inter.core_reassignment,
                    "status": "ok",
                }
            )
        except ValueError as exc:
            rows.append({"doi": item["doi"], "quartile": item["quartile"], "retrieval_jaccard": round(jacc, 4), "status": f"err:{exc}"})
        time.sleep(0.08)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict[str, Any] = {
        "n_scopus_attempted": len(scopus_items),
        "n_ok": len(ok_rows),
        "mean_retrieval_jaccard": round(float(np.mean([r["retrieval_jaccard"] for r in ok_rows])), 4) if ok_rows else None,
        "median_retrieval_jaccard": round(float(np.median([r["retrieval_jaccard"] for r in ok_rows])), 4) if ok_rows else None,
    }
    if len(ok_rows) >= 3:
        vals = np.array([float(r["delta_count_on_intersection"]) for r in ok_rows], dtype=float)
        jkeys = [r["journal_key"] for r in ok_rows]
        pt, lo, hi = mod04.stratified_journal_cluster_bootstrap_ci(vals, jkeys, n_reps=2000)
        summary["delta_count_intersection_mean_ci95"] = {"mean": pt, "ci95": [lo, hi]}
    summary["reviews"] = rows
    return summary


def run_s3_ai_hrm_subaudit(manifest: list[dict[str, Any]], n_seeds: int = 10) -> dict[str, Any]:
    """Focal AI & HRM / Workplace dark-side worker theme rank/share sensitivity under full vs fractional counting."""
    focal_items = [
        m for m in manifest
        if S3_TOPIC_RE.search(str(m.get("title") or "") + " " + str(m.get("verbatim_query") or ""))
        and m.get("network_type") == "bibliographic_coupling"
    ]
    rows: list[dict[str, Any]] = []
    for item in focal_items:
        corp = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
        works = corp["works"] if isinstance(corp, dict) else corp
        core_n = int(item.get("core_top_n") or 150)
        nodes, ef, er = build_sparse_edges_from_works(works, "bibliographic_coupling", core_top_n=core_n)
        lcc_idx, f_sub, r_sub = extract_shared_lcc_edges(len(nodes), ef, er)
        if len(lcc_idx) < 30 or not f_sub or not r_sub:
            continue
        lcc_nodes = [nodes[i] for i in lcc_idx]
        dark_flags = [bool(S3_DARK_SIDE_RE.search(str(w.get("title") or ""))) for w in lcc_nodes]
        n_dark = sum(dark_flags)
        if n_dark < 3:
            continue

        sf = edges_to_igraph(len(lcc_idx), f_sub, assoc_strength=True)
        sr = edges_to_igraph(len(lcc_idx), r_sub, assoc_strength=True)
        f_parts = _run_seeds(sf, "CPM", 1.0, n_seeds, n_random_starts=5, merge_small_clusters=True, min_cluster_size=5)
        r_parts = _run_seeds(sr, "CPM", 1.0, n_seeds, n_random_starts=5, merge_small_clusters=True, min_cluster_size=5)

        def _dark_metrics(parts: list[list[int]]) -> tuple[float, float, float]:
            max_shares, top_ranks, herfs = [], [], []
            for mem in parts:
                c_tot: dict[int, int] = defaultdict(int)
                c_dark: dict[int, int] = defaultdict(int)
                for cid, is_d in zip(mem, dark_flags):
                    c_tot[cid] += 1
                    if is_d:
                        c_dark[cid] += 1
                # Rank clusters by size descending (rank 1 = largest cluster)
                ordered_c = sorted(c_tot.keys(), key=lambda c: (-c_tot[c], c))
                # Cluster with highest dark-side share
                best_c = max(ordered_c, key=lambda c: (c_dark[c] / max(1, c_tot[c]), c_dark[c]))
                max_shares.append(c_dark[best_c] / max(1, c_tot[best_c]))
                top_ranks.append(float(ordered_c.index(best_c) + 1))
                # Concentration (HHI) of dark-side papers across clusters
                herfs.append(sum((c_dark[c] / n_dark) ** 2 for c in ordered_c))
            return float(np.mean(max_shares)), float(np.mean(top_ranks)), float(np.mean(herfs))

        f_sh, f_rk, f_hhi = _dark_metrics(f_parts)
        r_sh, r_rk, r_hhi = _dark_metrics(r_parts)
        cross_ami = float(np.mean([ami(f_parts[i], r_parts[j]) for i in range(n_seeds) for j in range(n_seeds)]))
        rows.append(
            {
                "doi": item["doi"],
                "quartile": item["quartile"],
                "title": item["title"],
                "n_lcc": len(lcc_idx),
                "n_dark_side_docs": n_dark,
                "dark_side_corpus_share": round(n_dark / len(lcc_idx), 4),
                "full_peak_dark_cluster_share": round(f_sh, 4),
                "frac_peak_dark_cluster_share": round(r_sh, 4),
                "delta_peak_dark_share": round(r_sh - f_sh, 4),
                "full_dark_cluster_rank": round(f_rk, 2),
                "frac_dark_cluster_rank": round(r_rk, 2),
                "full_dark_hhi": round(f_hhi, 4),
                "frac_dark_hhi": round(r_hhi, 4),
                "cross_ami_default_gamma": round(cross_ami, 4),
            }
        )

    return {
        "n_focal_bc_reviews": len(focal_items),
        "n_with_dark_side_ge_3": len(rows),
        "mean_delta_peak_dark_share": round(float(np.mean([r["delta_peak_dark_share"] for r in rows])), 4) if rows else None,
        "mean_abs_rank_shift": round(float(np.mean([abs(r["frac_dark_cluster_rank"] - r["full_dark_cluster_rank"]) for r in rows])), 3) if rows else None,
        "reviews": rows,
    }


def _eval_sens_item(task: tuple[dict[str, Any], int, bool, int, int]) -> tuple[float | None, str, str, bool]:
    item, cap_n, do_merge, n_starts, n_seeds = task
    corp = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
    works = corp["works"] if isinstance(corp, dict) else corp
    net_type = item.get("network_type", "bibliographic_coupling")
    try:
        nodes, ef, er = build_sparse_edges_from_works(works, net_type, core_top_n=cap_n)
        res = evaluate_counting_sensitivity(
            nodes, ef, er, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=n_starts, merge_small_clusters=do_merge
        )
        if res.degenerate_k:
            return None, item["journal_key"], item["quartile"], True
        return float(res.delta_count_mid), item["journal_key"], item["quartile"], False
    except ValueError:
        return None, item["journal_key"], item["quartile"], False


def run_sensitivity_ladder(manifest: list[dict[str, Any]], n_seeds: int = 20, workers: int = 8) -> dict[str, Any]:
    """Run core_top_n {100, 150, 250} grid + unmerged + single-start ablations in parallel."""
    mod04 = import_module("04_run_counting_audit")
    bal = json.loads((RES_DIR / "layer0_population_balance.json").read_text(encoding="utf-8"))
    q4_w = {q: float(bal["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    arms = [
        ("top100_vos", 100, True, 10),
        ("top150_vos_nomerge", 150, False, 10),
        ("top150_vos_1start", 150, True, 1),
        ("top250_vos", 250, True, 10),
    ]
    out: dict[str, Any] = {}
    for arm_name, cap_n, do_merge, n_starts in arms:
        tasks = [(item, cap_n, do_merge, n_starts, n_seeds) for item in manifest]
        vals, jkeys, qs = [], [], []
        n_deg = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for d_val, jk, q, is_deg in ex.map(_eval_sens_item, tasks):
                if is_deg:
                    n_deg += 1
                elif d_val is not None:
                    vals.append(d_val)
                    jkeys.append(jk)
                    qs.append(q)
        arr = np.asarray(vals, dtype=float)
        pt, lo, hi = mod04.stratified_journal_cluster_bootstrap_ci(
            arr, jkeys, strata=qs, strata_weights=q4_w, n_reps=2000, enforce_min_stratum_journals=True
        )
        out[arm_name] = {
            "n_nondegenerate": len(vals),
            "n_degenerate": n_deg,
            "ht4_delta_count_mean": pt,
            "ci95": [lo, hi],
            "rope_verdict": mod04.classify_rope_zone(pt, lo, hi),
        }
        LOGGER.info("[Sensitivity %s] n=%d | HT4 Delta=%.4f %s -> %s", arm_name, len(vals), pt, [lo, hi], out[arm_name]["rope_verdict"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(PROC_DIR / "layer2_manifest_60.json"))
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--preserve-s2", action="store_true", help="Keep cached s2_database_swap block without re-querying OpenAlex API.")
    ap.add_argument("--offline-s2", action="store_true", help="Recompute S2 from locally cached OpenAlex responses with zero API calls.")
    args = ap.parse_args()

    out_path = RES_DIR / "layer2_s2_s3_sensitivity.json"
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.preserve_s2 and out_path.exists():
        s2 = json.loads(out_path.read_text(encoding="utf-8")).get("s2_database_swap") or {}
    elif args.skip_s2:
        s2 = {}
    elif args.offline_s2:
        s2 = run_s2_database_swap(manifest, oakey=None, max_reviews=len(manifest), n_seeds=args.n_seeds)
    else:
        mod03 = import_module("03_build_repro_corpora")
        keys = mod03.load_env_keys(REPO_ROOT / ".env")
        oakey = keys["OPENALEX_API_KEY"]
        s2 = run_s2_database_swap(manifest, oakey, max_reviews=len(manifest), n_seeds=args.n_seeds)
    s3 = run_s3_ai_hrm_subaudit(manifest, n_seeds=args.n_seeds)
    sens = run_sensitivity_ladder(manifest, n_seeds=args.n_seeds, workers=args.workers)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": "1.2",
        "n_manifest": len(manifest),
        "s2_database_swap": s2,
        "s3_ai_hrm_dark_side": s3,
        "sensitivity_ladder": sens,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved S2/S3/Sensitivity results to %s", out_path)


if __name__ == "__main__":
    main()
