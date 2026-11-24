"""Layer-2 Continuation Legal OA Downloader & Blind Packet Builder.

Strictly continues the EXACT SAME locked per-quartile permutations (`MASTER_SEED + 100 + q_idx`)
starting strictly past the final `perm_rank` consumed by Layer 1 (`N = 250`), so Layer 1 remains
100% frozen while providing sufficient pure-random OA papers to reach the locked Layer-2 quota
(`n = 60`: `30 Q1 + 30 Q2-Q4`, `cap <= 3/journal`) without any selection bias or cherry-picking.

Uses 6-thread concurrent legal OA downloads (Elsevier Full-Text API + OpenAlex Green-OA PDFs +
Direct OA PDF, zero anti-bot/CAPTCHA bypass), then orders results strictly by locked `perm_rank`
and emits 8 blind extraction packet shards under `extract_packets_l2_cont/` (0 REGEX_FLAGS).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = EXP_DIR / "src_audit"
SCRIPTS_DIR = EXP_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from importlib import import_module

mod02 = import_module("02_download_and_parse_oa")
mod02b = import_module("02b_prepare_extraction_packets")
from extract_audit_features import parse_pdf_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.l2_cont")

PROC_DIR = EXP_DIR / "data" / "processed"
PDF_DIR = EXP_DIR / "data" / "pdfs"
FULLTEXT_DIR = EXP_DIR / "data" / "fulltext_api"
RES_DIR = EXP_DIR / "results"
CONT_PACKET_DIR = PROC_DIR / "extract_packets_l2_cont"
MASTER_SEED = 20261122


def _fetch_one(task: dict[str, Any], skey: str) -> dict[str, Any] | None:
    """Download & parse one legal OA article via 3-channel automated cascade (zero anti-bot bypass)."""
    q = task["quartile"]
    doi = task["doi"]
    sid = task["scopus_id"]
    stem = mod02.safe_doc_stem(doi, sid)
    pdf_dest = PDF_DIR / q / f"{stem}.pdf"
    txt_dest = FULLTEXT_DIR / q / f"{stem}.txt"

    doc_path: Path | None = None
    channel = ""
    if mod02.is_valid_pdf(pdf_dest):
        doc_path, channel = pdf_dest, "local_pdf"
    elif mod02.is_valid_elsevier_fulltext(txt_dest):
        doc_path, channel = txt_dest, "elsevier_fulltext_api"
    else:
        if task["is_elsevier_doi"] and skey:
            ok, _ = mod02.try_fetch_elsevier_fulltext(doi, txt_dest, skey)
            if ok:
                doc_path, channel = txt_dest, "elsevier_fulltext_api"
        if doc_path is None:
            for ch_lbl, url in task["candidate_urls"]:
                ok, _ = mod02.try_download_pdf(url, pdf_dest)
                if ok:
                    doc_path, channel = pdf_dest, ch_lbl
                    break

    if doc_path is None:
        return None
    try:
        feats = parse_pdf_features(doc_path, title=task["title"])
    except Exception:  # noqa: BLE001
        return None
    if feats.get("is_guide") or feats.get("n_body_pages", 0) == 0:
        return None
    return {
        **{k: v for k, v in task.items() if k not in ("candidate_urls", "is_elsevier_doi")},
        "retrieval_channel": channel,
        "doc_rel_path": str(doc_path.relative_to(EXP_DIR)).replace("\\", "/"),
        **feats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="Q1=130,Q2=55,Q3=55,Q4=40", help="Additional locked-order OA articles for Layer-2 pool")
    ap.add_argument("--per-journal-cap", type=int, default=5, help="Cap per journal within continuation pool")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-shards", type=int, default=8)
    args = ap.parse_args()

    targets = mod02.parse_targets(args.targets)
    keys = mod02.load_env_keys(REPO_ROOT / ".env")
    skey = keys.get("SCOPUS_API_KEY", "")

    df = pd.read_parquet(PROC_DIR / "busi_population.parquet")
    active = df[df["quartile"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()

    # Load Layer-1 already-consumed DOIs & max perm_rank per quartile so continuation starts strictly AFTER Layer 1
    l1_recs = json.loads((RES_DIR / "layer1_parsed_records.json").read_text(encoding="utf-8"))
    l1_dois = {str(r.get("doi") or "").lower() for r in l1_recs}
    max_rank_by_q: dict[str, int] = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        ranks = [int(r.get("perm_rank", -1)) for r in l1_recs if r["quartile"] == q]
        max_rank_by_q[q] = max(ranks) if ranks else -1

    cont_records: list[dict[str, Any]] = []
    for q_idx, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        target = targets.get(q, 40)
        has_channel = (
            (active["oa_pdf_url"].fillna("") != "")
            | (active["oa_green_pdf_count"].fillna(0) > 0)
            | (active["is_elsevier_doi"].fillna(False))
            | (active["oa_url"].fillna("") != "")
        )
        pool = active[(active["quartile"] == q) & (active["is_oa_combined"]) & has_channel].copy()
        pool = pool.sort_values(["scopus_id", "doi"]).reset_index(drop=True)
        q_rng = np.random.default_rng(MASTER_SEED + 100 + q_idx)
        pool = pool.iloc[q_rng.permutation(len(pool))].reset_index(drop=True)

        start_rank = max_rank_by_q[q] + 1
        # Build candidate slice strictly after Layer 1's max_rank
        tasks: list[dict[str, Any]] = []
        for perm_rank in range(start_rank, len(pool)):
            row = pool.iloc[perm_rank]
            doi = str(row["doi"] or "")
            if doi.lower() in l1_dois:
                continue
            tasks.append(
                {
                    "scopus_id": str(row["scopus_id"] or ""),
                    "doi": doi,
                    "quartile": q,
                    "perm_rank": perm_rank,
                    "citescore": float(row["citescore"]) if pd.notna(row["citescore"]) else None,
                    "max_pct_14xx": int(row["max_pct_14xx"]) if pd.notna(row["max_pct_14xx"]) else None,
                    "pub_year": int(row["pub_year"]) if pd.notna(row["pub_year"]) else None,
                    "publication_name": str(row["publication_name"]),
                    "publisher_group": str(row.get("publisher_group") or "Other"),
                    "journal_key": str(row["journal_key"]),
                    "oa_status": str(row["oa_status"]),
                    "oa_propensity_score": float(row["oa_propensity_score"]) if pd.notna(row.get("oa_propensity_score")) else None,
                    "ipw_weight": float(row["ipw_weight"]) if pd.notna(row.get("ipw_weight")) else 1.0,
                    "title": str(row["title"] or ""),
                    "scopus_cited_by": int(row["scopus_cited_by"]),
                    "is_elsevier_doi": bool(row.get("is_elsevier_doi")),
                    "candidate_urls": mod02.ordered_candidate_pdf_urls(row),
                }
            )

        # Process in ordered chunks of 40 concurrently, preserving exact locked perm_rank order on admission!
        got_q: list[dict[str, Any]] = []
        j_cnt: Counter[str] = Counter()
        chunk_sz = 40
        pos = 0
        while len(got_q) < target and pos < len(tasks):
            batch = [t for t in tasks[pos : pos + chunk_sz] if j_cnt[t["journal_key"]] < args.per_journal_cap]
            pos += chunk_sz
            if not batch:
                continue
            results_map: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_fetch_one, t, skey): t["perm_rank"] for t in batch}
                for fut in as_completed(futs):
                    res = fut.result()
                    if res is not None:
                        results_map[res["perm_rank"]] = res
            for rk in sorted(results_map.keys()):
                rec = results_map[rk]
                if len(got_q) >= target:
                    break
                if j_cnt[rec["journal_key"]] >= args.per_journal_cap:
                    continue
                j_cnt[rec["journal_key"]] += 1
                got_q.append(rec)
            LOGGER.info("[%s cont] progress: %d / %d (scanned %d/%d remaining pool, journals=%d)", q, len(got_q), target, pos, len(tasks), len(j_cnt))

        cont_records.extend(got_q)
        LOGGER.info("[%s cont] Done: retrieved %d empirical docs across %d journals", q, len(got_q), len(j_cnt))

    out_rec_path = PROC_DIR / "layer2_continuation_records.json"
    out_rec_path.write_text(json.dumps(cont_records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prepare 8 blind extraction shards strictly for the continuation records (ZERO REGEX_FLAGS!)
    CONT_PACKET_DIR.mkdir(parents=True, exist_ok=True)
    shard_lists: list[list[dict[str, Any]]] = [[] for _ in range(args.n_shards)]
    for g_idx, rec in enumerate(cont_records, start=1):
        s_id = (g_idx - 1) % args.n_shards + 1
        doc_path = EXP_DIR / rec["doc_rel_path"]
        pkt_txt = mod02b.build_compact_packet(doc_path, rec, max_chars=60_000)
        s_dir = CONT_PACKET_DIR / f"shard_{s_id}"
        s_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rec.get("doi") or rec.get("scopus_id")))[:80]
        pkt_file = s_dir / f"C{g_idx:03d}_{rec['quartile']}_{safe}.txt"
        pkt_file.write_text(pkt_txt, encoding="utf-8")
        shard_lists[s_id - 1].append(
            {
                "global_idx": g_idx,
                "quartile": rec["quartile"],
                "doi": rec["doi"],
                "scopus_id": rec["scopus_id"],
                "journal_key": rec["journal_key"],
                "publication_name": rec["publication_name"],
                "pub_year": rec["pub_year"],
                "source_format": rec["source_format"],
                "doc_rel_path": rec["doc_rel_path"],
                "packet_rel_path": str(pkt_file.relative_to(EXP_DIR)).replace("\\", "/"),
            }
        )
    for s_idx, items in enumerate(shard_lists, start=1):
        (CONT_PACKET_DIR / f"shard_{s_idx}_index.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Layer-2 continuation complete: %d empirical docs across %d blind shards -> %s", len(cont_records), args.n_shards, CONT_PACKET_DIR)


if __name__ == "__main__":
    main()
