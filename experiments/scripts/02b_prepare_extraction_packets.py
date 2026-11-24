"""Prepare blind, page-numbered Methods + Captions + Data-Availability packets for structured LLM extraction.

Design specifications:
- ZERO `REGEX_FLAGS` in packet headers (guarantees strict Stage-1 vs Stage-2 independence).
- High character budget (`max_chars=60_000`) with a guaranteed reserved tail budget (`12_000` chars)
  for the final 2 body pages + Data Availability / Appendix pages so late BC/CC figure captions and
  sharing/stability disclosures are NEVER silently truncated.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXP_DIR / "src_audit"
sys.path.insert(0, str(SRC_DIR))

from extract_audit_features import extract_body_pages, unwrap_soft_newlines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.packets")

PROC_DIR = EXP_DIR / "data" / "processed"
RES_DIR = EXP_DIR / "results"
PACKET_DIR = PROC_DIR / "extract_packets"

RELEVANT_PAGE_RE = re.compile(
    r"\b(scopus|web of science|wos|ssci|openalex|title-abs-key|ts\s*=|boolean|search string|search query|"
    r"prisma|methodology|methods|data collection|research design|bibliographic coupling|co-citation|"
    r"vosviewer|bibliometrix|biblioshiny|citespace|fractional counting|full counting|counting method|"
    r"null model|configuration model|stability|sensitivity|nmi|ari|data availability|github|zenodo|osf|figshare|"
    r"fig\.|figure\s+\d+|table\s+\d+|appendix)\b",
    re.I,
)


def build_compact_packet(
    doc_path: Path, meta: dict[str, Any], max_chars: int = 60_000, tail_reserve_chars: int = 12_000
) -> str:
    """Build blind page-numbered text packet reserving guaranteed budget for opening & closing body pages."""
    pages = [(pno, unwrap_soft_newlines(raw).strip()) for pno, raw in extract_body_pages(doc_path)]
    pages = [(pno, txt) for pno, txt in pages if txt]

    # Blind header: strictly bibliographic metadata only — ZERO Round-1 regex hints!
    header = (
        f"=== BLIND PAPER PACKET ===\n"
        f"DOI: {meta.get('doi')}\n"
        f"SCOPUS_ID: {meta.get('scopus_id')}\n"
        f"QUARTILE: {meta.get('quartile')} | YEAR: {meta.get('pub_year')} | JOURNAL: {meta.get('publication_name')}\n"
        f"FORMAT: {meta.get('source_format')} | BODY_PAGES: {len(pages)}\n"
        f"TITLE: {meta.get('title')}\n"
        f"==========================\n"
    )
    if not pages:
        return header

    # Partition pages into: head (first 2), tail (last 2), and middle relevant pages
    n_p = len(pages)
    head_idx = set(range(min(2, n_p)))
    tail_idx = set(range(max(0, n_p - 2), n_p)) - head_idx
    mid_indices = [i for i in range(n_p) if i not in head_idx and i not in tail_idx and RELEVANT_PAGE_RE.search(pages[i][1])]

    # Assemble reserved tail block first so we know its exact size
    tail_blocks: list[str] = []
    tail_used = 0
    for i in sorted(tail_idx):
        pno, txt = pages[i]
        blk = f"\n[PAGE {pno}]\n{txt}\n"
        if tail_used + len(blk) <= tail_reserve_chars:
            tail_blocks.append(blk)
            tail_used += len(blk)
        else:
            rem = max(0, tail_reserve_chars - tail_used - 40)
            if rem > 400:
                tail_blocks.append(blk[:rem] + "\n...[TAIL TRUNCATED]...\n")
                tail_used += rem + 25
            break

    head_mid_budget = max(4_000, max_chars - len(header) - tail_used)
    out_blocks: list[str] = [header]
    used = 0
    for i in sorted(head_idx) + mid_indices:
        pno, txt = pages[i]
        blk = f"\n[PAGE {pno}]\n{txt}\n"
        if used + len(blk) > head_mid_budget:
            rem = max(0, head_mid_budget - used - 40)
            if rem > 500:
                out_blocks.append(blk[:rem] + "\n...[MIDDLE TRUNCATED; TAIL PAGES FOLLOW]...\n")
            break
        out_blocks.append(blk)
        used += len(blk)

    out_blocks.extend(tail_blocks)
    return "".join(out_blocks)


def main() -> None:
    """Create blind sharded extraction packets and shard manifests for parallel full-text extraction."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=60_000)
    args = ap.parse_args()

    parsed_path = RES_DIR / "layer1_parsed_records.json"
    records = json.loads(parsed_path.read_text(encoding="utf-8"))
    emp = [r for r in records if not r.get("is_guide")]

    q1 = sorted([r for r in emp if r["quartile"] == "Q1"], key=lambda r: int(r.get("perm_rank", 0)))
    by_q24 = {
        q: sorted([r for r in emp if r["quartile"] == q], key=lambda r: int(r.get("perm_rank", 0)))
        for q in ("Q2", "Q3", "Q4")
    }
    q24_interleaved: list[dict[str, Any]] = []
    for idx in range(max((len(v) for v in by_q24.values()), default=0)):
        for q in ("Q2", "Q3", "Q4"):
            if idx < len(by_q24[q]):
                q24_interleaved.append(by_q24[q][idx])

    ordered = q1 + q24_interleaved
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    shard_lists: list[list[dict[str, Any]]] = [[] for _ in range(args.n_shards)]

    for global_idx, rec in enumerate(ordered, start=1):
        shard_id = (global_idx - 1) % args.n_shards + 1
        doc_path = EXP_DIR / rec["doc_rel_path"]
        pkt_txt = build_compact_packet(doc_path, rec, max_chars=args.max_chars)
        shard_dir = PACKET_DIR / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rec.get("doi") or rec.get("scopus_id")))[:80]
        pkt_file = shard_dir / f"{global_idx:03d}_{rec['quartile']}_{safe}.txt"
        pkt_file.write_text(pkt_txt, encoding="utf-8")
        shard_lists[shard_id - 1].append(
            {
                "global_idx": global_idx,
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
        idx_path = PACKET_DIR / f"shard_{s_idx}_index.json"
        idx_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Blind Shard %d: %d packets -> %s", s_idx, len(items), idx_path)

    LOGGER.info("Prepared %d blind empirical packets across %d shards (max_chars=%d).", len(ordered), args.n_shards, args.max_chars)


if __name__ == "__main__":
    main()
