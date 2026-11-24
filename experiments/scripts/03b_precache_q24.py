"""Atomic read-through cache pre-warmer strictly for Q2-Q4 Layer-2 corpora while 03 harvests Q1.

Uses the exact functions and `query_sig` format from `03_build_repro_corpora.py` and writes atomically
via temporary file rename, never touching Q1 corpora or `layer2_manifest.json`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from importlib import import_module
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = EXP_DIR / "scripts"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.precache_q24")


def main() -> None:
    mod03 = import_module("03_build_repro_corpora")
    keys = mod03.load_env_keys(REPO_ROOT / ".env")
    skey = keys["SCOPUS_API_KEY"]
    oakey = keys["OPENALEX_API_KEY"]
    specs = json.loads((RES_DIR / "layer2_viable_65.json").read_text(encoding="utf-8"))
    q24 = [sp for sp in specs if str(sp.get("quartile")) != "Q1"]
    RAW_CORP_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pre-caching %d Q2-Q4 Layer-2 corpora atomically...", len(q24))

    done = 0
    for idx, sp in enumerate(q24, start=1):
        net_type = str(sp.get("llm_network_type") or sp.get("network_type") or "bibliographic_coupling")
        db_src = str(sp.get("database_source") or "scopus").lower()
        core_n = max(50, min(500, int(sp.get("node_threshold") or sp.get("core_top_n") or 150)))
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sp.get("doi") or sp.get("scopus_id")))[:100] + ".json"
        corp_path = RAW_CORP_DIR / safe_id
        q_sig = f"{db_src}|{sp.get('verbatim_query')}|{sp.get('start_year')}|{sp.get('cutoff_year')}|{sp.get('subjarea_filter')}|{net_type}|{core_n}"

        if corp_path.exists():
            try:
                cached = json.loads(corp_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("query_sig") == q_sig and cached.get("works"):
                    done += 1
                    LOGGER.info("[Q2-4 %d/%d] CACHED %s", idx, len(q24), sp.get("doi"))
                    continue
            except Exception:  # noqa: BLE001
                pass

        join_rate, pct_refs = 1.0, 1.0
        harvest_source = "scopus_to_openalex"
        if "scopus" in db_src:
            total_hits, sc_recs, is_complete = mod03.fetch_scopus_records_complete(sp, skey, 10_000)
            if not is_complete or len(sc_recs) < 50:
                LOGGER.warning("[Q2-4 %d/%d] SKIP scopus %s (hits=%s)", idx, len(q24), sp.get("doi"), total_hits)
                continue
            works, join_rate, pct_refs = mod03.join_scopus_dois_to_openalex(sc_recs, oakey)
            if join_rate < 0.70 or len(works) < 50:
                LOGGER.warning("[Q2-4 %d/%d] SKIP low_join %s (%.3f)", idx, len(q24), sp.get("doi"), join_rate)
                continue
        else:
            harvest_source = "openalex_fallback"
            total_hits, works, is_complete = mod03.fetch_complete_openalex_corpus(sp, oakey, 10_000)
            if not is_complete or len(works) < 50:
                LOGGER.warning("[Q2-4 %d/%d] SKIP oa %s (hits=%s)", idx, len(q24), sp.get("doi"), total_hits)
                continue
            pct_refs = round(sum(1 for w in works if w.get("referenced_works")) / max(1, len(works)), 4)

        lcc_sz = mod03.compute_lcc_size(works, net_type, core_top_n=core_n)
        if lcc_sz < 50:
            LOGGER.warning("[Q2-4 %d/%d] SKIP small_lcc %s (lcc=%d)", idx, len(q24), sp.get("doi"), lcc_sz)
            continue

        ref_years = mod03.fetch_cited_reference_years(works, oakey, network_type=net_type, core_top_n=core_n)
        tmp_path = corp_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "query_sig": q_sig,
                    "harvest_source": harvest_source,
                    "source_total_hits": total_hits,
                    "scopus_to_oa_join_rate": join_rate,
                    "pct_works_with_refs": pct_refs,
                    "is_complete_corpus": is_complete,
                    "works": works,
                    "ref_years": ref_years,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(corp_path)
        done += 1
        LOGGER.info(
            "[Q2-4 %d/%d] SAVED %s | hits=%d | joined=%d (%.1f%%) | lcc=%d | ref_yrs=%d",
            idx, len(q24), sp.get("doi"), total_hits, len(works), 100 * join_rate, lcc_sz, len(ref_years),
        )
    LOGGER.info("Pre-cache Q2-Q4 complete: %d/%d corpora cached", done, len(q24))


if __name__ == "__main__":
    main()
