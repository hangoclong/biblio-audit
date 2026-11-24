"""Pre-harvest ALL remaining Scopus corpus DOI lists tonight using SCOPUS_API_KEY (0 OpenAlex calls).

Saves complete Scopus record lists into `data/raw/scopus_cache/<safe_id>` matching `03_build_repro_corpora.py`
so when OpenAlex resets at 00:00 UTC (07:00 VN), joining all 29 Scopus corpora takes only ~3 minutes!
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
SCOPUS_CACHE_DIR = EXP_DIR / "data" / "raw" / "scopus_cache"
RES_DIR = EXP_DIR / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.scopus_preharvest")


def main() -> None:
    mod03 = import_module("03_build_repro_corpora")
    skey = mod03.load_env_keys(REPO_ROOT / ".env")["SCOPUS_API_KEY"]
    specs = json.loads((RES_DIR / "layer2_viable_65.json").read_text(encoding="utf-8"))
    SCOPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Also seed global ref_years_cache.json right now from existing 18 cached corpora!
    ref_cache = mod03.load_ref_years_cache()
    LOGGER.info("Seeded global ref_years_cache.json with %d OpenAlex reference years", len(ref_cache))

    todo = []
    for sp in specs:
        db_src = str(sp.get("database_source") or "scopus").lower()
        if "scopus" not in db_src:
            continue
        net_type = str(sp.get("llm_network_type") or sp.get("network_type") or "bibliographic_coupling")
        core_n = max(50, min(500, int(sp.get("node_threshold") or sp.get("core_top_n") or 150)))
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sp.get("doi") or sp.get("scopus_id")))[:100] + ".json"
        q_sig = f"{db_src}|{sp.get('verbatim_query')}|{sp.get('start_year')}|{sp.get('cutoff_year')}|{sp.get('subjarea_filter')}|{net_type}|{core_n}"
        corp_p = RAW_CORP_DIR / safe_id
        if corp_p.exists():
            try:
                d = json.loads(corp_p.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d.get("query_sig") == q_sig and d.get("works"):
                    continue
            except Exception:  # noqa: BLE001
                pass
        sc_p = SCOPUS_CACHE_DIR / safe_id
        sc_q_sig = f"{sp.get('verbatim_query')}|{sp.get('start_year')}|{sp.get('cutoff_year')}|{sp.get('subjarea_filter')}"
        if sc_p.exists():
            try:
                sd = json.loads(sc_p.read_text(encoding="utf-8"))
                if isinstance(sd, dict) and sd.get("sc_q_sig") == sc_q_sig and sd.get("records"):
                    continue
            except Exception:  # noqa: BLE001
                pass
        todo.append((sp, safe_id, sc_q_sig))

    LOGGER.info("Pre-harvesting %d remaining Scopus corpora into %s (0 OpenAlex calls)...", len(todo), SCOPUS_CACHE_DIR)
    ok_cnt = 0
    for idx, (sp, safe_id, sc_q_sig) in enumerate(todo, start=1):
        total, recs, is_comp = mod03.fetch_scopus_records_complete(sp, skey, max_complete_works=10_000)
        if is_comp and len(recs) >= 50:
            tmp = (SCOPUS_CACHE_DIR / safe_id).with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"sc_q_sig": sc_q_sig, "total_hits": total, "is_complete": is_comp, "records": recs}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(SCOPUS_CACHE_DIR / safe_id)
            ok_cnt += 1
            LOGGER.info("[%d/%d] CACHED Scopus %s (%s) | hits=%d | recs=%d", idx, len(todo), sp.get("doi"), sp.get("quartile"), total, len(recs))
        else:
            LOGGER.warning("[%d/%d] SKIP Scopus %s | hits=%d | recs=%d | comp=%s", idx, len(todo), sp.get("doi"), total, len(recs), is_comp)
    LOGGER.info("Scopus pre-harvest complete: %d/%d cached cleanly", ok_cnt, len(todo))


if __name__ == "__main__":
    main()
