"""Expanded `biblio-audit` v1.0 Empirical Benchmark across all 59 Layer-2 Management Corpora.

Computes on 10 parallel cores (identical 4-stratum HT weights & 2,000-rep journal-clustered bootstrap CIs):
1. Primary `VOS_AS_CPM` (`selected-core`, `vosviewer` 1/(d_e-1)): adds chance-adjusted `AECS`, `Delta_AECS`,
   `ECS_cross`, `VI` (bits), `NVI`, and `tls_core_turnover_jaccard` (for truncated BC corpora).
2. 3rd VOSviewer Normalization: `VOS_Frac_CPM` (van Eck & Waltman 2009 Eq. 13 Fractionalization + CPM).
3. Pre-Truncation Denominator Sensitivity: `VOS_AS_CPM` with `denom_scope='full-corpus'`.
4. Document-Normalized Fractional Sensitivity: `VOS_AS_CPM` with `frac_scheme='refined'` (Leydesdorff & Park 2017).

Saves deterministic JSON provenance to `experiments/results/layer2_v1_expanded_benchmark.json`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = EXP_DIR.parent
sys.path.insert(0, str(PAPER_DIR / "biblio_audit"))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from biblio_audit.engine import audit_artifact  # noqa: E402
from biblio_audit.networks import build_network_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.v1_bench")

PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"


def _eval_one(task: tuple[int, dict[str, Any], int, int]) -> tuple[int, dict[str, Any]]:
    idx, item, n_seeds, n_starts = task
    raw = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
    works = raw["works"] if isinstance(raw, dict) and "works" in raw else raw
    net_type = item.get("network_type", "bibliographic_coupling")
    core_n = int(item.get("core_top_n") or 150)

    # 1. Primary selected-core VOS_AS_CPM (full v1.0 metrics: AMI + AECS + VI + NVI + TLS turnover)
    art_sel = build_network_artifact(works, network_type=net_type, core_top_n=core_n, denom_scope="selected-core", frac_scheme="vosviewer")
    r_as = audit_artifact(art_sel, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=n_starts, min_cluster_size=5)

    # 2. VOSviewer Fractionalization normalization (`VOS_Frac_CPM`, van Eck & Waltman 2009 Eq. 13)
    r_frac_norm = audit_artifact(art_sel, objective="VOS_Frac_CPM", n_seeds=n_seeds, n_random_starts=n_starts, min_cluster_size=5)

    # 3. Full-corpus pre-truncation denominator (`denom_scope='full-corpus'`)
    art_full_denom = build_network_artifact(works, network_type=net_type, core_top_n=core_n, denom_scope="full-corpus", frac_scheme="vosviewer")
    r_full_denom = audit_artifact(art_full_denom, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=n_starts, min_cluster_size=5)

    # 4. Leydesdorff & Park (2017) document-normalized fractional scheme (`2 / (d_e * (d_e - 1))`)
    art_ref = build_network_artifact(works, network_type=net_type, core_top_n=core_n, denom_scope="selected-core", frac_scheme="refined")
    r_refined = audit_artifact(art_ref, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=n_starts, min_cluster_size=5)

    return idx, {
        "doi": item["doi"],
        "quartile": item["quartile"],
        "journal_key": item["journal_key"],
        "network_type": net_type,
        "tls_core_turnover_jaccard": art_sel.tls_core_turnover_jaccard,
        "vos_as_v1": r_as.to_summary_dict(),
        "vos_frac_norm": r_frac_norm.to_summary_dict(),
        "vos_as_full_corpus_denom": r_full_denom.to_summary_dict(),
        "vos_as_refined_lp2017": r_refined.to_summary_dict(),
    }


def main() -> None:
    t0 = time.time()
    mod04 = import_module("04_run_counting_audit")
    manifest = json.loads((PROC_DIR / "layer2_manifest_60.json").read_text(encoding="utf-8"))
    bal = json.loads((RES_DIR / "layer0_population_balance.json").read_text(encoding="utf-8"))
    q4_w = {q: float(bal["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    n_seeds = 20
    n_starts = 10
    workers = max(1, min(10, (os.cpu_count() or 4) - 2))
    LOGGER.info("Running expanded v1.0 benchmark across %d corpora (seeds=%d, starts=%d, workers=%d)...", len(manifest), n_seeds, n_starts, workers)

    tasks = [(i, item, n_seeds, n_starts) for i, item in enumerate(manifest)]
    ordered: list[dict[str, Any] | None] = [None] * len(manifest)
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_eval_one, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            idx, rec = fut.result()
            ordered[idx] = rec
            done += 1
            if done % 10 == 0 or done == len(manifest):
                LOGGER.info("  [%d/%d] completed (%.1fs elapsed)", done, len(manifest), time.time() - t0)

    recs = [r for r in ordered if r is not None]

    def _agg(arm: str, metric: str) -> dict[str, Any]:
        sub = [r for r in recs if not r[arm]["degenerate_k"]]
        vals = np.array([float(r[arm][metric]) for r in sub], dtype=float)
        pt, lo, hi = mod04.stratified_journal_cluster_bootstrap_ci(
            vals,
            [r["journal_key"] for r in sub],
            strata=[r["quartile"] for r in sub],
            strata_weights=q4_w,
            enforce_min_stratum_journals=True,
        )
        return {
            "n": len(sub),
            "n_journals": len({r["journal_key"] for r in sub}),
            "mean": pt,
            "ci95": [lo, hi],
            "median": round(float(np.median(vals)), 4),
            "rope_verdict": mod04.classify_rope_zone(pt, lo, hi) if "delta_count" in metric else None,
        }

    bc_turn = [float(r["tls_core_turnover_jaccard"]) for r in recs if r["tls_core_turnover_jaccard"] is not None]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_reviews": len(recs),
        "n_seeds": n_seeds,
        "n_starts": n_starts,
        "vos_as_v1": {
            "delta_count_mid": _agg("vos_as_v1", "delta_count_mid"),
            "delta_aecs_mid": _agg("vos_as_v1", "delta_aecs_mid"),
            "aecs_cross_mid": _agg("vos_as_v1", "aecs_cross_mid"),
            "ecs_cross_mid": _agg("vos_as_v1", "ecs_cross_mid"),
            "vi_cross_bits": _agg("vos_as_v1", "vi_cross_bits"),
            "nvi_cross": _agg("vos_as_v1", "nvi_cross"),
            "core_reassignment": _agg("vos_as_v1", "core_reassignment"),
            "bc_tls_turnover_jaccard_n": len(bc_turn),
            "bc_tls_turnover_jaccard_mean": round(float(np.mean(bc_turn)), 4) if bc_turn else None,
            "bc_tls_turnover_jaccard_median": round(float(np.median(bc_turn)), 4) if bc_turn else None,
        },
        "vos_frac_norm": {
            "delta_count_mid": _agg("vos_frac_norm", "delta_count_mid"),
            "delta_aecs_mid": _agg("vos_frac_norm", "delta_aecs_mid"),
            "core_reassignment": _agg("vos_frac_norm", "core_reassignment"),
            "nvi_cross": _agg("vos_frac_norm", "nvi_cross"),
        },
        "vos_as_full_corpus_denom": {
            "delta_count_mid": _agg("vos_as_full_corpus_denom", "delta_count_mid"),
            "delta_aecs_mid": _agg("vos_as_full_corpus_denom", "delta_aecs_mid"),
            "core_reassignment": _agg("vos_as_full_corpus_denom", "core_reassignment"),
        },
        "vos_as_refined_lp2017": {
            "delta_count_mid": _agg("vos_as_refined_lp2017", "delta_count_mid"),
            "delta_aecs_mid": _agg("vos_as_refined_lp2017", "delta_aecs_mid"),
            "core_reassignment": _agg("vos_as_refined_lp2017", "core_reassignment"),
        },
        "reviews": recs,
    }
    out_p = RES_DIR / "layer2_v1_expanded_benchmark.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s in %.1fs", out_p, time.time() - t0)
    for arm in ("vos_as_v1", "vos_frac_norm", "vos_as_full_corpus_denom", "vos_as_refined_lp2017"):
        d = summary[arm]["delta_count_mid"]
        a = summary[arm]["delta_aecs_mid"]
        cr = summary[arm]["core_reassignment"]
        LOGGER.info("  %-26s | Delta=%.3f %s (%s) | Delta_AECS=%.3f | Reassign=%.1f%%", arm, d["mean"], d["ci95"], d["rope_verdict"], a["mean"], 100 * cr["mean"])


if __name__ == "__main__":
    main()
