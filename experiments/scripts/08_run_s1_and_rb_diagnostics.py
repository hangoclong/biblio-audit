"""Compute Norm 3 (LinLog/Modularity) Delta_AECS & NVI and exact Phipson-Smyth S1 null diagnostics.

Evaluates all 59 Layer-2 corpora across:
1. Norm 3 (`Leiden_RBConfig` / LinLog-Modularity) under `biblio-audit` v1.0.0 (`20` seeds x best-of-`10`
   starts, CWTS degree-volume small-cluster merging `< 5`) to produce HT4 `Delta_AECS`, `VI`, `NVI`,
   `Delta_count`, and core reassignment with 2,000 journal-clustered bootstrap replicates.
2. Exact Phipson & Smyth (2010) Monte Carlo p-values `(1 + #{Q_null >= Q_obs}) / (R + 1)` (`R = 100`
   degree-preserving causal temporal bipartite MCMC rewires, `10|E|` swap attempts enforcing
   `t_cited <= t_citing`) for BOTH fractional counting and full counting.

Saves deterministic output to `experiments/results/layer2_s1_and_rb_diagnostics.json`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

EXP_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = EXP_DIR.parent
PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"
sys.path.insert(0, str(PAPER_DIR / "biblio_audit"))
sys.path.insert(0, str(EXP_DIR / "scripts"))
sys.path.insert(0, str(EXP_DIR / "src_audit"))

from biblio_audit.engine import audit_artifact  # noqa: E402
from biblio_audit.networks import build_network_artifact  # noqa: E402
from counting_engine import (  # noqa: E402
    _leiden_partition,
    edges_to_igraph,
    merge_small_clusters_cwts,
    temporal_bipartite_rewire,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.s1_rb_diag")
MASTER_SEED = 20261122


def _select_bc_tls_works(works: list[dict[str, Any]], core_top_n: int) -> list[dict[str, Any]]:
    if len(works) <= core_top_n:
        return list(works)
    ref_freq: Counter[str] = Counter()
    for w in works:
        for r in set(w.get("referenced_works") or []):
            ref_freq[r] += 1
    scored = []
    for idx, w in enumerate(works):
        tls = sum(ref_freq[r] - 1 for r in set(w.get("referenced_works") or []) if ref_freq[r] >= 2)
        scored.append((tls, str(w.get("id", "")), idx))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [works[x[2]] for x in scored[:core_top_n]]


def _run_s1_exact_ps(
    works: list[dict[str, Any]],
    ref_years_lookup: dict[str, int],
    network_type: str,
    core_top_n: int = 150,
    n_rewires: int = 100,
    n_starts: int = 3,
    seed: int = MASTER_SEED,
    counting: str = "fractional",
) -> dict[str, float]:
    """Exact Phipson-Smyth S1 causal temporal bipartite null for fractional or full counting."""
    if network_type == "bibliographic_coupling":
        sub_works = _select_bc_tls_works(works, core_top_n)
        ref_counts: Counter[str] = Counter()
        for w in sub_works:
            for r in set(w.get("referenced_works") or []):
                ref_counts[r] += 1
        keep_refs = sorted(r for r, c in ref_counts.items() if c >= 2)
    else:
        ref_counts = Counter()
        for w in works:
            for r in set(w.get("referenced_works") or []):
                ref_counts[r] += 1
        keep_refs = [r for r, c in ref_counts.most_common(core_top_n) if c >= 2]
        keep_set = set(keep_refs)
        sub_works = [w for w in works if len(set(w.get("referenced_works") or []) & keep_set) >= 2]

    ref_to_idx = {r: i for i, r in enumerate(keep_refs)}
    paper_years: dict[int, int] = {}
    ref_earliest_cited: dict[int, int] = {}
    bip_edges: list[tuple[int, int]] = []
    for p_idx, w in enumerate(sub_works):
        py = int(w.get("publication_year") or 2025)
        paper_years[p_idx] = py
        for r in set(w.get("referenced_works") or []):
            if r in ref_to_idx:
                r_idx = ref_to_idx[r]
                bip_edges.append((p_idx, r_idx))
                if r_idx not in ref_earliest_cited or py < ref_earliest_cited[r_idx]:
                    ref_earliest_cited[r_idx] = py
    ref_years: dict[int, int] = {}
    for r, r_idx in ref_to_idx.items():
        earliest = ref_earliest_cited.get(r_idx, 2025)
        ref_years[r_idx] = min(int(ref_years_lookup[r]), earliest) if r in ref_years_lookup else earliest

    def project_metrics(b_edges: list[tuple[int, int]]) -> tuple[float, float, list[int], list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        if network_type == "bibliographic_coupling":
            n_proj = len(sub_works)
            for p, r in b_edges:
                groups[r].append(p)
        else:
            n_proj = len(ref_to_idx)
            for p, r in b_edges:
                groups[p].append(r)
        proj: dict[tuple[int, int], float] = defaultdict(float)
        for nlist in groups.values():
            nk = len(nlist)
            if nk < 2:
                continue
            inv = (1.0 / (nk - 1.0)) if counting == "fractional" else 1.0
            for a in range(nk):
                u = nlist[a]
                for b in range(a + 1, nk):
                    v = nlist[b]
                    proj[(u, v) if u < v else (v, u)] += inv
        if not proj:
            return 0.0, 0.0, [], []
        g_rb = edges_to_igraph(n_proj, proj, assoc_strength=False)
        comps = g_rb.connected_components()
        lcc = sorted(comps[max(range(len(comps)), key=lambda i: len(comps[i]))])
        sg_rb = g_rb.subgraph(lcc)
        best_q_rb = -float("inf")
        for s in range(n_starts):
            mem_rb, _ = _leiden_partition(sg_rb, "modularity", 1.0, s)
            qv = float(sg_rb.modularity(mem_rb, weights="weight"))
            if qv > best_q_rb:
                best_q_rb = qv
        old_to_new = {old: new for new, old in enumerate(lcc)}
        sub_proj = {(old_to_new[u], old_to_new[v]): w for (u, v), w in proj.items() if u in old_to_new and v in old_to_new}
        g_vos = edges_to_igraph(len(lcc), sub_proj, assoc_strength=True)
        best_q_vos = -float("inf")
        best_mem_vos: list[int] = []
        for s in range(n_starts):
            mem_vos, qv = _leiden_partition(g_vos, "CPM", 1.0, s)
            if qv > best_q_vos:
                best_q_vos = qv
                best_mem_vos = mem_vos
        m_vos = merge_small_clusters_cwts(g_vos, best_mem_vos, min_cluster_size=5)
        return best_q_rb, best_q_vos, lcc, m_vos

    q_obs_rb, q_obs_vos, obs_lcc, obs_mem = project_metrics(bip_edges)
    obs_map = dict(zip(obs_lcc, obs_mem))
    null_rbs, null_voss, null_amis = [], [], []
    for rep in range(n_rewires):
        rw, _ = temporal_bipartite_rewire(bip_edges, paper_years, ref_years, swaps_per_edge=10, seed=seed + rep)
        q_rb, q_vos, n_lcc, n_mem = project_metrics(rw)
        null_rbs.append(q_rb)
        null_voss.append(q_vos)
        if obs_lcc and n_lcc:
            n_map = dict(zip(n_lcc, n_mem))
            common = sorted(set(obs_map.keys()) & set(n_map.keys()))
            if len(common) >= 10:
                null_amis.append(float(ami([obs_map[u] for u in common], [n_map[u] for u in common])))

    p_vos = float((1 + sum(1 for v in null_voss if v >= q_obs_vos)) / (n_rewires + 1))
    p_rb = float((1 + sum(1 for v in null_rbs if v >= q_obs_rb)) / (n_rewires + 1))
    return {
        "q_obs_vos": round(q_obs_vos, 4),
        "q_null_vos_p95": round(float(np.percentile(null_voss, 95)), 4),
        "q_null_vos_max": round(float(max(null_voss)), 4),
        "p_ps_vos": round(p_vos, 4),
        "q_obs_rb": round(q_obs_rb, 4),
        "q_null_rb_p95": round(float(np.percentile(null_rbs, 95)), 4),
        "q_null_rb_max": round(float(max(null_rbs)), 4),
        "p_ps_rb": round(p_rb, 4),
        "ami_null_mean": round(float(np.mean(null_amis)), 4) if null_amis else 0.0,
    }


def _worker(task: tuple[int, dict[str, Any], int, int, int]) -> tuple[int, dict[str, Any]]:
    idx, item, n_seeds, n_starts, n_rewires = task
    raw = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
    works = raw["works"] if isinstance(raw, dict) and "works" in raw else raw
    ref_years = raw.get("ref_years") or {} if isinstance(raw, dict) else {}
    net_type = item.get("network_type", "bibliographic_coupling")
    core_n = int(item.get("core_top_n") or 150)

    art_sel = build_network_artifact(works, network_type=net_type, core_top_n=core_n, denom_scope="selected-core", frac_scheme="vosviewer")
    r_rb = audit_artifact(art_sel, objective="Leiden_RBConfig", n_seeds=n_seeds, n_random_starts=n_starts, min_cluster_size=5)

    s1_frac = _run_s1_exact_ps(works, ref_years, net_type, core_top_n=core_n, n_rewires=n_rewires, n_starts=3, counting="fractional")
    s1_full = _run_s1_exact_ps(works, ref_years, net_type, core_top_n=core_n, n_rewires=n_rewires, n_starts=3, counting="full")

    return idx, {
        "doi": item["doi"],
        "quartile": item["quartile"],
        "journal_key": item["journal_key"],
        "network_type": net_type,
        "rb_v1": r_rb.to_summary_dict(),
        "s1_frac": s1_frac,
        "s1_full": s1_full,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Norm 3 AECS/NVI and exact Phipson-Smyth S1 null diagnostics.")
    ap.add_argument("--manifest", type=str, default=str(PROC_DIR / "layer2_manifest_60.json"))
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--n-rewires", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    t0 = time.time()
    mod04 = import_module("04_run_counting_audit")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    l0 = json.loads((RES_DIR / "layer0_population_balance.json").read_text(encoding="utf-8"))
    q4_w = {q: float(l0["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    ordered: list[dict[str, Any] | None] = [None] * len(manifest)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, (i, item, args.n_seeds, args.n_starts, args.n_rewires)) for i, item in enumerate(manifest)]
        for fut in concurrent.futures.as_completed(futs):
            idx, rec = fut.result()
            ordered[idx] = rec
    recs = [r for r in ordered if r is not None]

    sub = [r for r in recs if not r["rb_v1"]["degenerate_k"]]
    rb_agg = {}
    for m in ("delta_count_mid", "delta_aecs_mid", "nvi_cross", "vi_cross_bits", "core_reassignment"):
        vals = np.array([float(r["rb_v1"][m]) for r in sub])
        pt, lo, hi = mod04.stratified_journal_cluster_bootstrap_ci(
            vals, [r["journal_key"] for r in sub], [r["quartile"] for r in sub], q4_w, enforce_min_stratum_journals=True
        )
        rb_agg[m] = {"mean": pt, "ci95": [lo, hi]}

    def _s1_sum(key: str) -> dict[str, Any]:
        pv = [r[key]["p_ps_vos"] for r in recs]
        pr = [r[key]["p_ps_rb"] for r in recs]
        p_min = round(1.0 / (args.n_rewires + 1), 4)
        return {
            "n": len(recs),
            "vos_p_le_005": sum(p <= 0.05 for p in pv),
            "vos_p_eq_00099": sum(abs(p - p_min) < 1e-6 for p in pv),
            "vos_p_max": max(pv),
            "vos_obs_mean": round(float(np.mean([r[key]["q_obs_vos"] for r in recs])), 4),
            "vos_p95_mean": round(float(np.mean([r[key]["q_null_vos_p95"] for r in recs])), 4),
            "rb_p_le_005": sum(p <= 0.05 for p in pr),
            "rb_p_eq_00099": sum(abs(p - p_min) < 1e-6 for p in pr),
            "rb_p_max": max(pr),
            "rb_obs_mean": round(float(np.mean([r[key]["q_obs_rb"] for r in recs])), 4),
            "rb_p95_mean": round(float(np.mean([r[key]["q_null_rb_p95"] for r in recs])), 4),
            "ami_null_mean": round(float(np.mean([r[key]["ami_null_mean"] for r in recs])), 4),
        }

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "rb_v1_ht4": rb_agg,
        "s1_frac_summary": _s1_sum("s1_frac"),
        "s1_full_summary": _s1_sum("s1_full"),
        "reviews": recs,
    }
    out_p = RES_DIR / "layer2_s1_and_rb_diagnostics.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Saved S1/RB diagnostics in %.1fs -> %s", out["elapsed_sec"], out_p)


if __name__ == "__main__":
    main()
