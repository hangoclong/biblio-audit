"""Compute Granularity, Mixing, and Coherence Diagnostics across all 59 Layer-2 corpora (`09_run_granularity_and_coherence_diagnostics.py`).

Evaluates deterministically (exact `biblio-audit` v1.0.0 / `counting_engine` seed schedule):
1. Unmatched fixed-gamma = 1.0 benchmark across all 59 corpora (`K_frac(1.0) != K_full(1.0)` share,
   Delta-K distribution, fixed-gamma = 1.0 HT4 Delta_count & core reassignment with 2,000 journal-
   clustered bootstrap replicates).
2. Effective granularity ratio G_frac(gamma*) / G_full(1.0) (Waltman et al. 2020; G = n / sum_c n_c^2)
   under both baseline (`min_cluster_size=5`) and unmerged (`min_cluster_size=1`) VOS_AS_CPM, plus
   journal-clustered Spearman correlation between Delta_count and |log(G_frac / G_full)|.
3. S1 causal temporal bipartite MCMC mixing diagnostics: exact bipartite edge-retention fraction
   |E_obs intersect E_rewired| / |E_obs| across R = 100 rewires (10|E| swap attempts) on all 59 corpora,
   plus bipartite density & degree diagnostics for the 3 high-null-VOS corpora (0.68-0.74).
4. External non-citation title TF-IDF cosine coherence (mean intra-cluster minus inter-cluster cosine
   similarity) on matched-K full vs. fractional Medoid partitions across all 31 BC corpora, plus paired
   journal-clustered contrasts on within-rule AMI and Leiden modularity, and n=19 out-of-band vs. n=40
   in-band corpus-size ratio comparison.

Saves deterministic output to `experiments/results/layer2_granularity_and_coherence_diagnostics.json`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score as ami

EXP_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = EXP_DIR.parent
PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"
sys.path.insert(0, str(PAPER_DIR / "biblio_audit"))
sys.path.insert(0, str(EXP_DIR / "scripts"))
sys.path.insert(0, str(EXP_DIR / "src_audit"))

from biblio_audit.engine import (  # noqa: E402
    _find_plateau,
    _major_cluster_stats,
    _run_seeds,
    edges_to_igraph,
    extract_shared_lcc_edges,
)
from biblio_audit.metrics import hungarian_core_match, select_medoid_partition  # noqa: E402
from biblio_audit.networks import build_network_artifact  # noqa: E402
from counting_engine import temporal_bipartite_rewire  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.gran_coh")
MASTER_SEED = 20261122


def _effective_granularity(membership: list[int]) -> float:
    """Waltman et al. (2020) effective granularity G = n / sum_c n_c^2."""
    n = len(membership)
    if n == 0:
        return 0.0
    counts = Counter(membership)
    denom = sum(sz * sz for sz in counts.values())
    return float(n / denom) if denom > 0 else 0.0


def _mean_within_ami(parts: list[list[int]]) -> float:
    n = len(parts)
    if n < 2:
        return 1.0
    return float(np.mean([ami(parts[i], parts[j]) for i in range(n) for j in range(i + 1, n)]))


def _title_tfidf_coherence(membership: list[int], node_records: list[dict[str, Any]]) -> dict[str, float] | None:
    """Compute mean intra-cluster, inter-cluster, and contrast (intra - inter) title TF-IDF cosine similarity."""
    texts = [
        str(r.get("title") or "").strip()
        if not str(r.get("title") or "").strip().startswith("http")
        else ""
        for r in node_records
    ]
    valid_idx = [i for i, t in enumerate(texts) if len(t) > 10]
    if len(valid_idx) < max(10, len(node_records) // 2):
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000, norm="l2")
        x = vec.fit_transform([texts[i] for i in valid_idx])
    except ValueError:
        return None
    if x.shape[1] < 5:
        return None
    sim = (x @ x.T).toarray()
    mems = np.asarray([membership[i] for i in valid_idx], dtype=int)
    n = len(valid_idx)
    iu, ju = np.triu_indices(n, k=1)
    same = mems[iu] == mems[ju]
    if not np.any(same) or np.all(same):
        return None
    intra = float(np.mean(sim[iu[same], ju[same]]))
    inter = float(np.mean(sim[iu[~same], ju[~same]]))
    return {
        "n_titled": n,
        "intra_cos": round(intra, 6),
        "inter_cos": round(inter, 6),
        "contrast": round(intra - inter, 6),
        "ratio": round(intra / max(inter, 1e-9), 6),
    }


def _fast_temporal_bipartite_rewire(
    edges: list[tuple[int, int]],
    paper_years: dict[int, int],
    ref_years: dict[int, int],
    swaps_per_edge: int = 10,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], float]:
    """Identical causal degree-preserving bipartite MCMC with pre-sampled uniform index pairs (15x faster)."""
    rng = np.random.default_rng(seed)
    edge_list = list(edges)
    edge_set = set(edge_list)
    m = len(edge_list)
    if m < 2:
        return edge_list, 0.0
    n_attempts = m * swaps_per_edge
    idx_i = rng.integers(0, m, size=n_attempts).tolist()
    idx_j = rng.integers(0, m, size=n_attempts).tolist()
    accepted = 0
    for i, j in zip(idx_i, idx_j):
        if i == j:
            continue
        p1, r1 = edge_list[i]
        p2, r2 = edge_list[j]
        if p1 == p2 or r1 == r2:
            continue
        if (p1, r2) in edge_set or (p2, r1) in edge_set:
            continue
        if ref_years[r2] > paper_years[p1] or ref_years[r1] > paper_years[p2]:
            continue
        edge_set.remove((p1, r1))
        edge_set.remove((p2, r2))
        edge_set.add((p1, r2))
        edge_set.add((p2, r1))
        edge_list[i] = (p1, r2)
        edge_list[j] = (p2, r1)
        accepted += 1
    return edge_list, accepted / n_attempts


def _bipartite_edge_retention(
    works: list[dict[str, Any]],
    ref_years_lookup: dict[str, int],
    network_type: str,
    core_top_n: int = 150,
    n_rewires: int = 100,
    seed: int = MASTER_SEED,
) -> dict[str, float]:
    """Measure exact bipartite edge retention |E_obs cap E_rw| / |E_obs| and density across S1 rewires."""
    if network_type == "bibliographic_coupling":
        if len(works) > core_top_n:
            ref_freq: Counter[str] = Counter()
            for w in works:
                for r in set(w.get("referenced_works") or []):
                    ref_freq[r] += 1
            scored = sorted(
                ((sum(ref_freq[r] - 1 for r in set(w.get("referenced_works") or []) if ref_freq[r] >= 2), str(w.get("id", "")), i)
                 for i, w in enumerate(works)),
                key=lambda x: (x[0], x[1]),
                reverse=True,
            )
            sub_works = [works[x[2]] for x in scored[:core_top_n]]
        else:
            sub_works = list(works)
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
    ref_earliest: dict[int, int] = {}
    bip_edges: list[tuple[int, int]] = []
    for p_idx, w in enumerate(sub_works):
        py = int(w.get("publication_year") or 2025)
        paper_years[p_idx] = py
        for r in set(w.get("referenced_works") or []):
            if r in ref_to_idx:
                r_idx = ref_to_idx[r]
                bip_edges.append((p_idx, r_idx))
                if r_idx not in ref_earliest or py < ref_earliest[r_idx]:
                    ref_earliest[r_idx] = py
    ref_years = {
        r_idx: (min(int(ref_years_lookup[r]), ref_earliest.get(r_idx, 2025)) if r in ref_years_lookup else ref_earliest.get(r_idx, 2025))
        for r, r_idx in ref_to_idx.items()
    }
    obs_set = set(bip_edges)
    m_bip = len(obs_set)
    n_p = len(sub_works)
    n_r = len(ref_to_idx)
    density = m_bip / max(1, n_p * n_r)

    rets: list[float] = []
    accs: list[float] = []
    for rep in range(n_rewires):
        rw, acc = _fast_temporal_bipartite_rewire(bip_edges, paper_years, ref_years, swaps_per_edge=10, seed=seed + rep)
        accs.append(acc)
        rets.append(len(obs_set & set(rw)) / max(1, m_bip))

    return {
        "n_papers_bip": n_p,
        "n_refs_bip": n_r,
        "m_bip_edges": m_bip,
        "bip_density": round(density, 5),
        "mean_edge_retention": round(float(np.mean(rets)), 4),
        "max_edge_retention": round(float(np.max(rets)), 4),
        "mean_swap_acceptance": round(float(np.mean(accs)), 4),
    }


def _eval_one(task: tuple[int, dict[str, Any], float, int, int, int]) -> tuple[int, dict[str, Any]]:
    idx, item, g_mid_saved, n_seeds, n_starts, n_rewires = task
    raw = json.loads((RAW_CORP_DIR / item["corpus_file"]).read_text(encoding="utf-8"))
    works = raw["works"] if isinstance(raw, dict) and "works" in raw else raw
    ref_years = raw.get("ref_years") or {} if isinstance(raw, dict) else {}
    net_type = item.get("network_type", "bibliographic_coupling")
    core_n = int(item.get("core_top_n") or 150)

    art = build_network_artifact(works, network_type=net_type, core_top_n=core_n, denom_scope="selected-core", frac_scheme="vosviewer")
    lcc_idx, f_sub, r_sub = extract_shared_lcc_edges(len(art.node_records), art.full_edges, art.frac_edges)
    lcc_records = [art.node_records[i] for i in lcc_idx]
    sf = edges_to_igraph(len(lcc_idx), f_sub, normalization="association_strength")
    sr = edges_to_igraph(len(lcc_idx), r_sub, normalization="association_strength")
    grid = np.geomspace(0.05, 5.0, 60)

    # --- A. Baseline min_cluster_size=5: fixed gamma=1.0 AND saved exact matched-K gamma* ---
    f_parts, f_qs = _run_seeds(sf, "CPM", 1.0, n_seeds, n_starts, True, 5, False)
    r_fixed, _ = _run_seeds(sr, "CPM", 1.0, n_seeds, n_starts, True, 5, False)

    k_full = int(round(float(np.median([_major_cluster_stats(m, 5)[0] for m in f_parts]))))
    k_frac_fixed = int(round(float(np.median([_major_cluster_stats(m, 5)[0] for m in r_fixed]))))

    g_mid = float(g_mid_saved)
    r_matched, r_qs = _run_seeds(sr, "CPM", g_mid, n_seeds, n_starts, True, 5, False)
    k_frac_matched = int(round(float(np.median([_major_cluster_stats(m, 5)[0] for m in r_matched]))))

    # Fixed-gamma = 1.0 metrics
    wf = _mean_within_ami(f_parts)
    wr_fix = _mean_within_ami(r_fixed)
    cx_fix = float(np.mean([ami(f_parts[i], r_fixed[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    delta_fixed = min(wf, wr_fix) - cx_fix
    reass_fixed = 1.0 - float(np.mean([hungarian_core_match(f_parts[i], r_fixed[j])[0] for i in range(n_seeds) for j in range(n_seeds)]))

    # Matched-K metrics & effective granularity G (min_cluster_size=5)
    wr_mat = _mean_within_ami(r_matched)
    cx_mat = float(np.mean([ami(f_parts[i], r_matched[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    delta_matched = min(wf, wr_mat) - cx_mat
    g_full_5 = float(np.median([_effective_granularity(m) for m in f_parts]))
    g_frac_5_mat = float(np.median([_effective_granularity(m) for m in r_matched]))
    g_frac_5_fix = float(np.median([_effective_granularity(m) for m in r_fixed]))
    g_ratio_5 = g_frac_5_mat / max(g_full_5, 1e-12)
    g_ratio_5_fix = g_frac_5_fix / max(g_full_5, 1e-12)

    # --- B. Unmerged min_cluster_size=1 effective granularity ---
    f_unm, _ = _run_seeds(sf, "CPM", 1.0, n_seeds, n_starts, False, 1, False)
    k_full_unm = int(round(float(np.median([_major_cluster_stats(m, 1)[0] for m in f_unm]))))
    _, g_mid_unm, _, _, _ = _find_plateau(sr, "CPM", max(1, k_full_unm), grid, 1, 0.55, 1.0, False, n_starts, False)
    r_unm, _ = _run_seeds(sr, "CPM", g_mid_unm, n_seeds, n_starts, False, 1, False)
    wf_u = _mean_within_ami(f_unm)
    wr_u = _mean_within_ami(r_unm)
    cx_u = float(np.mean([ami(f_unm[i], r_unm[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    delta_unm = min(wf_u, wr_u) - cx_u
    g_full_1 = float(np.median([_effective_granularity(m) for m in f_unm]))
    g_frac_1 = float(np.median([_effective_granularity(m) for m in r_unm]))
    g_ratio_1 = g_frac_1 / max(g_full_1, 1e-12)

    # --- C. External title TF-IDF cosine coherence on matched-K Medoids (BC corpora) ---
    coh_full = None
    coh_frac = None
    if net_type == "bibliographic_coupling":
        med_f = f_parts[select_medoid_partition(f_parts, f_qs)]
        med_r = r_matched[select_medoid_partition(r_matched, r_qs)]
        coh_full = _title_tfidf_coherence(med_f, lcc_records)
        coh_frac = _title_tfidf_coherence(med_r, lcc_records)

    # --- D. S1 bipartite edge retention & density ---
    s1_mix = _bipartite_edge_retention(works, ref_years, net_type, core_top_n=core_n, n_rewires=n_rewires, seed=MASTER_SEED)

    return idx, {
        "doi": item["doi"],
        "quartile": item["quartile"],
        "journal_key": item["journal_key"],
        "network_type": net_type,
        "n_lcc": len(lcc_idx),
        "proj_edge_density": round(2.0 * sf.ecount() / max(1, len(lcc_idx) * (len(lcc_idx) - 1)), 4),
        "k_full_g1": k_full,
        "k_frac_g1": k_frac_fixed,
        "delta_k_g1": k_frac_fixed - k_full,
        "k_frac_matched": k_frac_matched,
        "gamma_matched": round(float(g_mid), 4),
        "delta_count_fixed_g1": round(float(delta_fixed), 4),
        "reass_fixed_g1": round(float(reass_fixed), 4),
        "delta_count_matched": round(float(delta_matched), 4),
        "g_full_mcs5": round(g_full_5, 5),
        "g_frac_matched_mcs5": round(g_frac_5_mat, 5),
        "g_ratio_matched_mcs5": round(g_ratio_5, 4),
        "g_ratio_fixed_mcs5": round(g_ratio_5_fix, 4),
        "abs_log_g_ratio_mcs5": round(abs(math.log(max(g_ratio_5, 1e-12))), 4),
        "k_full_mcs1": k_full_unm,
        "delta_count_mcs1": round(float(delta_unm), 4),
        "g_ratio_matched_mcs1": round(g_ratio_1, 4),
        "abs_log_g_ratio_mcs1": round(abs(math.log(max(g_ratio_1, 1e-12))), 4),
        "title_coherence_full": coh_full,
        "title_coherence_frac": coh_frac,
        "s1_mixing": s1_mix,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute granularity, mixing, and coherence diagnostics across all 59 Layer-2 corpora.")
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
    prod = json.loads((RES_DIR / "layer2_counting_audit_production.json").read_text(encoding="utf-8"))
    s1rb = json.loads((RES_DIR / "layer2_s1_and_rb_diagnostics.json").read_text(encoding="utf-8"))
    v1b = json.loads((RES_DIR / "layer2_v1_expanded_benchmark.json").read_text(encoding="utf-8"))
    v1_gmid = {r["doi"].lower(): float(r["vos_as_v1"]["gamma_plateau_mid"]) for r in v1b["reviews"]}
    q4_w = {q: float(l0["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    tasks = [
        (i, item, v1_gmid[item["doi"].lower()], args.n_seeds, args.n_starts, args.n_rewires)
        for i, item in enumerate(manifest)
    ]
    # LPT order (largest corpus first) to eliminate tail-straggler idle cores
    tasks.sort(key=lambda t: (RAW_CORP_DIR / t[1]["corpus_file"]).stat().st_size, reverse=True)

    ordered: list[dict[str, Any] | None] = [None] * len(manifest)
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_eval_one, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            idx, rec = fut.result()
            ordered[idx] = rec
            done += 1
            if done % 10 == 0 or done == len(manifest):
                LOGGER.info("Progress: %d/%d corpora (%.1fs)", done, len(manifest), time.time() - t0)
    recs = [r for r in ordered if r is not None]

    jks = [r["journal_key"] for r in recs]
    qs = [r["quartile"] for r in recs]

    # 1. Fixed-gamma=1.0 summary
    dk_arr = np.array([r["delta_k_g1"] for r in recs], dtype=float)
    n_k_diff = int(np.sum(dk_arr != 0))
    n_k_frac_gt = int(np.sum(dk_arr > 0))
    n_k_frac_lt = int(np.sum(dk_arr < 0))
    d_fix_ht4 = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["delta_count_fixed_g1"] for r in recs]), jks, qs, q4_w, enforce_min_stratum_journals=True
    )
    r_fix_ht4 = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["reass_fixed_g1"] for r in recs]), jks, qs, q4_w, enforce_min_stratum_journals=True
    )
    # Paired diff: fixed_g1 - matched_K
    diff_fix_mat = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["delta_count_fixed_g1"] - r["delta_count_matched"] for r in recs]), jks, qs, q4_w, enforce_min_stratum_journals=True
    )

    # 2. Effective granularity summary (mcs=5 and mcs=1)
    gr5 = np.array([r["g_ratio_matched_mcs5"] for r in recs])
    gr1 = np.array([r["g_ratio_matched_mcs1"] for r in recs])
    sp_g5 = mod04.journal_clustered_spearman_test(
        np.array([r["abs_log_g_ratio_mcs5"] for r in recs]),
        np.array([r["delta_count_matched"] for r in recs]),
        jks,
        seed=MASTER_SEED + 301,
    )
    sp_g1 = mod04.journal_clustered_spearman_test(
        np.array([r["abs_log_g_ratio_mcs1"] for r in recs]),
        np.array([r["delta_count_mcs1"] for r in recs]),
        jks,
        seed=MASTER_SEED + 302,
    )
    g_tight_90 = [r for r in recs if 0.90 <= r["g_ratio_matched_mcs5"] <= 1.10]
    g_tight_90_ci = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["delta_count_matched"] for r in g_tight_90]),
        [r["journal_key"] for r in g_tight_90],
        seed=MASTER_SEED + 308,
    )
    g_tight_95 = [r for r in recs if 0.95 <= r["g_ratio_matched_mcs5"] <= 1.05]
    g_tight_95_ci = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["delta_count_matched"] for r in g_tight_95]),
        [r["journal_key"] for r in g_tight_95],
        seed=MASTER_SEED + 309,
    )

    # 3. S1 MCMC mixing & edge retention summary
    rets = np.array([r["s1_mixing"]["mean_edge_retention"] for r in recs])
    accs = np.array([r["s1_mixing"]["mean_swap_acceptance"] for r in recs])
    s1_by_doi = {r["doi"]: r for r in s1rb["reviews"]}
    ami_frac_arr = np.array([s1_by_doi[r["doi"]]["s1_frac"]["ami_null_mean"] for r in recs])
    ami_full_arr = np.array([s1_by_doi[r["doi"]]["s1_full"]["ami_null_mean"] for r in recs])
    top3_null = sorted(s1rb["reviews"], key=lambda x: x["s1_frac"]["q_null_vos_p95"], reverse=True)[:3]
    rec_by_doi = {r["doi"]: r for r in recs}
    top3_diag = []
    for t in top3_null:
        rd = rec_by_doi[t["doi"]]
        top3_diag.append({
            "doi": t["doi"],
            "quartile": t["quartile"],
            "n_lcc": rd["n_lcc"],
            "q_obs_vos": t["s1_frac"]["q_obs_vos"],
            "q_null_vos_p95": t["s1_frac"]["q_null_vos_p95"],
            "ami_null_mean": t["s1_frac"]["ami_null_mean"],
            "proj_edge_density": rd["proj_edge_density"],
            **rd["s1_mixing"],
        })

    # 4. External title TF-IDF cosine coherence across 31 BC corpora
    bc_coh = [r for r in recs if r["network_type"] == "bibliographic_coupling" and r["title_coherence_full"] and r["title_coherence_frac"]]
    coh_diffs = np.array([r["title_coherence_frac"]["contrast"] - r["title_coherence_full"]["contrast"] for r in bc_coh], dtype=float)
    coh_full_arr = np.array([r["title_coherence_full"]["contrast"] for r in bc_coh], dtype=float)
    coh_frac_arr = np.array([r["title_coherence_frac"]["contrast"] for r in bc_coh], dtype=float)
    coh_jks = [r["journal_key"] for r in bc_coh]
    coh_ci = mod04.stratified_journal_cluster_bootstrap_ci(coh_diffs, coh_jks, seed=MASTER_SEED + 303)
    # Centered journal-clustered bootstrap p-value for paired mean difference
    j_map: dict[str, list[int]] = defaultdict(list)
    for i, jk in enumerate(coh_jks):
        j_map[jk].append(i)
    j_keys = sorted(j_map.keys())
    rng = np.random.default_rng(MASTER_SEED + 304)
    b_means = np.empty(2000, dtype=float)
    for b in range(2000):
        drawn = rng.integers(0, len(j_keys), size=len(j_keys))
        idxs = [i for s in drawn for i in j_map[j_keys[int(s)]]]
        b_means[b] = float(np.mean(coh_diffs[idxs]))
    obs_diff = float(np.mean(coh_diffs))
    coh_p_boot = float((1 + np.sum(np.abs(b_means - obs_diff) >= abs(obs_diff))) / 2001)
    wilc = stats.wilcoxon(coh_frac_arr, coh_full_arr, zero_method="wilcox", alternative="two-sided")

    # 5. Stability-Modularity paired trade-off (from production & s1rb JSONs)
    prod_revs = prod["reviews"]
    p_jks = [r["journal_key"] for r in prod_revs]
    p_qs = [r["quartile"] for r in prod_revs]
    within_diff = np.array([r["vos_core"]["ami_within_frac"] - r["vos_core"]["ami_within_full"] for r in prod_revs], dtype=float)
    within_diff_ht4 = mod04.stratified_journal_cluster_bootstrap_ci(within_diff, p_jks, p_qs, q4_w, enforce_min_stratum_journals=True)
    mod_diff = np.array([s1_by_doi[r["doi"]]["s1_frac"]["q_obs_rb"] - s1_by_doi[r["doi"]]["s1_full"]["q_obs_rb"] for r in prod_revs], dtype=float)
    mod_diff_unw = mod04.stratified_journal_cluster_bootstrap_ci(mod_diff, p_jks, seed=MASTER_SEED + 305)
    mod_wilc = stats.wilcoxon(mod_diff, alternative="two-sided")
    within_wilc = stats.wilcoxon(within_diff, alternative="two-sided")

    # 6. Out-of-band n=19 vs in-band n=40 comparison
    inb = [r for r in prod_revs if r.get("ratio_repro_orig") is not None and 0.50 <= float(r["ratio_repro_orig"]) <= 2.00]
    outb = [r for r in prod_revs if r.get("ratio_repro_orig") is None or float(r["ratio_repro_orig"]) < 0.50 or float(r["ratio_repro_orig"]) > 2.00]
    outb_ci = mod04.stratified_journal_cluster_bootstrap_ci(
        np.array([r["vos_core"]["delta_count_mid"] for r in outb]), [r["journal_key"] for r in outb], seed=MASTER_SEED + 306
    )
    in_vs_out = mod04.journal_clustered_group_diff_test(
        np.array([r["vos_core"]["delta_count_mid"] for r in outb]), [r["journal_key"] for r in outb],
        np.array([r["vos_core"]["delta_count_mid"] for r in inb]), [r["journal_key"] for r in inb],
        seed=MASTER_SEED + 307,
    )

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "n_corpora": len(recs),
        "fixed_gamma_1_benchmark": {
            "n_k_diff": n_k_diff,
            "pct_k_diff": round(100.0 * n_k_diff / len(recs), 1),
            "n_k_frac_gt_full": n_k_frac_gt,
            "n_k_frac_lt_full": n_k_frac_lt,
            "delta_k_median": float(np.median(dk_arr)),
            "delta_k_iqr": [float(np.percentile(dk_arr, 25)), float(np.percentile(dk_arr, 75))],
            "delta_k_range": [int(np.min(dk_arr)), int(np.max(dk_arr))],
            "abs_delta_k_mean": round(float(np.mean(np.abs(dk_arr))), 3),
            "ht4_delta_count_fixed_g1": {"mean": d_fix_ht4[0], "ci95": [d_fix_ht4[1], d_fix_ht4[2]]},
            "ht4_reassignment_fixed_g1": {"mean": r_fix_ht4[0], "ci95": [r_fix_ht4[1], r_fix_ht4[2]]},
            "ht4_diff_fixed_minus_matched": {"mean": diff_fix_mat[0], "ci95": [diff_fix_mat[1], diff_fix_mat[2]]},
        },
        "effective_granularity": {
            "mcs5_g_ratio_median": round(float(np.median(gr5)), 3),
            "mcs5_g_ratio_iqr": [round(float(np.percentile(gr5, 25)), 3), round(float(np.percentile(gr5, 75)), 3)],
            "mcs5_g_ratio_range": [round(float(np.min(gr5)), 3), round(float(np.max(gr5)), 3)],
            "mcs5_spearman_abs_log_g_vs_delta": sp_g5,
            "mcs1_g_ratio_median": round(float(np.median(gr1)), 3),
            "mcs1_g_ratio_iqr": [round(float(np.percentile(gr1, 25)), 3), round(float(np.percentile(gr1, 75)), 3)],
            "mcs1_g_ratio_range": [round(float(np.min(gr1)), 3), round(float(np.max(gr1)), 3)],
            "mcs1_spearman_abs_log_g_vs_delta": sp_g1,
            "tight_g_90_110": {
                "n": len(g_tight_90),
                "n_journals": len({r["journal_key"] for r in g_tight_90}),
                "mean": g_tight_90_ci[0],
                "ci95": [g_tight_90_ci[1], g_tight_90_ci[2]],
            },
            "tight_g_95_105": {
                "n": len(g_tight_95),
                "n_journals": len({r["journal_key"] for r in g_tight_95}),
                "mean": g_tight_95_ci[0],
                "ci95": [g_tight_95_ci[1], g_tight_95_ci[2]],
            },
        },
        "s1_mcmc_mixing": {
            "swap_accept_min": round(float(np.min(accs)), 4),
            "swap_accept_iqr": [round(float(np.percentile(accs, 25)), 4), round(float(np.percentile(accs, 75)), 4)],
            "swap_accept_median": round(float(np.median(accs)), 4),
            "swap_accept_max": round(float(np.max(accs)), 4),
            "edge_retention_mean": round(float(np.mean(rets)), 4),
            "edge_retention_median": round(float(np.median(rets)), 4),
            "edge_retention_iqr": [round(float(np.percentile(rets, 25)), 4), round(float(np.percentile(rets, 75)), 4)],
            "edge_retention_max": round(float(np.max(rets)), 4),
            "ami_obs_vs_null_frac_max": round(float(np.max(ami_frac_arr)), 4),
            "ami_obs_vs_null_full_max": round(float(np.max(ami_full_arr)), 4),
            "top3_high_null_vos_corpora": top3_diag,
        },
        "bc_title_tfidf_coherence": {
            "n_bc_evaluated": len(bc_coh),
            "n_journals": len(set(coh_jks)),
            "mean_contrast_full": round(float(np.mean(coh_full_arr)), 4),
            "mean_contrast_frac": round(float(np.mean(coh_frac_arr)), 4),
            "mean_intra_full": round(float(np.mean([r["title_coherence_full"]["intra_cos"] for r in bc_coh])), 4),
            "mean_intra_frac": round(float(np.mean([r["title_coherence_frac"]["intra_cos"] for r in bc_coh])), 4),
            "mean_inter_full": round(float(np.mean([r["title_coherence_full"]["inter_cos"] for r in bc_coh])), 4),
            "mean_inter_frac": round(float(np.mean([r["title_coherence_frac"]["inter_cos"] for r in bc_coh])), 4),
            "diff_frac_minus_full": {"mean": round(obs_diff, 4), "ci95": [coh_ci[1], coh_ci[2]], "cluster_boot_p": round(coh_p_boot, 4)},
            "wilcoxon_p": round(float(wilc.pvalue), 4),
            "n_frac_higher": int(np.sum(coh_diffs > 0)),
            "n_full_higher": int(np.sum(coh_diffs < 0)),
            "n_ties": int(np.sum(coh_diffs == 0)),
        },
        "stability_modularity_tradeoff": {
            "within_ami_frac_minus_full_ht4": {"mean": within_diff_ht4[0], "ci95": [within_diff_ht4[1], within_diff_ht4[2]], "wilcoxon_p": round(float(within_wilc.pvalue), 4)},
            "leiden_modularity_frac_minus_full": {"mean": mod_diff_unw[0], "ci95": [mod_diff_unw[1], mod_diff_unw[2]], "wilcoxon_p": round(float(mod_wilc.pvalue), 6), "n_frac_higher": int(np.sum(mod_diff > 0))},
        },
        "out_of_band_ratio_sensitivity": {
            "n_outband": len(outb),
            "n_journals_outband": len({r["journal_key"] for r in outb}),
            "outband_mean_ci95": {"mean": outb_ci[0], "ci95": [outb_ci[1], outb_ci[2]]},
            "outband_minus_inband": in_vs_out,
        },
        "reviews": recs,
    }
    out_p = RES_DIR / "layer2_granularity_and_coherence_diagnostics.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Saved granularity/coherence diagnostics in %.1fs -> %s", out["elapsed_sec"], out_p)


if __name__ == "__main__":
    main()
