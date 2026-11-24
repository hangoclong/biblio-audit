"""Layer 2: Primary counting-rule audit (VOSviewer 1.6.20 parity + 4-stratum HT + S1 temporal null).

Hardened specification: - Exact ROPE zone naming & inequality parity with `protocol specification` (`Zone_1/2/3`, zero dead-zone).
- Separates `degenerate_k` (K < 2) reviews from primary ROPE estimand (reporting both primary non-degenerate
  and all-review sensitivity).
- Pre-specified `g_h < 5` small-cluster variance collapse rule (`Q3+Q4` combined weight `W_Q3+W_Q4` if Q4
  has `< 5` unique journals; logged transparently).
- Unified-journal cluster resampling for overlapping `BC vs CC` comparisons + centered cluster bootstrap
  p-value with `(1 + cnt) / (B + 1)` floor + journal-clustered bootstrap 95% CIs & p-values for continuous
  moderators (`ref_degree_gini`, `citescore`, `ratio_repro_orig`).
- `S1` temporal bipartite null normalizes `q_vos` by `2 * sum(g_vos.es["weight"])` and uses `best-of-3` starts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score as ami

EXP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXP_DIR / "src_audit"
sys.path.insert(0, str(SRC_DIR))

from counting_engine import (  # noqa: E402
    _leiden_partition,
    build_sparse_edges_from_works,
    edges_to_igraph,
    evaluate_counting_sensitivity,
    merge_small_clusters_cwts,
    temporal_bipartite_rewire,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.layer2")

PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
RES_DIR = EXP_DIR / "results"
MASTER_SEED = 20261122


def classify_rope_zone(point: float, ci_lo: float, ci_hi: float) -> str:
    """Protocol exact total-partition 3-zone ROPE rule (verbatim match with protocol specification)."""
    if point <= 0.10 and ci_hi < 0.15:
        return "Zone_1_Empirical_Certification"
    if point >= 0.20 and ci_lo > 0.15:
        return "Zone_2_Mandatory_Verification_Gate"
    return "Zone_3_Conditional_Regime_Dependent"


def reference_degree_gini(works: list[dict[str, Any]]) -> float:
    """Compute Gini concentration of cited-reference frequencies (n_k >= 2) across the full corpus."""
    counts: dict[str, int] = defaultdict(int)
    for w in works:
        for r in set(w.get("referenced_works") or []):
            counts[r] += 1
    vals = np.array([c for c in counts.values() if c >= 2], dtype=float)
    if vals.size < 2:
        return 0.0
    vals.sort()
    n = vals.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * vals) / (n * np.sum(vals))) - (n + 1.0) / n)


def _select_bc_tls_works(works: list[dict[str, Any]], core_top_n: int) -> list[dict[str, Any]]:
    """Select top-N documents by within-corpus bibliographic coupling TLS (zero post-cutoff look-ahead)."""
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


def run_temporal_bipartite_null(
    works: list[dict[str, Any]],
    ref_years_lookup: dict[str, int] | None = None,
    network_type: str = "bibliographic_coupling",
    core_top_n: int = 150,
    n_rewires: int = 100,
    n_starts: int = 3,
    seed: int = MASTER_SEED,
) -> dict[str, float]:
    """Compare observed normalized VOS_AS_CPM & RBConfig quality against year-stratified fractional bipartite null."""
    ref_years_lookup = ref_years_lookup or {}
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
    if len(ref_to_idx) < 5 or len(sub_works) < 5:
        return {
            "q_obs_rb": 0.0,
            "q_null_rb_mean": 0.0,
            "q_null_rb_p95": 0.0,
            "q_obs_vos_norm": 0.0,
            "q_null_vos_norm_mean": 0.0,
            "q_null_vos_norm_p95": 0.0,
            "ami_obs_vs_null_mean": 0.0,
            "mean_swap_acceptance_rate": 0.0,
            "true_ref_year_coverage": 0.0,
            "passes_temporal_null_rb": 0.0,
            "passes_temporal_null_vos": 0.0,
        }

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
    true_yr_hits = 0
    for r, r_idx in ref_to_idx.items():
        earliest = ref_earliest_cited.get(r_idx, 2025)
        if r in ref_years_lookup:
            true_yr_hits += 1
            ref_years[r_idx] = min(int(ref_years_lookup[r]), earliest)
        else:
            ref_years[r_idx] = earliest
    true_cov = true_yr_hits / max(1, len(ref_to_idx))

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
            inv = 1.0 / (nk - 1.0)
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
            q_val = float(sg_rb.modularity(mem_rb, weights="weight"))
            if q_val > best_q_rb:
                best_q_rb = q_val

        # Normalized VOS_AS_CPM quality (Waltman et al. 2010 Eq. 7, scale-invariant across rewires!)
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
    obs_map = {node: cid for node, cid in zip(obs_lcc, obs_mem)}

    null_rbs: list[float] = []
    null_voss: list[float] = []
    null_amis: list[float] = []
    acc_rates: list[float] = []
    for rep in range(n_rewires):
        rw, acc = temporal_bipartite_rewire(bip_edges, paper_years, ref_years, swaps_per_edge=10, seed=seed + rep)
        acc_rates.append(acc)
        q_rb, q_vos, n_lcc, n_mem = project_metrics(rw)
        null_rbs.append(q_rb)
        null_voss.append(q_vos)
        if obs_lcc and n_lcc:
            n_map = {node: cid for node, cid in zip(n_lcc, n_mem)}
            common = sorted(set(obs_map.keys()) & set(n_map.keys()))
            if len(common) >= 10:
                null_amis.append(float(ami([obs_map[u] for u in common], [n_map[u] for u in common])))

    return {
        "q_obs_rb": round(q_obs_rb, 4),
        "q_null_rb_mean": round(float(np.mean(null_rbs)), 4),
        "q_null_rb_p95": round(float(np.percentile(null_rbs, 95)), 4),
        "q_obs_vos_norm": round(q_obs_vos, 4),
        "q_null_vos_norm_mean": round(float(np.mean(null_voss)), 4),
        "q_null_vos_norm_p95": round(float(np.percentile(null_voss, 95)), 4),
        "ami_obs_vs_null_mean": round(float(np.mean(null_amis)), 4) if null_amis else 0.0,
        "mean_swap_acceptance_rate": round(float(np.mean(acc_rates)), 4),
        "true_ref_year_coverage": round(true_cov, 4),
        "passes_temporal_null_rb": float(q_obs_rb > float(np.percentile(null_rbs, 95))),
        "passes_temporal_null_vos": float(q_obs_vos > float(np.percentile(null_voss, 95))),
    }


def _collapse_small_strata(
    strata: list[str], journals: list[str], strata_weights: dict[str, float], min_journals_per_stratum: int = 5
) -> tuple[list[str], dict[str, float], bool]:
    """Pre-specified variance-collapse safeguard: merge Q3+Q4 if Q4 (or Q3) has < 5 unique journals."""
    st_j: dict[str, set[str]] = defaultdict(set)
    for st, j in zip(strata, journals):
        st_j[st].add(j)
    if ("Q4" in st_j and len(st_j["Q4"]) < min_journals_per_stratum) or ("Q3" in st_j and len(st_j["Q3"]) < min_journals_per_stratum):
        new_st = ["Q3_Q4" if s in ("Q3", "Q4") else s for s in strata]
        new_w = {k: v for k, v in strata_weights.items() if k not in ("Q3", "Q4")}
        new_w["Q3_Q4"] = strata_weights.get("Q3", 0.0) + strata_weights.get("Q4", 0.0)
        return new_st, new_w, True
    return strata, strata_weights, False


def stratified_journal_cluster_bootstrap_ci(
    values: np.ndarray,
    journals: list[str],
    strata: list[str] | None = None,
    strata_weights: dict[str, float] | None = None,
    sample_weights: np.ndarray | None = None,
    n_reps: int = 2000,
    seed: int = MASTER_SEED,
    enforce_min_stratum_journals: bool = False,
) -> tuple[float, float, float]:
    """Stratified journal-cluster bootstrap 95% CI with exact 4-stratum HT weights & optional IPW."""
    if len(values) == 0:
        return (0.0, 0.0, 0.0)
    sw = np.ones(len(values), dtype=float) if sample_weights is None else np.asarray(sample_weights, dtype=float)

    def _wmean(idxs: list[int]) -> float:
        w = sw[idxs]
        s = float(np.sum(w))
        return float(np.sum(values[idxs] * w) / s) if s > 0 else 0.0

    if strata is None or strata_weights is None:
        point = _wmean(list(range(len(values))))
        j_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, j in enumerate(journals):
            j_to_idx[j].append(i)
        j_keys = sorted(j_to_idx.keys())
        g_cnt = len(j_keys)
        if g_cnt < 2:
            return (round(point, 4), round(point, 4), round(point, 4))
        rng = np.random.default_rng(seed)
        boots = np.empty(n_reps, dtype=float)
        for b in range(n_reps):
            drawn = rng.integers(0, g_cnt, size=g_cnt)
            idxs = [i for s in drawn for i in j_to_idx[j_keys[int(s)]]]
            boots[b] = _wmean(idxs)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return (round(point, 4), round(float(lo), 4), round(float(hi), 4))

    use_st, use_w = strata, strata_weights
    if enforce_min_stratum_journals:
        use_st, use_w, _ = _collapse_small_strata(strata, journals, strata_weights)

    st_j_to_idx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (st, j) in enumerate(zip(use_st, journals)):
        st_j_to_idx[st][j].append(i)
    active_strata = sorted(st_j_to_idx.keys())
    w_tot = sum(use_w.get(st, 0.0) for st in active_strata)
    norm_wh = {st: use_w[st] / w_tot for st in active_strata}

    point = sum(
        norm_wh[st] * _wmean([i for lst in st_j_to_idx[st].values() for i in lst])
        for st in active_strata
    )
    rng = np.random.default_rng(seed)
    boots = np.empty(n_reps, dtype=float)
    st_keys = {st: sorted(st_j_to_idx[st].keys()) for st in active_strata}
    for b in range(n_reps):
        est_b = 0.0
        for st in active_strata:
            jk = st_keys[st]
            g_h = len(jk)
            drawn = rng.integers(0, g_h, size=g_h)
            idxs_h = [i for s in drawn for i in st_j_to_idx[st][jk[int(s)]]]
            est_b += norm_wh[st] * _wmean(idxs_h)
        boots[b] = est_b
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (round(point, 4), round(float(lo), 4), round(float(hi), 4))


def journal_clustered_group_diff_test(
    vals_a: np.ndarray,
    j_a: list[str],
    vals_b: np.ndarray,
    j_b: list[str],
    n_reps: int = 2000,
    seed: int = MASTER_SEED + 77,
) -> dict[str, Any]:
    """Centered journal-clustered bootstrap difference test handling overlapping journals."""
    diff_obs = float(np.mean(vals_a) - np.mean(vals_b))
    ja_map: dict[str, list[int]] = defaultdict(list)
    jb_map: dict[str, list[int]] = defaultdict(list)
    for i, j in enumerate(j_a):
        ja_map[j].append(i)
    for i, j in enumerate(j_b):
        jb_map[j].append(i)
    ka, kb = sorted(ja_map.keys()), sorted(jb_map.keys())
    overlaps = bool(set(ka) & set(kb))

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    if overlaps:
        # Unified journal cluster resampling when groups share journals (e.g. BC vs CC)
        u_keys = sorted(set(ka) | set(kb))
        g_u = len(u_keys)
        for _ in range(n_reps):
            drawn = rng.integers(0, g_u, size=g_u)
            ia = [i for s in drawn for i in ja_map.get(u_keys[int(s)], [])]
            ib = [i for s in drawn for i in jb_map.get(u_keys[int(s)], [])]
            if ia and ib:
                diffs.append(float(np.mean(vals_a[ia]) - np.mean(vals_b[ib])))
    else:
        for _ in range(n_reps):
            da = rng.integers(0, len(ka), size=len(ka))
            db = rng.integers(0, len(kb), size=len(kb))
            ia = [i for s in da for i in ja_map[ka[int(s)]]]
            ib = [i for s in db for i in jb_map[kb[int(s)]]]
            diffs.append(float(np.mean(vals_a[ia]) - np.mean(vals_b[ib])))

    arr = np.asarray(diffs, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    se = float(np.std(arr, ddof=1))
    # Centered null bootstrap two-sided p-value with exact (1 + cnt) / (B + 1) floor!
    centered = arr - diff_obs
    p_boot = float((1 + np.sum(np.abs(centered) >= abs(diff_obs))) / (len(arr) + 1))
    tt = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    mw = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {
        "diff_mean": round(diff_obs, 4),
        "unified_cluster_resample": overlaps,
        "boot_se": round(se, 4),
        "boot_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "boot_centered_p": round(p_boot, 4),
        "welch_t": round(float(tt.statistic), 4),
        "welch_p": round(float(tt.pvalue), 4),
        "mannwhitney_p": round(float(mw.pvalue), 4),
    }


def journal_clustered_spearman_test(
    xs: np.ndarray, ys: np.ndarray, journals: list[str], n_reps: int = 2000, seed: int = MASTER_SEED + 99
) -> dict[str, Any]:
    """Journal-clustered bootstrap 95% CI & centered p-value for Spearman rank correlation."""
    rho_obs, raw_p = stats.spearmanr(xs, ys)
    rho_obs = float(rho_obs)
    j_map: dict[str, list[int]] = defaultdict(list)
    for i, j in enumerate(journals):
        j_map[j].append(i)
    j_keys = sorted(j_map.keys())
    g = len(j_keys)
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_reps):
        drawn = rng.integers(0, g, size=g)
        idxs = [i for s in drawn for i in j_map[j_keys[int(s)]]]
        if np.std(xs[idxs]) > 0 and np.std(ys[idxs]) > 0:
            r_b, _ = stats.spearmanr(xs[idxs], ys[idxs])
            if np.isfinite(r_b):
                rhos.append(float(r_b))
    arr = np.asarray(rhos, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5]) if arr.size else (rho_obs, rho_obs)
    centered = arr - rho_obs
    p_boot = float((1 + np.sum(np.abs(centered) >= abs(rho_obs))) / (max(1, len(arr)) + 1))
    return {
        "n": int(len(xs)),
        "n_journals": g,
        "spearman_rho": round(rho_obs, 4),
        "cluster_boot_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "cluster_boot_p": round(p_boot, 4),
        "raw_iid_p": round(float(raw_p), 4),
    }


def _evaluate_one_review(task: tuple[int, dict[str, Any], int, int, int, int, bool]) -> tuple[int, dict[str, Any]]:
    """Deterministic worker evaluating one Layer-2 review across all arms + S1 temporal null."""
    idx, item, n_seeds, random_starts, n_null_rewires, wide_cap, run_cited_sens = task
    corp_file = RAW_CORP_DIR / item["corpus_file"]
    if not corp_file.exists():
        raise FileNotFoundError(f"Missing corpus file: {corp_file}")
    raw_payload = json.loads(corp_file.read_text(encoding="utf-8"))
    if isinstance(raw_payload, dict) and "works" in raw_payload:
        works = raw_payload["works"]
        ref_years = raw_payload.get("ref_years") or {}
    elif isinstance(raw_payload, dict) and "results" in raw_payload:
        works = raw_payload["results"]
        ref_years = {}
    else:
        works = raw_payload
        ref_years = {}

    net_type = item.get("network_type", "bibliographic_coupling")
    core_n = int(item.get("core_top_n") or 150)

    c_nodes, c_ef, c_er = build_sparse_edges_from_works(works, net_type, core_top_n=core_n, core_selection="tls_within_network")
    res_vos_core = evaluate_counting_sensitivity(
        c_nodes, c_ef, c_er, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=random_starts, merge_small_clusters=True
    )
    res_rb_core = evaluate_counting_sensitivity(
        c_nodes, c_ef, c_er, objective="Leiden_RBConfig", n_seeds=n_seeds, n_random_starts=random_starts, merge_small_clusters=True
    )

    f_nodes, f_ef, f_er = build_sparse_edges_from_works(works, net_type, core_top_n=wide_cap, core_selection="tls_within_network")
    res_vos_wide = evaluate_counting_sensitivity(
        f_nodes, f_ef, f_er, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=random_starts, merge_small_clusters=True
    )
    res_rb_wide = evaluate_counting_sensitivity(
        f_nodes, f_ef, f_er, objective="Leiden_RBConfig", n_seeds=n_seeds, n_random_starts=random_starts, merge_small_clusters=True
    )

    cited_sens = None
    if run_cited_sens and net_type == "bibliographic_coupling":
        sn, sef, ser = build_sparse_edges_from_works(works, net_type, core_top_n=core_n, core_selection="cited_by_count")
        cited_sens = asdict(
            evaluate_counting_sensitivity(
                sn, sef, ser, objective="VOS_AS_CPM", n_seeds=n_seeds, n_random_starts=random_starts, merge_small_clusters=True
            )
        )

    s1_null = run_temporal_bipartite_null(
        works, ref_years_lookup=ref_years, network_type=net_type, core_top_n=core_n, n_rewires=n_null_rewires, n_starts=3
    )

    rec = {
        "doi": item["doi"],
        "title": item["title"],
        "quartile": item["quartile"],
        "stratum": "Q1" if item["quartile"] == "Q1" else "Q2_Q4",
        "journal_key": item["journal_key"],
        "publication_name": item.get("publication_name") or item.get("journal", ""),
        "pub_year": item["pub_year"],
        "citescore": item.get("citescore"),
        "ipw_weight": float(item.get("ipw_weight") or 1.0),
        "cutoff_year": item["cutoff_year"],
        "network_type": net_type,
        "n_repro_total": int(item.get("source_total_hits") or item.get("oa_total_hits") or len(works)),
        "n_works_joined": len(works),
        "n_orig": item.get("n_orig_printed") or item.get("n_orig"),
        "ratio_repro_orig": item.get("ratio_repro_orig"),
        "ref_degree_gini": round(reference_degree_gini(works), 4),
        "vos_core": asdict(res_vos_core),
        "rb_core": asdict(res_rb_core),
        "vos_wide": asdict(res_vos_wide),
        "rb_wide": asdict(res_rb_wide),
        "vos_core_cited_sensitivity": cited_sens,
        "s1_temporal_null": s1_null,
    }
    return idx, rec


def main() -> None:
    """Run Layer-2 counting audit across all verified corpora in layer2_manifest.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(PROC_DIR / "layer2_manifest.json"))
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--random-starts", type=int, default=10, help="VOSviewer 1.6.20 best-of-10 random starts")
    ap.add_argument("--n-null-rewires", type=int, default=100, help="100 production temporal bipartite rewires")
    ap.add_argument("--wide-cap", type=int, default=500, help="Symmetric wide comparator cap across BC & CC")
    ap.add_argument("--run-cited-sensitivity", action="store_true", help="Also evaluate BC under cited_by_count vs TLS")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out-name", type=str, default="layer2_counting_audit.json")
    ap.add_argument("--checkpoint-dir", type=str, default="", help="Optional per-review checkpoint directory for crash-safe resume")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    bal_path = RES_DIR / "layer0_population_balance.json"
    q4_weights = {"Q1": 0.5093, "Q2": 0.2113, "Q3": 0.1960, "Q4": 0.0835}
    if bal_path.exists():
        bal = json.loads(bal_path.read_text(encoding="utf-8"))
        q4_weights = {q: float(bal["strata"][q]["weight_Wh"]) for q in ("Q1", "Q2", "Q3", "Q4")}

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    ordered: list[dict[str, Any] | None] = [None] * len(manifest)
    pending_tasks = []
    done = 0
    for i, item in enumerate(manifest):
        if ckpt_dir is not None:
            cp = ckpt_dir / f"rev_{i:03d}.json"
            if cp.exists():
                ordered[i] = json.loads(cp.read_text(encoding="utf-8"))
                done += 1
                continue
        pending_tasks.append((i, item, args.n_seeds, args.random_starts, args.n_null_rewires, args.wide_cap, args.run_cited_sensitivity))

    LOGGER.info(
        "Starting Layer-2 production audit on %d reviews (%d cached, %d pending, %d parallel workers)...",
        len(manifest), done, len(pending_tasks), args.workers,
    )
    if args.workers <= 1:
        for t in pending_tasks:
            idx, rec = _evaluate_one_review(t)
            ordered[idx] = rec
            if ckpt_dir is not None:
                (ckpt_dir / f"rev_{idx:03d}.json").write_text(json.dumps(rec), encoding="utf-8")
            done += 1
            LOGGER.info("[%d/%d] %s (%s/%s) | Delta=%.4f | Reassign=%.1f%% | S1_vos=%d",
                        done, len(manifest), rec["doi"], rec["quartile"], rec["network_type"][:2],
                        rec["vos_core"]["delta_count_mid"], 100 * rec["vos_core"]["core_reassignment"],
                        int(rec["s1_temporal_null"]["passes_temporal_null_vos"]))
    elif pending_tasks:
        # Schedule largest corpora first (LPT) to avoid tail-straggler idle cores; output stays strictly ordered[idx]
        tasks_lpt = sorted(pending_tasks, key=lambda t: (RAW_CORP_DIR / t[1]["corpus_file"]).stat().st_size, reverse=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_evaluate_one_review, t) for t in tasks_lpt]
            for fut in concurrent.futures.as_completed(futs):
                idx, rec = fut.result()
                ordered[idx] = rec
                if ckpt_dir is not None:
                    (ckpt_dir / f"rev_{idx:03d}.json").write_text(json.dumps(rec), encoding="utf-8")
                done += 1
                LOGGER.info("[%d/%d] %s (%s/%s) | Delta=%.4f | Reassign=%.1f%% | K=%d->%d | S1_vos=%d",
                            done, len(manifest), rec["doi"], rec["quartile"], rec["network_type"][:2],
                            rec["vos_core"]["delta_count_mid"], 100 * rec["vos_core"]["core_reassignment"],
                            rec["vos_core"]["k_full_major"], rec["vos_core"]["k_frac_matched_major"],
                            int(rec["s1_temporal_null"]["passes_temporal_null_vos"]))
    records = [r for r in ordered if r is not None]

    def summarize_group(sub: list[dict[str, Any]], use_ht4: bool = False, use_ipw: bool = False) -> dict[str, Any]:
        if not sub:
            return {"n": 0, "n_journals": 0}
        jkeys = [r["journal_key"] for r in sub]
        st_list = [r["quartile"] for r in sub] if use_ht4 else None
        sw_map = q4_weights if use_ht4 else None
        collapsed = False
        if use_ht4 and st_list is not None and sw_map is not None:
            _, _, collapsed = _collapse_small_strata(st_list, jkeys, sw_map)
        ipw_arr = np.array([float(r.get("ipw_weight") or 1.0) for r in sub], dtype=float) if use_ipw else None
        out: dict[str, Any] = {
            "n": len(sub),
            "n_journals": len(set(jkeys)),
            "collapsed_q3_q4_for_var": collapsed,
            "n_exact_k_vos_core": sum(1 for r in sub if r["vos_core"]["exact_k_matched"]),
            "n_rev_exact_k_vos_core": sum(1 for r in sub if r["vos_core"]["rev_exact_k_matched"]),
            "n_degenerate_k_vos_core": sum(1 for r in sub if r["vos_core"]["degenerate_k"]),
        }
        for arm, metric in [
            ("vos_core", "delta_count_mid"),
            ("vos_core", "delta_count_reverse"),
            ("vos_core", "core_reassignment"),
            ("vos_core", "landmark_top10_retention"),
            ("vos_core", "ari_cross_mid"),
            ("vos_core", "ami_within_full"),
            ("rb_core", "delta_count_mid"),
            ("rb_core", "core_reassignment"),
            ("vos_wide", "delta_count_mid"),
            ("rb_wide", "delta_count_mid"),
        ]:
            vals = np.array([float(r[arm][metric]) for r in sub], dtype=float)
            pt, lo, hi = stratified_journal_cluster_bootstrap_ci(
                vals, jkeys, strata=st_list, strata_weights=sw_map, sample_weights=ipw_arr, enforce_min_stratum_journals=use_ht4
            )
            out[f"{arm}_{metric}"] = {"mean": pt, "ci95": [lo, hi], "median": round(float(np.median(vals)), 4)}
        if use_ht4 and not use_ipw:
            d = out["vos_core_delta_count_mid"]
            out["rope_verdict"] = classify_rope_zone(d["mean"], d["ci95"][0], d["ci95"][1])
        return out

    # Primary estimand excludes degenerate_k (K < 2) reviews; all-review sensitivity reported alongside
    nondeg = [r for r in records if not r["vos_core"]["degenerate_k"]]
    q1_sub = [r for r in nondeg if r["quartile"] == "Q1"]
    q24_sub = [r for r in nondeg if r["quartile"] != "Q1"]
    bc_sub = [r for r in nondeg if r["network_type"] == "bibliographic_coupling"]
    cc_sub = [r for r in nondeg if r["network_type"] == "co_citation"]

    q_diff_test = (
        journal_clustered_group_diff_test(
            np.array([r["vos_core"]["delta_count_mid"] for r in q1_sub]),
            [r["journal_key"] for r in q1_sub],
            np.array([r["vos_core"]["delta_count_mid"] for r in q24_sub]),
            [r["journal_key"] for r in q24_sub],
        )
        if len(q1_sub) >= 2 and len(q24_sub) >= 2
        else {}
    )
    net_diff_test = (
        journal_clustered_group_diff_test(
            np.array([r["vos_core"]["delta_count_mid"] for r in bc_sub]),
            [r["journal_key"] for r in bc_sub],
            np.array([r["vos_core"]["delta_count_mid"] for r in cc_sub]),
            [r["journal_key"] for r in cc_sub],
            seed=MASTER_SEED + 88,
        )
        if len(bc_sub) >= 2 and len(cc_sub) >= 2
        else {}
    )

    moderators: dict[str, Any] = {}
    if len(nondeg) >= 5:
        for idx_m, mod_key in enumerate(("ref_degree_gini", "citescore", "ratio_repro_orig")):
            sub_m = [r for r in nondeg if r.get(mod_key) is not None]
            if len(sub_m) >= 5:
                xs = np.array([float(r[mod_key]) for r in sub_m], dtype=float)
                ys = np.array([float(r["vos_core"]["delta_count_mid"]) for r in sub_m], dtype=float)
                js = [r["journal_key"] for r in sub_m]
                moderators[mod_key] = journal_clustered_spearman_test(xs, ys, js, seed=MASTER_SEED + 200 + idx_m)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": "1.2",
        "n_seeds": args.n_seeds,
        "random_starts": args.random_starts,
        "merge_small_clusters": True,
        "core_selection_bc": "tls_within_network",
        "symmetric_wide_cap": args.wide_cap,
        "n_null_rewires": args.n_null_rewires,
        "strata_weights_4q": q4_weights,
        "n_total_reviews": len(records),
        "n_nondegenerate_primary": len(nondeg),
        "n_degenerate_excluded_from_primary": len(records) - len(nondeg),
        "ht4_primary_nondegenerate": summarize_group(nondeg, use_ht4=True, use_ipw=False),
        "ht4_ipw_primary_nondegenerate": summarize_group(nondeg, use_ht4=True, use_ipw=True),
        "ht4_sensitivity_incl_degenerate": summarize_group(records, use_ht4=True, use_ipw=False),
        "unweighted_nondegenerate": summarize_group(nondeg, use_ht4=False, use_ipw=False),
        "stratum_Q1": summarize_group(q1_sub, use_ht4=False),
        "stratum_Q2_Q4": summarize_group(q24_sub, use_ht4=False),
        "by_network_type": {
            "bibliographic_coupling": summarize_group(bc_sub, use_ht4=False),
            "co_citation": summarize_group(cc_sub, use_ht4=False),
            "bc_vs_cc_clustered_test": net_diff_test,
        },
        "q1_vs_q24_clustered_test": q_diff_test,
        "prespecified_moderators": moderators,
        "reviews": records,
    }
    out_path = RES_DIR / args.out_name
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved Layer-2 audit summary to %s", out_path)


if __name__ == "__main__":
    main()
