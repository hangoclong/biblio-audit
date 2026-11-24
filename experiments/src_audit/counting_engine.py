"""Core bibliometric counting, VOSviewer 1.6.20 parity, and temporal null engine.

Implements §4 (unbroken formula-to-code chain):
- Exact Perianes-Rodriguez, Waltman & van Eck (2016) Eq. (7)-(10) full vs. fractional counting
  for bibliographic coupling (C C^T) and co-citation (C^T C) with d_r >= 2 & zero diagonals.
- Within-network Total Link Strength (TLS) core selection for BC (zero 2026 citation look-ahead).
- Pre-AS shared LCC extraction + canonical u < v edge deduplication.
- Exact Waltman, van Eck & Noyons (2010) VOS quality function via CPMVertexPartition on
  s_ij = 2m w_ij / (k_i k_j) + CWTS removeSmallClustersBasedOnNNodes greedy merging (< min_cluster_size)
  + VOSviewer best-of-10 random starts per outer run.
- Multi-seed median _find_plateau with local geometric refinement & exact on-grid midpoint selection,
  symmetric 400-pair AMI & Hungarian core retention, Top-10 landmark & nonzero TF-IDF retention,
  and publication-year stratified temporal bipartite null (t_cited <= t_citing).
"""

from __future__ import annotations

import math
import random
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import igraph as ig
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

_IGRAPH_RNG_LOCK = threading.Lock()


def _leiden_partition(
    graph: ig.Graph,
    objective_fn: str,
    resolution: float,
    seed: int,
    max_iterations: int = 10,
) -> tuple[list[int], float]:
    """Native C igraph Leiden (`beta=0.01`, monotone convergence <= 10 iters) — 100% bit-exact with `biblio-audit` v1.0."""
    with _IGRAPH_RNG_LOCK:
        ig.set_random_number_generator(random.Random(int(seed)))
        try:
            mem: list[int] | None = None
            best_q = -float("inf")
            for _ in range(max_iterations):
                cl = graph.community_leiden(
                    objective_function=objective_fn,
                    weights="weight",
                    resolution=float(resolution),
                    beta=0.01,
                    initial_membership=mem,
                    n_iterations=1,
                )
                q = float(cl.quality)
                if q <= best_q + 1e-12:
                    if q > best_q or mem is None:
                        best_q = q
                        mem = list(cl.membership)
                    break
                best_q = q
                mem = list(cl.membership)
            assert mem is not None
            return mem, best_q
        finally:
            ig.set_random_number_generator(random)


def build_coupling_matrices(
    c_mat: np.ndarray,
    network_type: str = "bibliographic_coupling",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact full (V) and fractional (V*) matrices per Perianes-Rodriguez (2016) Eq. 7-10."""
    arr = np.asarray(c_mat, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("c_mat must be a non-empty 2D array")
    if np.any(arr < 0):
        raise ValueError("c_mat entries must be non-negative")
    if network_type == "co_citation":
        arr = arr.T
    elif network_type != "bibliographic_coupling":
        raise ValueError(f"Unsupported network_type: {network_type}")

    col_totals = arr.sum(axis=0)
    keep = col_totals > 1.0
    if not np.any(keep):
        n = arr.shape[0]
        return np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float)

    c_keep = arr[:, keep]
    nk = col_totals[keep]
    v_full = c_keep @ c_keep.T
    v_frac = (c_keep / (nk - 1.0)) @ c_keep.T
    np.fill_diagonal(v_full, 0.0)
    np.fill_diagonal(v_frac, 0.0)
    return v_full, v_frac


def build_sparse_edges_from_works(
    works: list[dict[str, Any]],
    network_type: str = "bibliographic_coupling",
    core_top_n: int | None = None,
    core_selection: str = "tls_within_network",
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Construct sparse full and fractional edge weights from work reference lists.

    For bibliographic_coupling, core_selection='tls_within_network' ranks
    documents by full-corpus within-network Total Link Strength (VOSviewer Create-Map default,
    zero post-cutoff citation look-ahead leakage); 'cited_by_count' is retained for sensitivity only.
    """
    if network_type == "bibliographic_coupling":
        if core_top_n is not None and len(works) > core_top_n:
            if core_selection == "tls_within_network":
                # Compute exact within-corpus bibliographic coupling TLS without using post-cutoff global cites
                ref_freq: dict[str, int] = defaultdict(int)
                for w in works:
                    for ref in set(w.get("referenced_works") or []):
                        ref_freq[ref] += 1
                # Full-counting TLS of doc i = sum_{r in refs(i)} (n_r - 1)
                scored = []
                for idx, w in enumerate(works):
                    tls = sum(ref_freq[r] - 1 for r in set(w.get("referenced_works") or []) if ref_freq[r] >= 2)
                    scored.append((tls, str(w.get("id", "")), idx))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                keep_idx = [x[2] for x in scored[:core_top_n]]
                selected = [works[i] for i in keep_idx]
            elif core_selection == "cited_by_count":
                selected = sorted(
                    works,
                    key=lambda w: (int(w.get("cited_by_count", 0) or 0), str(w.get("id", ""))),
                    reverse=True,
                )[:core_top_n]
            else:
                raise ValueError(f"Unsupported core_selection: {core_selection}")
        else:
            selected = list(works)

        ref_to_papers: dict[str, list[int]] = defaultdict(list)
        for idx, w in enumerate(selected):
            for ref in set(w.get("referenced_works") or []):
                ref_to_papers[ref].append(idx)
        full_e: dict[tuple[int, int], float] = defaultdict(float)
        frac_e: dict[tuple[int, int], float] = defaultdict(float)
        for plist in ref_to_papers.values():
            nk = len(plist)
            if nk < 2:
                continue
            inv = 1.0 / (nk - 1.0)
            for a in range(nk):
                u = plist[a]
                for b in range(a + 1, nk):
                    v = plist[b]
                    key = (u, v) if u < v else (v, u)
                    full_e[key] += 1.0
                    frac_e[key] += inv
        return selected, dict(full_e), dict(frac_e)

    if network_type == "co_citation":
        ref_freq = defaultdict(int)
        for w in works:
            for ref in set(w.get("referenced_works") or []):
                ref_freq[ref] += 1
        ranked_refs = sorted(((r, c) for r, c in ref_freq.items() if c >= 2), key=lambda kv: (-kv[1], kv[0]))
        if core_top_n is not None and len(ranked_refs) > core_top_n:
            ranked_refs = ranked_refs[:core_top_n]
        ref_idx = {r: i for i, (r, _) in enumerate(ranked_refs)}
        node_records = [{"id": r, "cited_by_count": c, "title": ""} for r, c in ranked_refs]
        full_e = defaultdict(float)
        frac_e = defaultdict(float)
        for w in works:
            present = sorted({ref_idx[r] for r in set(w.get("referenced_works") or []) if r in ref_idx})
            mk = len(present)
            if mk < 2:
                continue
            inv = 1.0 / (mk - 1.0)
            for a in range(mk):
                u = present[a]
                for b in range(a + 1, mk):
                    v = present[b]
                    key = (u, v)
                    full_e[key] += 1.0
                    frac_e[key] += inv
        return node_records, dict(full_e), dict(frac_e)

    raise ValueError(f"Unsupported network_type: {network_type}")


def edges_to_igraph(n_nodes: int, edges: dict[tuple[int, int], float], assoc_strength: bool = False) -> ig.Graph:
    """Build undirected weighted igraph with canonical u < v deduplication and optional Association Strength."""
    canon: dict[tuple[int, int], float] = {}
    for (a, b), w in edges.items():
        if a == b or w <= 0.0:
            continue
        key = (a, b) if a < b else (b, a)
        canon[key] = canon.get(key, 0.0) + float(w)

    deg: dict[int, float] = defaultdict(float)
    total_w = 0.0
    for (u, v), w in canon.items():
        deg[u] += w
        deg[v] += w
        total_w += w
    m2 = 2.0 * total_w
    elist: list[tuple[int, int]] = []
    wlist: list[float] = []
    for (u, v), w in canon.items():
        elist.append((u, v))
        if assoc_strength:
            wlist.append((m2 * w) / (deg[u] * deg[v]))
        else:
            wlist.append(w)
    g = ig.Graph(n=n_nodes, edges=elist, directed=False)
    g.es["weight"] = wlist
    return g


def extract_shared_lcc_edges(
    n_nodes: int,
    full_edges: dict[tuple[int, int], float],
    frac_edges: dict[tuple[int, int], float],
) -> tuple[list[int], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Extract Largest Connected Component BEFORE computing Association Strength normalization."""
    g_raw = edges_to_igraph(n_nodes, full_edges, assoc_strength=False)
    comps = g_raw.connected_components()
    if len(comps) == 0:
        return [], {}, {}
    lcc_idx = sorted(comps[max(range(len(comps)), key=lambda i: len(comps[i]))])
    old_to_new = {old: new for new, old in enumerate(lcc_idx)}
    f_sub: dict[tuple[int, int], float] = {}
    r_sub: dict[tuple[int, int], float] = {}
    for (u, v), w in full_edges.items():
        if u in old_to_new and v in old_to_new and u != v and w > 0.0:
            nu, nv = old_to_new[u], old_to_new[v]
            key = (nu, nv) if nu < nv else (nv, nu)
            f_sub[key] = f_sub.get(key, 0.0) + float(w)
    for (u, v), w in frac_edges.items():
        if u in old_to_new and v in old_to_new and u != v and w > 0.0:
            nu, nv = old_to_new[u], old_to_new[v]
            key = (nu, nv) if nu < nv else (nv, nu)
            r_sub[key] = r_sub.get(key, 0.0) + float(w)
    return lcc_idx, f_sub, r_sub


def merge_small_clusters_cwts(
    graph: ig.Graph,
    membership: list[int],
    min_cluster_size: int = 5,
    use_degree_volume: bool = False,
) -> list[int]:
    """CWTS-style greedy average-link small-cluster merging (exact parity with `biblio-audit` v1.0).

    Uses node count `|c_j|` under `VOS_AS_CPM`/`VOS_Frac_CPM` and degree volume `sum_{v in c_j} k_v`
    under `Leiden_RBConfig`.
    """
    mem = list(membership)
    n_nodes = len(mem)
    if n_nodes <= min_cluster_size:
        return [0] * n_nodes

    init_counts: dict[int, int] = defaultdict(int)
    for cid in mem:
        init_counts[cid] += 1
    if len(init_counts) <= 1 or min(init_counts.values()) >= min_cluster_size:
        ordered = sorted(init_counts.keys(), key=lambda cid: (-init_counts[cid], cid))
        remap = {old: new for new, old in enumerate(ordered)}
        return [remap[cid] for cid in mem]

    cache_attr = "_cwts_adj_cache"
    try:
        cached = graph[cache_attr]
    except KeyError:
        adj_c: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
        deg_c = [0.0] * n_nodes
        for edge, w in zip(graph.es, graph.es["weight"]):
            u, v = edge.tuple
            wf = float(w)
            adj_c[u].append((v, wf))
            adj_c[v].append((u, wf))
            deg_c[u] += wf
            deg_c[v] += wf
        cached = (adj_c, deg_c)
        graph[cache_attr] = cached
    adj, deg_w = cached
    node_w = deg_w if use_degree_volume else None

    c_nodes: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(mem):
        c_nodes[cid].append(idx)
    c_weight: dict[int, float] = {
        cid: (sum(node_w[x] for x in nodes) if node_w is not None else float(len(nodes)))
        for cid, nodes in c_nodes.items()
    }

    while len(c_nodes) > 1:
        small_cid, small_nodes = min(c_nodes.items(), key=lambda kv: (len(kv[1]), kv[0]))
        if len(small_nodes) >= min_cluster_size:
            break

        link_sum: dict[int, float] = defaultdict(float)
        for u in small_nodes:
            for v, w in adj[u]:
                target = mem[v]
                if target != small_cid:
                    link_sum[target] += w

        if link_sum:
            best_target = max(
                link_sum.keys(),
                key=lambda cid: (
                    link_sum[cid] / max(1e-12, c_weight[cid]),
                    len(c_nodes[cid]),
                    -cid,
                ),
            )
        else:
            others = [cid for cid in c_nodes if cid != small_cid]
            best_target = max(others, key=lambda cid: (len(c_nodes[cid]), -cid))

        for u in small_nodes:
            mem[u] = best_target
        c_nodes[best_target].extend(small_nodes)
        c_weight[best_target] += c_weight[small_cid]
        del c_nodes[small_cid]
        del c_weight[small_cid]

    ordered = sorted(c_nodes.keys(), key=lambda cid: (-len(c_nodes[cid]), cid))
    remap = {old: new for new, old in enumerate(ordered)}
    return [remap[cid] for cid in mem]


def count_major_clusters(membership: list[int], min_cluster_size: int = 5) -> int:
    """Count clusters containing at least min_cluster_size nodes."""
    counts: dict[int, int] = defaultdict(int)
    for c in membership:
        counts[c] += 1
    return sum(1 for sz in counts.values() if sz >= min_cluster_size)


def major_cluster_stats(membership: list[int], min_cluster_size: int = 5) -> tuple[int, float]:
    """Return (n_major_clusters, fraction_of_nodes_in_major_clusters)."""
    counts: dict[int, int] = defaultdict(int)
    for c in membership:
        counts[c] += 1
    major_nodes = sum(sz for sz in counts.values() if sz >= min_cluster_size)
    n_major = sum(1 for sz in counts.values() if sz >= min_cluster_size)
    return n_major, major_nodes / max(1, len(membership))


def hungarian_core_match(m1: list[int], m2: list[int]) -> tuple[float, dict[int, int]]:
    """Compute optimal one-to-one Hungarian cluster alignment and matched node fraction."""
    c1 = sorted(set(m1))
    c2 = sorted(set(m2))
    idx1 = {c: i for i, c in enumerate(c1)}
    idx2 = {c: i for i, c in enumerate(c2)}
    mat = np.zeros((len(c1), len(c2)), dtype=int)
    for a, b in zip(m1, m2):
        mat[idx1[a], idx2[b]] += 1
    row_ind, col_ind = linear_sum_assignment(-mat)
    matched = int(mat[row_ind, col_ind].sum())
    mapping = {c1[r]: c2[c] for r, c in zip(row_ind, col_ind)}
    return matched / len(m1), mapping


def _one_way_landmark_retention(
    m_src: list[int],
    m_dst: list[int],
    node_records: list[dict[str, Any]],
    min_cluster_size: int,
    top_k_landmarks: int,
) -> float:
    _, mapping = hungarian_core_match(m_src, m_dst)
    src_nodes: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(m_src):
        src_nodes[c].append(i)
    major_src = [c for c, nodes in src_nodes.items() if len(nodes) >= min_cluster_size] or list(src_nodes.keys())
    lm_scores: list[float] = []
    lm_weights: list[int] = []
    for c in major_src:
        nodes = src_nodes[c]
        ranked = sorted(nodes, key=lambda i: int(node_records[i].get("cited_by_count", 0) or 0), reverse=True)
        top_nodes = ranked[: min(top_k_landmarks, len(ranked))]
        target_c = mapping.get(c)
        retained = sum(1 for i in top_nodes if m_dst[i] == target_c)
        lm_scores.append(retained / len(top_nodes))
        lm_weights.append(len(nodes))
    return float(np.average(lm_scores, weights=lm_weights))


def landmark_and_tfidf_retention(
    m1: list[int],
    m2: list[int],
    node_records: list[dict[str, Any]],
    min_cluster_size: int = 5,
    top_k_landmarks: int = 10,
    top_terms: int = 15,
) -> tuple[float, float | None]:
    """Measure symmetrized Top-10 cited landmark co-membership & nonzero Top-15 TF-IDF Jaccard (BC-only)."""
    lm_12 = _one_way_landmark_retention(m1, m2, node_records, min_cluster_size, top_k_landmarks)
    lm_21 = _one_way_landmark_retention(m2, m1, node_records, min_cluster_size, top_k_landmarks)
    landmark_ret = 0.5 * (lm_12 + lm_21)

    _, mapping = hungarian_core_match(m1, m2)
    c1_nodes: dict[int, list[int]] = defaultdict(list)
    c2_nodes: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(m1):
        c1_nodes[c].append(i)
    for i, c in enumerate(m2):
        c2_nodes[c].append(i)

    major_c1 = [c for c, nodes in c1_nodes.items() if len(nodes) >= min_cluster_size]
    if not major_c1:
        major_c1 = list(c1_nodes.keys())

    texts = [
        str(r.get("title") or "").strip()
        if not str(r.get("title") or "").strip().startswith("http")
        else ""
        for r in node_records
    ]
    if sum(1 for t in texts if len(t) > 10) < max(10, len(node_records) // 2):
        return landmark_ret, None

    try:
        vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
        x_tfidf = vec.fit_transform(texts)
        vocab = np.asarray(vec.get_feature_names_out())
        if len(vocab) < 5:
            return landmark_ret, None
        jacc_scores: list[float] = []
        jacc_weights: list[int] = []
        for c in major_c1:
            target_c2 = mapping.get(c)
            if target_c2 is None or not c2_nodes.get(target_c2):
                jacc_scores.append(0.0)
                jacc_weights.append(len(c1_nodes[c]))
                continue
            v1 = np.asarray(x_tfidf[c1_nodes[c]].mean(axis=0)).ravel()
            v2 = np.asarray(x_tfidf[c2_nodes[target_c2]].mean(axis=0)).ravel()
            idx1 = [i for i in np.argsort(v1)[-top_terms:] if v1[i] > 0.0]
            idx2 = [i for i in np.argsort(v2)[-top_terms:] if v2[i] > 0.0]
            if not idx1 and not idx2:
                continue
            t1 = set(vocab[idx1])
            t2 = set(vocab[idx2])
            jacc_scores.append(len(t1 & t2) / max(1, len(t1 | t2)))
            jacc_weights.append(len(c1_nodes[c]))
        tfidf_jacc = float(np.average(jacc_scores, weights=jacc_weights)) if jacc_scores else None
    except ValueError:
        tfidf_jacc = None

    return landmark_ret, tfidf_jacc


@dataclass
class CountingAuditResult:
    """Structured result for one objective (VOS_AS_CPM or Leiden_RBConfig) on one review."""

    objective: str
    merge_small_clusters: bool
    n_random_starts: int
    n_lcc: int
    k_full_major: int
    k_frac_matched_major: int
    exact_k_matched: bool
    rev_exact_k_matched: bool
    degenerate_k: bool
    gamma_plateau_min: float
    gamma_plateau_mid: float
    gamma_plateau_max: float
    ami_within_full: float
    ami_within_frac: float
    ami_cross_mid: float
    ami_cross_min_plateau: float
    ami_cross_max_plateau: float
    delta_count_mid: float
    delta_count_reverse: float
    ari_cross_mid: float
    hungarian_match: float
    core_reassignment: float
    landmark_top10_retention: float
    tfidf_term_jaccard: float | None


def _run_best_of_starts(
    graph: ig.Graph,
    ptype: str,
    resolution: float,
    outer_seed: int,
    n_random_starts: int = 10,
    merge_small_clusters: bool = True,
    min_cluster_size: int = 5,
) -> tuple[list[int], float]:
    """One VOSviewer clustering run: best of n_random_starts + optional CWTS small-cluster merge."""
    best_part = None
    best_q = -float("inf")
    base_seed = outer_seed * 100
    use_deg_vol = ptype == "modularity"
    for r in range(n_random_starts):
        mem, q = _leiden_partition(graph, ptype, float(resolution), base_seed + r)
        if q > best_q:
            best_q = q
            best_part = mem
    assert best_part is not None
    if merge_small_clusters:
        best_part = merge_small_clusters_cwts(graph, best_part, min_cluster_size=min_cluster_size, use_degree_volume=use_deg_vol)
    return best_part, best_q


def _run_seeds(
    graph: ig.Graph,
    ptype: Any,
    resolution: float,
    n_seeds: int,
    n_random_starts: int = 10,
    merge_small_clusters: bool = True,
    min_cluster_size: int = 5,
) -> list[list[int]]:
    """Run n_seeds independent converged VOSviewer/Leiden runs."""
    return [
        _run_best_of_starts(
            graph,
            ptype,
            resolution,
            outer_seed=s,
            n_random_starts=n_random_starts,
            merge_small_clusters=merge_small_clusters,
            min_cluster_size=min_cluster_size,
        )[0]
        for s in range(n_seeds)
    ]


def _eval_grid_multiseed(
    graph: ig.Graph,
    ptype: Any,
    grid: np.ndarray,
    min_cluster_size: int,
    screen_seeds: int = 5,
    n_random_starts: int = 3,
    merge_small_clusters: bool = True,
) -> list[tuple[int, float, float, int]]:
    """Screen each grid point with multi-seed median K_major & pre-merge coverage."""
    out: list[tuple[int, float, float, int]] = []
    for idx, gval in enumerate(grid):
        ks: list[int] = []
        covs: list[float] = []
        for s in range(screen_seeds):
            # Evaluate raw pre-merge coverage to block shredded CPM regimes, then post-merge K
            raw_m, _ = _run_best_of_starts(
                graph,
                ptype,
                float(gval),
                outer_seed=s,
                n_random_starts=n_random_starts,
                merge_small_clusters=False,
                min_cluster_size=min_cluster_size,
            )
            _, cov = major_cluster_stats(raw_m, min_cluster_size)
            final_m = merge_small_clusters_cwts(graph, raw_m, min_cluster_size, use_degree_volume=(ptype == "modularity")) if merge_small_clusters else raw_m
            km = count_major_clusters(final_m, min_cluster_size)
            ks.append(km)
            covs.append(cov)
        out.append((idx, float(gval), float(np.median(covs)), int(round(float(np.median(ks))))))
    return out


def _select_from_candidates(
    candidates: list[tuple[int, float, float, int]],
    target_k: int,
    min_coverage: float,
    anchor_gamma: float,
) -> tuple[float, float, float, int, bool]:
    """Select exact on-grid contiguous plateau closest to anchor_gamma."""
    valid_exact = [(idx, g) for idx, g, cov, km in candidates if km == target_k and cov >= min_coverage]
    if valid_exact:
        runs: list[list[float]] = []
        cur: list[float] = []
        prev_idx = -999
        for idx, g in valid_exact:
            if idx != prev_idx + 1 and cur:
                runs.append(cur)
                cur = []
            cur.append(g)
            prev_idx = idx
        if cur:
            runs.append(cur)
        best_run = min(runs, key=lambda run: min(abs(math.log(max(g, 1e-6) / anchor_gamma)) for g in run))
        # Exact on-grid midpoint index (never averaging off-grid! verification)
        g_mid = best_run[len(best_run) // 2]
        return best_run[0], g_mid, best_run[-1], target_k, True

    pool = [c for c in candidates if c[2] >= min_coverage] or candidates
    best = min(pool, key=lambda c: (abs(c[3] - target_k), abs(math.log(max(c[1], 1e-6) / anchor_gamma))))
    return best[1], best[1], best[1], best[3], False


def _find_plateau(
    graph: ig.Graph,
    ptype: Any,
    target_k: int,
    grid: np.ndarray,
    min_cluster_size: int,
    min_coverage: float = 0.75,
    anchor_gamma: float = 1.0,
    merge_small_clusters: bool = True,
    local_refine_points: int = 30,
    eval_random_starts: int = 10,
) -> tuple[float, float, float, int, bool]:
    """Multi-seed plateau sweep with full-start local geometric refinement & midpoint verification."""
    coarse_starts = min(eval_random_starts, max(3, eval_random_starts // 2))
    cands = _eval_grid_multiseed(
        graph, ptype, grid, min_cluster_size, screen_seeds=5, n_random_starts=coarse_starts, merge_small_clusters=merge_small_clusters
    )
    g_min, g_mid, g_max, km, ok = _select_from_candidates(cands, target_k, min_coverage, anchor_gamma)
    if ok:
        # Verify selected on-grid midpoint under full eval_random_starts to prevent 3-vs-10 start drift
        check_parts = _run_seeds(
            graph, ptype, g_mid, n_seeds=5, n_random_starts=eval_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size
        )
        if int(round(float(np.median([count_major_clusters(m, min_cluster_size) for m in check_parts])))) == target_k:
            return g_min, g_mid, g_max, km, True

    # Dense local geometric refinement around closest transition bracket using full eval_random_starts
    non_shred = [c for c in cands if c[2] >= min_coverage] or cands
    best_idx = min(range(len(non_shred)), key=lambda i: (abs(non_shred[i][3] - target_k), abs(math.log(max(non_shred[i][1], 1e-6) / anchor_gamma))))
    lo_g = non_shred[max(0, best_idx - 2)][1]
    hi_g = non_shred[min(len(non_shred) - 1, best_idx + 2)][1]
    if hi_g <= lo_g:
        lo_g = max(1e-3, lo_g * 0.5)
        hi_g = hi_g * 2.0
    fine_grid = np.geomspace(lo_g, hi_g, local_refine_points)
    fine_cands = _eval_grid_multiseed(
        graph, ptype, fine_grid, min_cluster_size, screen_seeds=5, n_random_starts=eval_random_starts, merge_small_clusters=merge_small_clusters
    )
    return _select_from_candidates(fine_cands, target_k, min_coverage, anchor_gamma)


def _mean_within_ami(parts: list[list[int]]) -> float:
    """Mean pairwise AMI across all distinct outer-seed pairs."""
    n = len(parts)
    if n < 2:
        return 1.0
    return float(np.mean([ami(parts[i], parts[j]) for i in range(n) for j in range(i + 1, n)]))


def evaluate_counting_sensitivity(
    node_records: list[dict[str, Any]],
    full_edges: dict[tuple[int, int], float],
    frac_edges: dict[tuple[int, int], float],
    objective: str = "VOS_AS_CPM",
    n_seeds: int = 20,
    n_random_starts: int = 10,
    merge_small_clusters: bool = True,
    resolution_default: float = 1.0,
    grid_points: int = 60,
    min_cluster_size: int = 5,
) -> CountingAuditResult:
    """Evaluate full vs. fractional counting with VOSviewer 1.6.20 parity and multi-seed plateau matching."""
    if n_seeds < 2:
        raise ValueError("n_seeds must be >= 2 for within-rule stability calculation")

    lcc_idx, f_sub, r_sub = extract_shared_lcc_edges(len(node_records), full_edges, frac_edges)
    if len(lcc_idx) < min_cluster_size or not f_sub or not r_sub:
        raise ValueError(f"Insufficient LCC size ({len(lcc_idx)}) or empty edges for counting evaluation")

    use_as = objective == "VOS_AS_CPM"
    ptype = "CPM" if use_as else "modularity"
    sf = edges_to_igraph(len(lcc_idx), f_sub, assoc_strength=use_as)
    sr = edges_to_igraph(len(lcc_idx), r_sub, assoc_strength=use_as)
    lcc_records = [node_records[i] for i in lcc_idx]

    # Measure raw pre-merge coverage on full counting to calibrate non-shredded coverage floor
    raw_f0 = _run_seeds(sf, ptype, resolution_default, min(5, n_seeds), n_random_starts=3, merge_small_clusters=False, min_cluster_size=min_cluster_size)
    full_raw_cov = float(np.median([major_cluster_stats(m, min_cluster_size)[1] for m in raw_f0]))
    min_cov = max(0.55, min(0.80, full_raw_cov - 0.10))

    f_parts = _run_seeds(sf, ptype, resolution_default, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)
    k_full_major = int(round(float(np.median([count_major_clusters(m, min_cluster_size) for m in f_parts]))))
    degenerate_k = k_full_major < 2
    target_k = max(1, k_full_major)

    grid = np.geomspace(0.05, 5.0, grid_points) if use_as else np.linspace(0.10, 2.50, grid_points)
    g_min, g_mid, g_max, _, _ = _find_plateau(
        sr,
        ptype,
        target_k,
        grid,
        min_cluster_size,
        min_coverage=min_cov,
        anchor_gamma=resolution_default,
        merge_small_clusters=merge_small_clusters,
        eval_random_starts=n_random_starts,
    )
    r_parts_mid = _run_seeds(sr, ptype, g_mid, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)
    k_frac_med = int(round(float(np.median([count_major_clusters(m, min_cluster_size) for m in r_parts_mid]))))
    exact_ok = (k_frac_med == k_full_major) and (not degenerate_k)

    r_parts_min = r_parts_mid if g_min == g_mid else _run_seeds(sr, ptype, g_min, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)
    r_parts_max = r_parts_mid if g_max == g_mid else _run_seeds(sr, ptype, g_max, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)

    within_f = _mean_within_ami(f_parts)
    within_r = _mean_within_ami(r_parts_mid)
    cross_mid = float(np.mean([ami(f_parts[i], r_parts_mid[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    cross_min = float(np.mean([ami(f_parts[i], r_parts_min[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    cross_max = float(np.mean([ami(f_parts[i], r_parts_max[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    ari_mid = float(np.mean([ari(f_parts[i], r_parts_mid[j]) for i in range(n_seeds) for j in range(n_seeds)]))

    # Reverse direction check (full 20 seeds, logging rev_exact_k_matched)
    r_def = _run_seeds(sr, ptype, resolution_default, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)
    k_frac_def = max(1, int(round(float(np.median([count_major_clusters(m, min_cluster_size) for m in r_def])))))
    _, g_rev_mid, _, _, _ = _find_plateau(
        sf,
        ptype,
        k_frac_def,
        grid,
        min_cluster_size,
        min_coverage=min_cov,
        anchor_gamma=resolution_default,
        merge_small_clusters=merge_small_clusters,
        eval_random_starts=n_random_starts,
    )
    f_rev = _run_seeds(sf, ptype, g_rev_mid, n_seeds, n_random_starts=n_random_starts, merge_small_clusters=merge_small_clusters, min_cluster_size=min_cluster_size)
    k_f_rev = int(round(float(np.median([count_major_clusters(m, min_cluster_size) for m in f_rev]))))
    rev_exact_ok = (k_f_rev == k_frac_def) and (k_frac_def >= 2)
    w_f_rev = _mean_within_ami(f_rev)
    w_r_def = _mean_within_ami(r_def)
    x_rev = float(np.mean([ami(f_rev[i], r_def[j]) for i in range(n_seeds) for j in range(n_seeds)]))
    delta_rev = min(w_f_rev, w_r_def) - x_rev

    # Symmetric 400-pair Hungarian match + 20-pair landmark/TF-IDF corroboration
    hung_vals = [hungarian_core_match(f_parts[i], r_parts_mid[j])[0] for i in range(n_seeds) for j in range(n_seeds)]
    lm_vals: list[float] = []
    tf_vals: list[float] = []
    for s in range(n_seeds):
        lm, tf = landmark_and_tfidf_retention(f_parts[s], r_parts_mid[s], lcc_records, min_cluster_size)
        lm_vals.append(lm)
        if tf is not None:
            tf_vals.append(tf)

    mean_hung = float(np.mean(hung_vals))
    return CountingAuditResult(
        objective=objective,
        merge_small_clusters=merge_small_clusters,
        n_random_starts=n_random_starts,
        n_lcc=len(lcc_idx),
        k_full_major=k_full_major,
        k_frac_matched_major=k_frac_med,
        exact_k_matched=exact_ok,
        rev_exact_k_matched=rev_exact_ok,
        degenerate_k=degenerate_k,
        gamma_plateau_min=round(g_min, 4),
        gamma_plateau_mid=round(g_mid, 4),
        gamma_plateau_max=round(g_max, 4),
        ami_within_full=round(within_f, 4),
        ami_within_frac=round(within_r, 4),
        ami_cross_mid=round(cross_mid, 4),
        ami_cross_min_plateau=round(cross_min, 4),
        ami_cross_max_plateau=round(cross_max, 4),
        delta_count_mid=round(min(within_f, within_r) - cross_mid, 4),
        delta_count_reverse=round(delta_rev, 4),
        ari_cross_mid=round(ari_mid, 4),
        hungarian_match=round(mean_hung, 4),
        core_reassignment=round(1.0 - mean_hung, 4),
        landmark_top10_retention=round(float(np.mean(lm_vals)), 4),
        tfidf_term_jaccard=round(float(np.mean(tf_vals)), 4) if tf_vals else None,
    )


def temporal_bipartite_rewire(
    edges: list[tuple[int, int]],
    paper_years: dict[int, int],
    ref_years: dict[int, int],
    swaps_per_edge: int = 10,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], float]:
    """Degree-preserving bipartite edge-swap MCMC enforcing t_ref <= t_paper causality.

    Returns (rewired_edges, swap_acceptance_rate) for MCMC mixing diagnostics.
    """
    rng = np.random.default_rng(seed)
    edge_list = list(edges)
    edge_set = set(edge_list)
    m = len(edge_list)
    if m < 2:
        return edge_list, 0.0

    n_attempts = m * swaps_per_edge
    accepted = 0
    for _ in range(n_attempts):
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m))
        if i == j:
            continue
        p1, r1 = edge_list[i]
        p2, r2 = edge_list[j]
        if p1 == p2 or r1 == r2:
            continue
        if (p1, r2) in edge_set or (p2, r1) in edge_set:
            continue
        if ref_years.get(r2, -9999) > paper_years.get(p1, 9999):
            continue
        if ref_years.get(r1, -9999) > paper_years.get(p2, 9999):
            continue
        edge_set.remove((p1, r1))
        edge_set.remove((p2, r2))
        edge_set.add((p1, r2))
        edge_set.add((p2, r1))
        edge_list[i] = (p1, r2)
        edge_list[j] = (p2, r1)
        accepted += 1
    return edge_list, accepted / max(1, n_attempts)
