"""VOSviewer 1.6.20/1.6.21 parity clustering and counting-sensitivity engine for `biblio-audit`.

Implements:
- All 3 VOSviewer normalizations (`Network.java` / `ClusteringCreator.js`):
  1. `VOS_AS_CPM`: Association Strength `2m * w_ij / (k_i * k_j)` + Leiden `CPMVertexPartition`
  2. `VOS_Frac_CPM`: VOSviewer Fractionalization `(n/2) * (w_ij/k_i + w_ij/k_j)` + Leiden `CPMVertexPartition`
  3. `Leiden_RBConfig`: LinLog / Modularity (`RBConfigurationVertexPartition` on raw weights)
- Pre-normalization shared Largest Connected Component (LCC) extraction
- Exact CWTS `removeSmallClustersBasedOnNNodes` greedy average-link merge (node-count denominator for
  CPM normalizations; degree-volume denominator for `Leiden_RBConfig`)
- Best-of-10 random starts per outer run & 1-step boundary-adaptive matched-K resolution plateau sweep
- Full multi-perspective diagnostics: `Delta_count` (AMI), `AECS` / `Delta_AECS` (exact hypergeometric
  chance-adjusted Element-Centric Similarity), `VI` / `NVI`, `ARI`, Hungarian reassignment,
  symmetrized Top-10 landmark retention, BC TF-IDF Jaccard, Medoid consensus node table, and guarded ROPE badge.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import igraph as ig
import numpy as np

from .exceptions import InsufficientNetworkError
from .metrics import (
    align_fractional_to_full,
    hungarian_core_match,
    landmark_and_tfidf_retention,
    mean_pairwise_metrics,
    select_medoid_partition,
)
from .networks import (  # re-exported for backward compatibility
    NetworkBuildArtifact,
    build_coupling_matrices,
    build_network_artifact,
    build_sparse_edges_from_works,
    reference_degree_gini,
)
from .nulls import _IGRAPH_RNG_LOCK, BipartiteNullResult, evaluate_bipartite_null

SUPPORTED_OBJECTIVES = ("VOS_AS_CPM", "VOS_Frac_CPM", "Leiden_RBConfig")


def _leiden_partition(
    graph: ig.Graph,
    objective_fn: str,
    resolution: float,
    seed: int,
    max_iterations: int = 10,
) -> tuple[list[int], float]:
    """Run deterministic native C igraph Leiden (`Traag, Waltman & van Eck, 2019`, `beta=0.01`).

    Uses bounded monotone convergence (`<= max_iterations`, default 10 matching VOSviewer 1.6.20,
    stopping early when `Delta Q <= 1e-12`) to remain bit-exact with full convergence while making
    floating-point tie-oscillation infinite loops impossible across both `leidenalg` and C `igraph`.
    """
    if graph.vcount() == 0:
        return [], 0.0
    if graph.ecount() == 0:
        return list(range(graph.vcount())), 0.0
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


def classify_single_map_band(
    delta_count: float,
    degenerate_k: bool = False,
    lcc_fraction: float = 1.0,
    min_within_ami: float = 1.0,
    core_reassignment: float = 0.0,
) -> tuple[str, list[str]]:
    """Classify a single review against the BC/CC-calibrated [0.10, 0.20] ROPE thresholds with safety guards."""
    flags: list[str] = []
    if degenerate_k:
        return "DEGENERATE_K_LT_2", ["DEGENERATE_K_LT_2: Full counting yielded fewer than 2 major clusters."]
    if lcc_fraction < 0.50:
        flags.append(f"FRAGMENTED_LCC_LT_50PCT: Shared LCC retains only {100.0 * lcc_fraction:.1f}% of selected core nodes.")
    if min_within_ami < 0.85 or core_reassignment >= 0.25:
        if delta_count < 0.20:
            flags.append(
                f"HIGH_TURNOVER_OR_SOLVER_DISPERSION: Within-rule AMI={min_within_ami:.3f}, core reassignment={100.0 * core_reassignment:.1f}%."
            )

    if delta_count <= 0.10:
        base = "LOW_SENSITIVITY_PASS" if not flags else "LOW_SENSITIVITY_GUARDED"
    elif delta_count >= 0.20:
        base = "HIGH_SENSITIVITY_REPORT_FRACTIONAL"
    else:
        base = "MODERATE_SENSITIVITY_CHECK_BOTH"
    return base, flags


def edges_to_igraph(
    n_nodes: int,
    edges: dict[tuple[int, int], float],
    normalization: str = "none",
    assoc_strength: bool | None = None,
) -> ig.Graph:
    """Build undirected weighted igraph with canonical u < v deduplication and VOSviewer normalization.

    `normalization` accepts `'none'`, `'association_strength'`, or `'fractionalization'`.
    Legacy boolean `assoc_strength=True` is preserved for backward compatibility.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be >= 0, got {n_nodes}.")
    if assoc_strength is not None:
        normalization = "association_strength" if assoc_strength else "none"
    if normalization not in {"none", "association_strength", "fractionalization"}:
        raise ValueError(f"Unsupported normalization: {normalization}")

    canon: dict[tuple[int, int], float] = {}
    for (a, b), w in edges.items():
        if a < 0 or b < 0 or a >= n_nodes or b >= n_nodes:
            raise ValueError(f"Edge ({a}, {b}) out of bounds for n_nodes={n_nodes}.")
        wf = float(w)
        if not math.isfinite(wf):
            raise ValueError(f"Edge weight for ({a}, {b}) must be finite, got {w!r}.")
        if a == b or wf <= 0.0:
            continue
        key = (a, b) if a < b else (b, a)
        canon[key] = canon.get(key, 0.0) + wf

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
        du, dv = deg[u], deg[v]
        if normalization == "association_strength":
            if du <= 0.0 or dv <= 0.0:
                continue
            # Divide sequentially to prevent float underflow in du * dv on tiny positive weights
            nw = (m2 / du) * (w / dv)
        elif normalization == "fractionalization":
            if du <= 0.0 or dv <= 0.0:
                continue
            # Van Eck & Waltman (2009) Eq. (13) / CWTS Network.java: (n / 2) * (w_ij/k_i + w_ij/k_j)
            nw = 0.5 * float(n_nodes) * ((w / du) + (w / dv))
        else:
            nw = w
        elist.append((u, v))
        wlist.append(nw)

    g = ig.Graph(n=n_nodes, edges=elist, directed=False)
    g.es["weight"] = wlist
    return g


def extract_shared_lcc_edges(
    n_nodes: int,
    full_edges: dict[tuple[int, int], float],
    frac_edges: dict[tuple[int, int], float],
) -> tuple[list[int], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Extract Largest Connected Component BEFORE applying VOSviewer edge normalization."""
    g_raw = edges_to_igraph(n_nodes, full_edges, normalization="none")
    comps = g_raw.connected_components()
    if len(comps) == 0:
        return [], {}, {}
    lcc_idx = sorted(comps[max(range(len(comps)), key=lambda i: len(comps[i]))])
    old_to_new = {old: new for new, old in enumerate(lcc_idx)}
    f_sub: dict[tuple[int, int], float] = {}
    r_sub: dict[tuple[int, int], float] = {}
    for (u, v), w in full_edges.items():
        wf = float(w)
        if u in old_to_new and v in old_to_new and u != v and math.isfinite(wf) and wf > 0.0:
            nu, nv = old_to_new[u], old_to_new[v]
            key = (nu, nv) if nu < nv else (nv, nu)
            f_sub[key] = f_sub.get(key, 0.0) + wf
    for (u, v), w in frac_edges.items():
        wf = float(w)
        if u in old_to_new and v in old_to_new and u != v and math.isfinite(wf) and wf > 0.0:
            nu, nv = old_to_new[u], old_to_new[v]
            key = (nu, nv) if nu < nv else (nv, nu)
            r_sub[key] = r_sub.get(key, 0.0) + wf
    return lcc_idx, f_sub, r_sub


def merge_small_clusters_cwts(
    graph: ig.Graph,
    membership: list[int],
    min_cluster_size: int = 5,
    use_degree_volume: bool = False,
) -> list[int]:
    """CWTS `removeSmallClustersBasedOnNNodes` greedy merge (< min_cluster_size).

    Matches `CPMClusteringAlgorithm.java`: maximizes `totalEdgeWeightPerCluster[j] / clusterWeights[j]`,
    where `clusterWeights[j]` is node count `|c_j|` under `VOS_AS_CPM`/`VOS_Frac_CPM` (`n'_i = 1`)
    and degree volume `sum_{v in c_j} k_v` under `Leiden_RBConfig` (`n'_i = k_i`).
    """
    mem = list(membership)
    n_nodes = len(mem)
    if min_cluster_size <= 1 or n_nodes <= min_cluster_size:
        final_counts: dict[int, int] = defaultdict(int)
        for cid in mem:
            final_counts[cid] += 1
        ordered = sorted(final_counts.keys(), key=lambda cid: (-final_counts[cid], cid))
        remap = {old: new for new, old in enumerate(ordered)}
        return [remap[cid] for cid in mem]

    # Fast early-exit when every cluster already meets min_cluster_size (avoids graph traversal)
    init_counts: dict[int, int] = defaultdict(int)
    for cid in mem:
        init_counts[cid] += 1
    if len(init_counts) <= 1 or min(init_counts.values()) >= min_cluster_size:
        ordered = sorted(init_counts.keys(), key=lambda cid: (-init_counts[cid], cid))
        remap = {old: new for new, old in enumerate(ordered)}
        return [remap[cid] for cid in mem]

    # Cache adjacency list and degree weights on the igraph object across repeated sweeps (100% bit-exact)
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


def _major_cluster_stats(membership: list[int], min_cluster_size: int = 5) -> tuple[int, float]:
    counts: dict[int, int] = defaultdict(int)
    for c in membership:
        counts[c] += 1
    eff_min = max(1, min_cluster_size)
    major_nodes = sum(sz for sz in counts.values() if sz >= eff_min)
    n_major = sum(1 for sz in counts.values() if sz >= eff_min)
    return n_major, major_nodes / max(1, len(membership))


@dataclass
class NodeStabilityRow:
    """Node-level consensus cluster assignment and ECS stability diagnostic."""

    node_id: str
    label: str
    cited_by_count: int
    occurrences: int
    full_cluster_medoid: int
    frac_cluster_aligned: int
    switched_cluster: bool
    ecs_within_full: float
    ecs_within_frac: float
    ecs_cross: float
    delta_ecs_node: float
    is_boundary_shifter: bool


@dataclass
class CountingAuditResult:
    """Structured VOSviewer-parity counting sensitivity report for one bibliometric network."""

    network_type: str
    objective: str
    frac_scheme: str
    denom_scope: str
    merge_small_clusters: bool
    min_cluster_size: int
    n_random_starts: int
    n_seeds: int
    n_core_selected: int
    n_lcc: int
    lcc_fraction: float
    event_degree_gini: float
    tls_core_turnover_jaccard: float | None
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
    delta_count_mid: float
    delta_count_reverse: float
    ecs_within_full: float
    ecs_within_frac: float
    ecs_cross_mid: float
    aecs_within_full: float
    aecs_within_frac: float
    aecs_cross_mid: float
    delta_aecs_mid: float
    vi_cross_bits: float
    nvi_cross: float
    ari_cross_mid: float
    hungarian_match: float
    core_reassignment: float
    landmark_top10_retention: float
    tfidf_term_jaccard: float | None
    triage_band: str
    warnings: list[str] = field(default_factory=list)
    null_benchmark: dict[str, Any] | None = None
    node_rows: list[NodeStabilityRow] = field(default_factory=list, repr=False)
    medoid_full_membership: list[int] = field(default_factory=list, repr=False)
    medoid_frac_membership: list[int] = field(default_factory=list, repr=False)
    lcc_full_edges: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    lcc_frac_edges: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize scalar diagnostics and warnings cleanly (excluding bulky edge/node tables)."""
        d = asdict(self)
        for bulky in ("node_rows", "medoid_full_membership", "medoid_frac_membership", "lcc_full_edges", "lcc_frac_edges"):
            d.pop(bulky, None)
        return d


def _run_best_of_starts(
    graph: ig.Graph,
    objective_fn: str,
    resolution: float,
    outer_seed: int,
    n_random_starts: int,
    merge_small: bool,
    min_cluster_size: int,
    use_degree_volume: bool,
) -> tuple[list[int], float]:
    best_part: list[int] | None = None
    best_q = -float("inf")
    base_seed = outer_seed * 100
    for r in range(n_random_starts):
        mem, q = _leiden_partition(graph, objective_fn, float(resolution), base_seed + r)
        if q > best_q:
            best_q = q
            best_part = mem
    assert best_part is not None
    final_mem = (
        merge_small_clusters_cwts(graph, best_part, min_cluster_size, use_degree_volume=use_degree_volume)
        if merge_small
        else best_part
    )
    return final_mem, best_q


def _run_seeds(
    graph: ig.Graph,
    ptype: Any,
    resolution: float,
    n_seeds: int,
    n_random_starts: int,
    merge_small: bool,
    min_cluster_size: int,
    use_degree_volume: bool,
) -> tuple[list[list[int]], list[float]]:
    parts, qs = [], []
    for s in range(n_seeds):
        m, q = _run_best_of_starts(
            graph, ptype, resolution, s, n_random_starts, merge_small, min_cluster_size, use_degree_volume
        )
        parts.append(m)
        qs.append(q)
    return parts, qs


def _eval_grid(
    graph: ig.Graph,
    ptype: Any,
    grid: np.ndarray,
    min_cluster_size: int,
    n_starts: int,
    merge_small: bool,
    use_degree_volume: bool,
) -> list[tuple[int, float, float, int]]:
    out = []
    for idx, gval in enumerate(grid):
        ks, covs = [], []
        for s in range(5):
            raw_m, _ = _run_best_of_starts(graph, ptype, float(gval), s, n_starts, False, min_cluster_size, use_degree_volume)
            _, cov = _major_cluster_stats(raw_m, min_cluster_size)
            fin_m = (
                merge_small_clusters_cwts(graph, raw_m, min_cluster_size, use_degree_volume=use_degree_volume)
                if merge_small
                else raw_m
            )
            km, _ = _major_cluster_stats(fin_m, min_cluster_size)
            ks.append(km)
            covs.append(cov)
        out.append((idx, float(gval), float(np.median(covs)), int(round(float(np.median(ks))))))
    return out


def _select_plateau(
    cands: list[tuple[int, float, float, int]], target_k: int, min_cov: float, anchor_g: float
) -> tuple[float, float, float, int, bool]:
    valid = [(idx, g) for idx, g, cov, km in cands if km == target_k and cov >= min_cov]
    if valid:
        runs: list[list[float]] = []
        cur: list[float] = []
        prev = -999
        for idx, g in valid:
            if idx != prev + 1 and cur:
                runs.append(cur)
                cur = []
            cur.append(g)
            prev = idx
        if cur:
            runs.append(cur)
        best = min(runs, key=lambda run: min(abs(math.log(max(g, 1e-6) / anchor_g)) for g in run))
        return best[0], best[len(best) // 2], best[-1], target_k, True
    pool = [c for c in cands if c[2] >= min_cov] or cands
    b = min(pool, key=lambda c: (abs(c[3] - target_k), abs(math.log(max(c[1], 1e-6) / anchor_g))))
    return b[1], b[1], b[1], b[3], False


def _find_plateau(
    graph: ig.Graph,
    ptype: Any,
    target_k: int,
    grid: np.ndarray,
    min_cluster_size: int,
    min_cov: float,
    anchor_g: float,
    merge_small: bool,
    eval_starts: int,
    use_degree_volume: bool,
) -> tuple[float, float, float, int, bool]:
    coarse_starts = min(eval_starts, max(3, eval_starts // 2))
    cands = _eval_grid(graph, ptype, grid, min_cluster_size, coarse_starts, merge_small, use_degree_volume)
    g_min, g_mid, g_max, km, ok = _select_plateau(cands, target_k, min_cov, anchor_g)
    if ok:
        chk, _ = _run_seeds(graph, ptype, g_mid, 5, eval_starts, merge_small, min_cluster_size, use_degree_volume)
        if int(round(float(np.median([_major_cluster_stats(m, min_cluster_size)[0] for m in chk])))) == target_k:
            return g_min, g_mid, g_max, km, True

    # 1-step boundary-adaptive expansion triggered directly from grid boundary cluster counts
    if not ok:
        if cands[0][3] > target_k:
            ext = np.geomspace(max(1e-4, float(grid[0]) * 0.1), float(grid[0]), 25)
            c_ext = _eval_grid(graph, ptype, ext, min_cluster_size, eval_starts, merge_small, use_degree_volume)
            g_min2, g_mid2, g_max2, km2, ok2 = _select_plateau(c_ext, target_k, min_cov, anchor_g)
            if ok2 or abs(km2 - target_k) < abs(km - target_k):
                return g_min2, g_mid2, g_max2, km2, ok2
        elif cands[-1][3] < target_k:
            ext = np.geomspace(float(grid[-1]), float(grid[-1]) * 10.0, 25)
            c_ext = _eval_grid(graph, ptype, ext, min_cluster_size, eval_starts, merge_small, use_degree_volume)
            g_min2, g_mid2, g_max2, km2, ok2 = _select_plateau(c_ext, target_k, min_cov, anchor_g)
            if ok2 or abs(km2 - target_k) < abs(km - target_k):
                return g_min2, g_mid2, g_max2, km2, ok2

    non_shred = [c for c in cands if c[2] >= min_cov] or cands
    bidx = min(range(len(non_shred)), key=lambda i: (abs(non_shred[i][3] - target_k), abs(math.log(max(non_shred[i][1], 1e-6) / anchor_g))))
    lo_g = non_shred[max(0, bidx - 2)][1]
    hi_g = non_shred[min(len(non_shred) - 1, bidx + 2)][1]
    if hi_g <= lo_g:
        lo_g, hi_g = max(1e-3, lo_g * 0.5), hi_g * 2.0
    fine = _eval_grid(graph, ptype, np.geomspace(lo_g, hi_g, 30), min_cluster_size, eval_starts, merge_small, use_degree_volume)
    return _select_plateau(fine, target_k, min_cov, anchor_g)


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
    network_type: str = "bibliographic_coupling",
    frac_scheme: str = "vosviewer",
    denom_scope: str = "selected-core",
    event_degree_gini: float = 0.0,
    tls_core_turnover_jaccard: float | None = None,
    extra_warnings: list[str] | None = None,
    artifact: NetworkBuildArtifact | None = None,
    null_rewires: int = 0,
    null_starts: int = 3,
    null_seed: int = 20261122,
) -> CountingAuditResult:
    """Evaluate full vs. fractional counting sensitivity under exact matched-K plateau resolution."""
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective '{objective}'. Choose from: {', '.join(SUPPORTED_OBJECTIVES)}.")
    if n_seeds < 2:
        raise ValueError("n_seeds must be >= 2")
    if n_random_starts < 1:
        raise ValueError("n_random_starts must be >= 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if null_rewires < 0:
        raise ValueError("null_rewires must be >= 0")
    if null_rewires > 0 and null_starts < 1:
        raise ValueError("null_starts must be >= 1 when null_rewires > 0")

    lcc_idx, f_sub, r_sub = extract_shared_lcc_edges(len(node_records), full_edges, frac_edges)
    if len(lcc_idx) < max(3, min_cluster_size) or not f_sub or not r_sub:
        raise InsufficientNetworkError(
            f"Insufficient shared LCC size ({len(lcc_idx)} nodes, min_cluster_size={min_cluster_size}) or empty edges."
        )

    if objective == "VOS_AS_CPM":
        norm_mode = "association_strength"
        ptype = "CPM"
        use_deg_vol = False
        grid = np.geomspace(0.05, 5.0, grid_points)
    elif objective == "VOS_Frac_CPM":
        norm_mode = "fractionalization"
        ptype = "CPM"
        use_deg_vol = False
        grid = np.geomspace(0.05, 5.0, grid_points)
    else:  # Leiden_RBConfig (VOSviewer LinLog / modularity)
        norm_mode = "none"
        ptype = "modularity"
        use_deg_vol = True
        grid = np.linspace(0.10, 2.50, grid_points)

    sf = edges_to_igraph(len(lcc_idx), f_sub, normalization=norm_mode)
    sr = edges_to_igraph(len(lcc_idx), r_sub, normalization=norm_mode)
    lcc_records = [node_records[i] for i in lcc_idx]

    raw_f0, _ = _run_seeds(sf, ptype, resolution_default, min(5, n_seeds), min(3, n_random_starts), False, min_cluster_size, use_deg_vol)
    min_cov = max(0.55, min(0.80, float(np.median([_major_cluster_stats(m, min_cluster_size)[1] for m in raw_f0])) - 0.10))

    f_parts, f_qs = _run_seeds(sf, ptype, resolution_default, n_seeds, n_random_starts, merge_small_clusters, min_cluster_size, use_deg_vol)
    k_full = int(round(float(np.median([_major_cluster_stats(m, min_cluster_size)[0] for m in f_parts]))))
    deg_k = k_full < 2
    g_min, g_mid, g_max, _, _ = _find_plateau(
        sr, ptype, max(1, k_full), grid, min_cluster_size, min_cov, resolution_default, merge_small_clusters, n_random_starts, use_deg_vol
    )

    r_mid, r_qs = _run_seeds(sr, ptype, g_mid, n_seeds, n_random_starts, merge_small_clusters, min_cluster_size, use_deg_vol)
    k_frac = int(round(float(np.median([_major_cluster_stats(m, min_cluster_size)[0] for m in r_mid]))))
    exact_ok = (k_frac == k_full) and (not deg_k)

    # Full multi-perspective metrics (AMI, ARI, ECS, AECS, VI, NVI, per-node ECS)
    wf = mean_pairwise_metrics(f_parts, None)
    wr = mean_pairwise_metrics(r_mid, None)
    cx = mean_pairwise_metrics(f_parts, r_mid)

    # Reverse matching (match full counting onto default fractional K_frac)
    r_def, _ = _run_seeds(sr, ptype, resolution_default, n_seeds, n_random_starts, merge_small_clusters, min_cluster_size, use_deg_vol)
    k_r_def = max(1, int(round(float(np.median([_major_cluster_stats(m, min_cluster_size)[0] for m in r_def])))))
    _, g_rev, _, _, _ = _find_plateau(
        sf, ptype, k_r_def, grid, min_cluster_size, min_cov, resolution_default, merge_small_clusters, n_random_starts, use_deg_vol
    )
    f_rev, _ = _run_seeds(sf, ptype, g_rev, n_seeds, n_random_starts, merge_small_clusters, min_cluster_size, use_deg_vol)
    k_f_rev = int(round(float(np.median([_major_cluster_stats(m, min_cluster_size)[0] for m in f_rev]))))
    rev_ok = (k_f_rev == k_r_def) and (k_r_def >= 2)
    wf_rev = mean_pairwise_metrics(f_rev, None)["ami"]
    wr_def = mean_pairwise_metrics(r_def, None)["ami"]
    cx_rev = mean_pairwise_metrics(f_rev, r_def)["ami"]
    delta_rev = min(wf_rev, wr_def) - cx_rev

    # Hungarian core match + Landmarks + TF-IDF
    hung_vals = [hungarian_core_match(f_parts[i], r_mid[j])[0] for i in range(n_seeds) for j in range(n_seeds)]
    lm_vals, tf_vals = [], []
    for s in range(n_seeds):
        lm, tf = landmark_and_tfidf_retention(f_parts[s], r_mid[s], lcc_records, network_type=network_type, min_cluster_size=min_cluster_size)
        lm_vals.append(lm)
        if tf is not None:
            tf_vals.append(tf)

    # Consensus Medoid partitions + per-node ECS instability table (eliminating arbitrary seed-0 bias!)
    med_f_idx = select_medoid_partition(f_parts, f_qs)
    med_r_idx = select_medoid_partition(r_mid, r_qs)
    med_f = f_parts[med_f_idx]
    med_r_aligned, _ = align_fractional_to_full(med_f, r_mid[med_r_idx])

    node_rows: list[NodeStabilityRow] = []
    wf_node_ecs = wf["node_ecs"]
    wr_node_ecs = wr["node_ecs"]
    cx_node_ecs = cx["node_ecs"]
    for i, rec in enumerate(lcc_records):
        d_node = float(min(wf_node_ecs[i], wr_node_ecs[i]) - cx_node_ecs[i])
        switched = med_f[i] != med_r_aligned[i]
        node_rows.append(
            NodeStabilityRow(
                node_id=str(rec.get("id", f"node_{i}")),
                label=str(rec.get("label") or rec.get("title") or rec.get("id", f"node_{i}")),
                cited_by_count=int(rec.get("cited_by_count") or 0),
                occurrences=int(rec.get("occurrences") or 1),
                full_cluster_medoid=int(med_f[i]),
                frac_cluster_aligned=int(med_r_aligned[i]),
                switched_cluster=switched,
                ecs_within_full=round(float(wf_node_ecs[i]), 4),
                ecs_within_frac=round(float(wr_node_ecs[i]), 4),
                ecs_cross=round(float(cx_node_ecs[i]), 4),
                delta_ecs_node=round(d_node, 4),
                is_boundary_shifter=bool(switched or d_node >= 0.20),
            )
        )

    delta_mid = round(min(wf["ami"], wr["ami"]) - cx["ami"], 4)
    delta_aecs = round(min(wf["aecs"], wr["aecs"]) - cx["aecs"], 4)
    mean_hung = float(np.mean(hung_vals))
    reass = 1.0 - mean_hung
    lcc_frac = len(lcc_idx) / max(1, len(node_records))

    triage, guard_warns = classify_single_map_band(
        delta_mid,
        degenerate_k=deg_k,
        lcc_fraction=lcc_frac,
        min_within_ami=min(wf["ami"], wr["ami"]),
        core_reassignment=reass,
    )
    all_warnings = list(extra_warnings or []) + guard_warns

    null_dict: dict[str, Any] | None = None
    if null_rewires > 0 and artifact is not None:
        null_res: BipartiteNullResult = evaluate_bipartite_null(
            artifact, edges_to_igraph, n_rewires=null_rewires, n_starts=null_starts, seed=null_seed
        )
        null_dict = asdict(null_res)
        all_warnings.extend(null_res.warnings)

    return CountingAuditResult(
        network_type=network_type,
        objective=objective,
        frac_scheme=frac_scheme,
        denom_scope=denom_scope,
        merge_small_clusters=merge_small_clusters,
        min_cluster_size=min_cluster_size,
        n_random_starts=n_random_starts,
        n_seeds=n_seeds,
        n_core_selected=len(node_records),
        n_lcc=len(lcc_idx),
        lcc_fraction=round(lcc_frac, 4),
        event_degree_gini=round(event_degree_gini, 4),
        tls_core_turnover_jaccard=tls_core_turnover_jaccard,
        k_full_major=k_full,
        k_frac_matched_major=k_frac,
        exact_k_matched=exact_ok,
        rev_exact_k_matched=rev_ok,
        degenerate_k=deg_k,
        gamma_plateau_min=round(g_min, 4),
        gamma_plateau_mid=round(g_mid, 4),
        gamma_plateau_max=round(g_max, 4),
        ami_within_full=round(wf["ami"], 4),
        ami_within_frac=round(wr["ami"], 4),
        ami_cross_mid=round(cx["ami"], 4),
        delta_count_mid=delta_mid,
        delta_count_reverse=round(delta_rev, 4),
        ecs_within_full=round(wf["ecs"], 4),
        ecs_within_frac=round(wr["ecs"], 4),
        ecs_cross_mid=round(cx["ecs"], 4),
        aecs_within_full=round(wf["aecs"], 4),
        aecs_within_frac=round(wr["aecs"], 4),
        aecs_cross_mid=round(cx["aecs"], 4),
        delta_aecs_mid=delta_aecs,
        vi_cross_bits=round(cx["vi"], 4),
        nvi_cross=round(cx["nvi"], 4),
        ari_cross_mid=round(cx["ari"], 4),
        hungarian_match=round(mean_hung, 4),
        core_reassignment=round(reass, 4),
        landmark_top10_retention=round(float(np.mean(lm_vals)), 4),
        tfidf_term_jaccard=round(float(np.mean(tf_vals)), 4) if tf_vals else None,
        triage_band=triage,
        warnings=all_warnings,
        null_benchmark=null_dict,
        node_rows=node_rows,
        medoid_full_membership=med_f,
        medoid_frac_membership=med_r_aligned,
        lcc_full_edges=f_sub,
        lcc_frac_edges=r_sub,
    )


def audit_artifact(
    artifact: NetworkBuildArtifact,
    objective: str = "VOS_AS_CPM",
    n_seeds: int = 20,
    n_random_starts: int = 10,
    merge_small_clusters: bool = True,
    min_cluster_size: int = 5,
    null_rewires: int = 0,
    null_starts: int = 3,
    null_seed: int = 20261122,
) -> CountingAuditResult:
    """High-level wrapper evaluating a `NetworkBuildArtifact` end-to-end."""
    return evaluate_counting_sensitivity(
        node_records=artifact.node_records,
        full_edges=artifact.full_edges,
        frac_edges=artifact.frac_edges,
        objective=objective,
        n_seeds=n_seeds,
        n_random_starts=n_random_starts,
        merge_small_clusters=merge_small_clusters,
        min_cluster_size=min_cluster_size,
        network_type=artifact.network_type,
        frac_scheme=artifact.frac_scheme,
        denom_scope=artifact.denom_scope,
        event_degree_gini=artifact.event_degree_gini,
        tls_core_turnover_jaccard=artifact.tls_core_turnover_jaccard,
        extra_warnings=artifact.warnings,
        artifact=artifact,
        null_rewires=null_rewires,
        null_starts=null_starts,
        null_seed=null_seed,
    )
