"""Degree-preserving bipartite null models for bibliometric network validation.

Implements two mathematically rigorous bipartite randomization algorithms with strict
Phipson & Smyth (2010) Monte Carlo exchangeability (`p = (1 + #{Q_null >= Q_obs}) / (R + 1)`):
1. **Causal Temporal Bipartite Edge-Swap MCMC** (Maslov & Sneppen, 2002; Fosdick et al., 2018):
   Preserves exact row and column degrees while enforcing causal chronology (`t_cited <= t_citing`)
   for citation networks (`bibliographic_coupling`, `co_citation`). Because standard offline exports
   lack external publication-year metadata for cited references, standalone `biblio-audit` proxies
   `t_cited` by the conservative upper bound `min(citing_year in core)` (`t_cited_true <= t_earliest_citing`),
   emitting `LowMixingWarning` if narrow corpus publication spans constrain swap acceptance below 5%.
2. **Minimum-Dimension Curveball Algorithm** (Strona et al., 2014 *Nature Communications*):
   Fast, unbiased degree-preserving binary incidence randomization for non-temporal bipartite
   networks (`co_authorship`, `keyword_cooccurrence`, or citation corpora lacking years),
   transposing automatically to `min(R, C)` with `max(20 * min(R, C), 5000)` trade steps.
"""

from __future__ import annotations

import random
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

from .networks import NetworkBuildArtifact

_IGRAPH_RNG_LOCK = threading.Lock()


def _leiden_null_partition(
    graph: ig.Graph,
    objective_fn: str,
    seed: int,
    max_iterations: int = 10,
) -> tuple[list[int], float]:
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
                    resolution=1.0,
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


@dataclass
class BipartiteNullResult:
    """Exchangeable bipartite null-model benchmark results."""

    method: str  # "causal_temporal_mcmc" or "curveball"
    n_rewires: int
    n_starts_per_eval: int
    q_obs_rb_fixed_n: float
    q_null_rb_mean: float
    q_null_rb_p95: float
    p_value_rb: float
    passes_null_rb: bool
    q_obs_vos_norm: float
    q_null_vos_norm_mean: float
    q_null_vos_norm_p95: float
    p_value_vos: float
    passes_null_vos: bool
    ami_obs_vs_null_mean: float | None
    mean_null_lcc_ratio: float
    mean_effective_swap_rate: float
    warnings: list[str]


def curveball_rewire(
    bip_edges: list[tuple[int, int]],
    n_steps: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Unbiased Curveball randomization (Strona et al., 2014) oriented along `min(R, C)`."""
    if not bip_edges:
        return [], 0.0
    gen = rng if rng is not None else np.random.default_rng(20261122)

    rows = sorted({u for u, _ in bip_edges})
    cols = sorted({v for _, v in bip_edges})
    if len(rows) < 2 or len(cols) < 2:
        return list(bip_edges), 0.0

    # Orient along smaller dimension so pairwise trades hit overlapping neighborhoods efficiently
    transpose = len(cols) < len(rows)
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in bip_edges:
        if transpose:
            adj[v].add(u)
        else:
            adj[u].add(v)

    keys = sorted(adj.keys())
    m_small = len(keys)
    steps = int(n_steps) if n_steps is not None else max(20 * m_small, 5000)

    n_effective = 0
    for _ in range(steps):
        i_idx = int(gen.integers(0, m_small))
        j_idx = int(gen.integers(0, m_small - 1))
        if j_idx >= i_idx:
            j_idx += 1
        ka, kb = keys[i_idx], keys[j_idx]
        sa, sb = adj[ka], adj[kb]
        a_only = sa - sb
        if not a_only:
            continue
        b_only = sb - sa
        if not b_only:
            continue

        pool = list(a_only) + list(b_only)
        gen.shuffle(pool)
        na = len(a_only)
        new_a_only = set(pool[:na])
        if new_a_only != a_only:
            n_effective += 1
            shared = sa & sb
            adj[ka] = shared | new_a_only
            adj[kb] = shared | set(pool[na:])

    out: list[tuple[int, int]] = []
    for k, nbrs in adj.items():
        for nb in nbrs:
            out.append((nb, k) if transpose else (k, nb))
    return out, n_effective / max(1, steps)


def causal_temporal_bipartite_rewire(
    bip_edges: list[tuple[int, int]],
    node_years: dict[int, int],
    event_years: dict[int, int],
    network_type: str,
    swaps_per_edge: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Degree-preserving causal temporal edge swap enforcing `t_cited <= t_citing`."""
    edges = list(dict.fromkeys(bip_edges))
    if len(edges) < 2:
        return edges, 0.0
    gen = rng if rng is not None else np.random.default_rng(20261122)

    edge_set = set(edges)
    m = len(edges)
    n_attempts = max(1, swaps_per_edge * m)
    accepted = 0

    for _ in range(n_attempts):
        i = int(gen.integers(0, m))
        j = int(gen.integers(0, m - 1))
        if j >= i:
            j += 1
        n1, e1 = edges[i]
        n2, e2 = edges[j]
        if n1 == n2 or e1 == e2:
            continue
        if (n1, e2) in edge_set or (n2, e1) in edge_set:
            continue
        # Causal chronology:
        # - BC: node = citing paper, event = cited ref -> event_year <= node_year
        # - CC: node = cited ref, event = citing paper -> node_year <= event_year
        if network_type == "bibliographic_coupling":
            if event_years.get(e2, 0) > node_years.get(n1, 9999) or event_years.get(e1, 0) > node_years.get(n2, 9999):
                continue
        else:
            if node_years.get(n1, 0) > event_years.get(e2, 9999) or node_years.get(n2, 0) > event_years.get(e1, 9999):
                continue

        edge_set.remove((n1, e1))
        edge_set.remove((n2, e2))
        edge_set.add((n1, e2))
        edge_set.add((n2, e1))
        edges[i] = (n1, e2)
        edges[j] = (n2, e1)
        accepted += 1

    return edges, accepted / float(n_attempts)


def _project_fractional_from_bipartite(
    bip_edges: list[tuple[int, int]],
    frac_scheme: str = "vosviewer",
    event_degrees: dict[int, int] | None = None,
    denom_scope: str = "selected-core",
) -> dict[tuple[int, int], float]:
    groups: dict[int, list[int]] = defaultdict(list)
    for node_i, ev_i in bip_edges:
        groups[ev_i].append(node_i)
    proj: dict[tuple[int, int], float] = defaultdict(float)
    for ev_i, nlist in groups.items():
        if len(nlist) < 2:
            continue
        if denom_scope == "full-corpus" and event_degrees is not None and ev_i in event_degrees:
            dk = max(len(nlist), int(event_degrees[ev_i]))
        else:
            dk = len(nlist)
        w = (1.0 / (dk - 1.0)) if frac_scheme == "vosviewer" else (2.0 / (dk * (dk - 1.0)))
        for a in range(len(nlist)):
            u = nlist[a]
            for b in range(a + 1, len(nlist)):
                v = nlist[b]
                key = (u, v) if u < v else (v, u)
                proj[key] += w
    return dict(proj)


def evaluate_bipartite_null(
    artifact: NetworkBuildArtifact,
    edges_to_igraph_fn: Any,
    n_rewires: int = 99,
    n_starts: int = 3,
    seed: int = 20261122,
) -> BipartiteNullResult:
    """Run exchangeable Phipson-Smyth bipartite null validation on both fixed-N RBConfig & LCC VOS_AS_CPM."""
    if n_rewires < 1:
        raise ValueError("n_rewires must be >= 1")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    n_nodes = len(artifact.node_records)
    # Require complete real publication years on citation networks; otherwise fall back cleanly to Curveball
    all_nodes_dated = len(artifact.node_years) == n_nodes and n_nodes > 0
    unique_events = {e for _, e in artifact.bipartite_edges}
    all_events_dated = len(unique_events) > 0 and all(e in artifact.event_years for e in unique_events)
    warnings: list[str] = []
    if artifact.network_type in {"bibliographic_coupling", "co_citation"} and all_nodes_dated and all_events_dated:
        use_causal = True
    else:
        use_causal = False
        if artifact.network_type in {"bibliographic_coupling", "co_citation"}:
            warnings.append("MissingYearsFallbackWarning: incomplete publication years in citation corpus; used Curveball bipartite null instead of causal temporal MCMC.")
    method = "causal_temporal_mcmc" if use_causal else "curveball"

    ss = np.random.SeedSequence(seed)
    obs_seed_seq, *rep_seed_seqs = ss.spawn(1 + n_rewires)

    def _eval_one(b_edges: list[tuple[int, int]], child_ss: np.random.SeedSequence) -> tuple[float, float, list[int], list[int]]:
        proj = _project_fractional_from_bipartite(
            b_edges,
            frac_scheme=artifact.frac_scheme,
            event_degrees=artifact.event_degrees,
            denom_scope=artifact.denom_scope,
        )
        if not proj:
            return 0.0, 0.0, [], []
        g_all = edges_to_igraph_fn(n_nodes, proj, normalization="none")
        start_seeds = [int(s) for s in child_ss.generate_state(2 * n_starts)]

        # 1. Fixed-N_core RBConfiguration modularity (immune to LCC expansion/collapse artifacts!)
        best_q_rb = -float("inf")
        for s_i in range(n_starts):
            mem_rb, _ = _leiden_null_partition(g_all, "modularity", start_seeds[s_i])
            qv = float(g_all.modularity(mem_rb, weights="weight"))
            if qv > best_q_rb:
                best_q_rb = qv

        # 2. Scale-invariant pre-merge VOS_AS_CPM quality on LCC: Waltman et al. (2010) Eq. 7
        comps = g_all.connected_components()
        lcc = sorted(comps[max(range(len(comps)), key=lambda i: len(comps[i]))])
        old_to_new = {old: new for new, old in enumerate(lcc)}
        sub_proj = {(old_to_new[u], old_to_new[v]): w for (u, v), w in proj.items() if u in old_to_new and v in old_to_new}
        if not sub_proj:
            return best_q_rb, 0.0, lcc, [0] * len(lcc)
        g_vos = edges_to_igraph_fn(len(lcc), sub_proj, normalization="association_strength")
        best_q_vos = -float("inf")
        best_mem: list[int] = []
        for s_i in range(n_starts):
            mem_vos, q_vos = _leiden_null_partition(g_vos, "CPM", start_seeds[n_starts + s_i])
            if q_vos > best_q_vos:
                best_q_vos = q_vos
                best_mem = mem_vos
        return best_q_rb, best_q_vos, lcc, best_mem

    q_obs_rb, q_obs_vos, obs_lcc, obs_mem = _eval_one(artifact.bipartite_edges, obs_seed_seq)
    obs_map = dict(zip(obs_lcc, obs_mem))
    obs_lcc_set = set(obs_lcc)

    null_rbs: list[float] = []
    null_voss: list[float] = []
    null_amis: list[float] = []
    lcc_ratios: list[float] = []
    swap_rates: list[float] = []
    n_low_overlap = 0
    min_common_req = min(5, max(2, len(obs_lcc)))

    for rep_ss in rep_seed_seqs:
        rw_ss, eval_ss = rep_ss.spawn(2)
        rng = np.random.default_rng(rw_ss)
        if use_causal:
            rw_edges, rate = causal_temporal_bipartite_rewire(
                artifact.bipartite_edges, artifact.node_years, artifact.event_years, artifact.network_type, rng=rng
            )
        else:
            rw_edges, rate = curveball_rewire(artifact.bipartite_edges, rng=rng)
        swap_rates.append(rate)
        q_rb, q_vos, n_lcc, n_mem = _eval_one(rw_edges, eval_ss)
        null_rbs.append(q_rb)
        null_voss.append(q_vos)
        lcc_ratios.append(len(n_lcc) / max(1, len(obs_lcc)))

        common = sorted(obs_lcc_set & set(n_lcc))
        if len(obs_lcc) > 0 and (len(common) / len(obs_lcc)) >= 0.50 and len(common) >= min_common_req:
            n_map = dict(zip(n_lcc, n_mem))
            null_amis.append(float(ami([obs_map[u] for u in common], [n_map[u] for u in common])))
        else:
            n_low_overlap += 1

    mean_swap = float(np.mean(swap_rates)) if swap_rates else 0.0
    mean_lcc_ratio = float(np.mean(lcc_ratios)) if lcc_ratios else 1.0
    if mean_swap < 0.05:
        warnings.append(f"LowMixingWarning: mean effective bipartite swap rate={mean_swap:.3f} (< 0.05); degree sequence is highly constrained.")
    if mean_lcc_ratio > 1.50 or mean_lcc_ratio < 0.65 or n_low_overlap > n_rewires // 2:
        warnings.append(
            f"LCC_SHIFT_WARNING: null rewiring shifted LCC size by {mean_lcc_ratio:.2f}x; rely on fixed-N RBConfig null p-value rather than LCC VOS quality."
        )

    p_rb = float((1 + sum(1 for v in null_rbs if v >= q_obs_rb)) / (n_rewires + 1))
    p_vos = float((1 + sum(1 for v in null_voss if v >= q_obs_vos)) / (n_rewires + 1))
    p95_rb = float(np.percentile(null_rbs, 95)) if null_rbs else 0.0
    p95_vos = float(np.percentile(null_voss, 95)) if null_voss else 0.0

    return BipartiteNullResult(
        method=method,
        n_rewires=n_rewires,
        n_starts_per_eval=n_starts,
        q_obs_rb_fixed_n=round(q_obs_rb, 4),
        q_null_rb_mean=round(float(np.mean(null_rbs)), 4),
        q_null_rb_p95=round(p95_rb, 4),
        p_value_rb=round(p_rb, 4),
        passes_null_rb=bool(p_rb <= 0.05),
        q_obs_vos_norm=round(q_obs_vos, 4),
        q_null_vos_norm_mean=round(float(np.mean(null_voss)), 4),
        q_null_vos_norm_p95=round(p95_vos, 4),
        p_value_vos=round(p_vos, 4),
        passes_null_vos=bool(p_vos <= 0.05),
        ami_obs_vs_null_mean=round(float(np.mean(null_amis)), 4) if (null_amis and n_low_overlap <= n_rewires // 2) else None,
        mean_null_lcc_ratio=round(mean_lcc_ratio, 3),
        mean_effective_swap_rate=round(mean_swap, 4),
        warnings=warnings,
    )
