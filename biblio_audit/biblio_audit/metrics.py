"""Mathematically verified partition-comparison and node-level stability metrics.

Implements:
- Chance-corrected Adjusted Mutual Information (`AMI`; Vinh, Epps & Bailey, 2010)
- Adjusted Rand Index (`ARI`; Hubert & Arabie, 1985)
- Element-Centric Similarity (`ECS`; Gates, Wood, Hetrick & Ahn, 2019 *Scientific Reports*)
  via exact O(N) closed-form disjoint PPR-L1 identity: S(i) = |c_i^A ∩ c_i^B| / max(|c_i^A|, |c_i^B|)
- Exact closed-form hypergeometric permutation **Chance-Adjusted ECS (`AECS`)** removing small-K
  baseline inflation (Gates & Ahn, 2017 *JMLR*)
- Metric Variation of Information (`VI` in bits & Normalized `NVI = VI / log2(N)`; Meilă, 2007)
- Optimal one-to-one Hungarian cluster alignment with deterministic surplus-cluster ID mapping
- Medoid consensus partition selector (eliminating arbitrary seed-0 bias)
- Symmetrized Top-10 landmark retention and BC nonzero Top-15 TF-IDF label Jaccard
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari


def element_centric_scores(m1: list[int], m2: list[int]) -> np.ndarray:
    """Exact O(N) Element-Centric Similarity S(i; A, B) for disjoint partitions (Gates et al., 2019).

    Algebraically identical to `clusim.sim.element_sim_elscore(c1, c2, alpha)` for any `alpha in (0, 1]`
    because `(1 - alpha)` cancels on diagonal terms and `alpha` cancels with `1 / (2 * alpha)` in `cL1`.
    """
    n = len(m1)
    if n != len(m2):
        raise ValueError("Partitions must have identical length")
    if n == 0:
        return np.array([], dtype=float)

    c1 = Counter(m1)
    c2 = Counter(m2)
    inter = Counter(zip(m1, m2))
    out = np.empty(n, dtype=float)
    for i, (a, b) in enumerate(zip(m1, m2)):
        out[i] = inter[(a, b)] / max(c1[a], c2[b])
    return out


def element_centric_similarity(m1: list[int], m2: list[int]) -> float:
    """Global Element-Centric Similarity (mean of element scores S(i))."""
    scores = element_centric_scores(m1, m2)
    return float(np.mean(scores)) if scores.size else 1.0


def expected_ecs_fixed_marginals(m1: list[int], m2: list[int]) -> float:
    """Exact O(K1 * K2) closed-form permutation expectation E_pi[ECS(A, B)] under fixed cluster sizes.

    Since sum_i S(i) = sum_{r,c} O_{rc}^2 / max(a_r, b_c) and O_{rc} ~ Hypergeometric(N, a_r, b_c),
    E[O_{rc}^2] = Var(O_{rc}) + (E[O_{rc}])^2 gives the exact fixed-marginals null baseline with
    zero Monte Carlo variance.
    """
    n = len(m1)
    if n != len(m2):
        raise ValueError("Partitions must have identical length")
    if n <= 1:
        return 1.0
    sizes1 = list(Counter(m1).values())
    sizes2 = list(Counter(m2).values())
    total = 0.0
    denom_var = float(n * n * (n - 1))
    for a in sizes1:
        for b in sizes2:
            mean = (a * b) / float(n)
            var = (a * b * (n - a) * (n - b)) / denom_var
            e_sq = var + mean * mean
            total += e_sq / float(max(a, b))
    return float(min(1.0, total / float(n)))


def adjusted_element_centric_similarity(m1: list[int], m2: list[int]) -> float:
    """Chance-adjusted Element-Centric Similarity: (ECS - E_pi[ECS]) / (1 - E_pi[ECS])."""
    ecs = element_centric_similarity(m1, m2)
    e_ecs = expected_ecs_fixed_marginals(m1, m2)
    if e_ecs >= 1.0 - 1e-12:
        return 1.0 if ecs >= 1.0 - 1e-12 else 0.0
    return float((ecs - e_ecs) / (1.0 - e_ecs))


def variation_of_information(m1: list[int], m2: list[int]) -> tuple[float, float]:
    """Compute metric Variation of Information VI (bits) and normalized NVI = VI / log2(N) (Meilă, 2007)."""
    n = len(m1)
    if n != len(m2):
        raise ValueError("Partitions must have identical length")
    if n <= 1:
        return 0.0, 0.0

    c1 = Counter(m1)
    c2 = Counter(m2)
    joint = Counter(zip(m1, m2))

    h1 = -sum((v / n) * math.log2(v / n) for v in c1.values())
    h2 = -sum((v / n) * math.log2(v / n) for v in c2.values())
    mi = 0.0
    for (a, b), v in joint.items():
        p_ab = v / n
        mi += p_ab * math.log2((p_ab) / ((c1[a] / n) * (c2[b] / n)))
    vi = max(0.0, h1 + h2 - 2.0 * mi)
    nvi = min(1.0, vi / math.log2(n))
    return float(vi), float(nvi)


def select_medoid_partition(parts: list[list[int]], qualities: list[float] | None = None) -> int:
    """Select the consensus Medoid seed index maximizing mean pairwise AMI to all other runs.

    Eliminates arbitrary seed-0 bias when exporting node-level cluster memberships; tie-breaks
    by highest partition quality and then lowest seed index.
    """
    n = len(parts)
    if n <= 1:
        return 0
    best_idx = 0
    best_key = (-1.0, -float("inf"), 0)
    for i in range(n):
        mean_a = float(np.mean([ami(parts[i], parts[j]) for j in range(n) if j != i]))
        q = float(qualities[i]) if qualities is not None and i < len(qualities) else 0.0
        key = (round(mean_a, 8), round(q, 8), -i)
        if key > best_key:
            best_key = key
            best_idx = i
    return best_idx


def hungarian_core_match(m1: list[int], m2: list[int]) -> tuple[float, dict[int, int]]:
    """Optimal one-to-one Hungarian cluster alignment mapping m1 cluster IDs -> m2 cluster IDs."""
    if len(m1) != len(m2):
        raise ValueError("Partitions must have identical length")
    if not m1:
        return 1.0, {}
    c1, c2 = sorted(set(m1)), sorted(set(m2))
    idx1, idx2 = {c: i for i, c in enumerate(c1)}, {c: i for i, c in enumerate(c2)}
    mat = np.zeros((len(c1), len(c2)), dtype=int)
    for a, b in zip(m1, m2):
        mat[idx1[a], idx2[b]] += 1
    row_ind, col_ind = linear_sum_assignment(-mat)
    matched = int(mat[row_ind, col_ind].sum())
    return matched / len(m1), {c1[r]: c2[c] for r, c in zip(row_ind, col_ind)}


def align_fractional_to_full(m_full: list[int], m_frac: list[int]) -> tuple[list[int], dict[int, int]]:
    """Relabel fractional cluster IDs onto full cluster IDs via Hungarian matching, mapping surplus IDs deterministically."""
    _, full_to_frac = hungarian_core_match(m_full, m_frac)
    frac_to_full: dict[int, int] = {fc: uc for uc, fc in full_to_frac.items()}
    next_id = (max(m_full) + 1) if m_full else 0
    for fc in sorted(set(m_frac)):
        if fc not in frac_to_full:
            frac_to_full[fc] = next_id
            next_id += 1
    return [frac_to_full[c] for c in m_frac], frac_to_full


def _node_rank_score(rec: dict[str, Any], network_type: str) -> tuple[int, int]:
    if network_type in {"co_authorship", "keyword_cooccurrence"}:
        return (int(rec.get("occurrences") or 0), int(rec.get("cited_by_count") or 0))
    return (int(rec.get("cited_by_count") or 0), int(rec.get("occurrences") or 0))


def _one_way_landmark_retention(
    m_src: list[int],
    m_dst: list[int],
    node_records: list[dict[str, Any]],
    network_type: str,
    min_cluster_size: int,
    top_k: int = 10,
) -> float:
    if not m_src:
        return 1.0
    _, mapping = hungarian_core_match(m_src, m_dst)
    src_nodes: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(m_src):
        src_nodes[c].append(i)
    major_src = [c for c, nodes in src_nodes.items() if len(nodes) >= min_cluster_size] or list(src_nodes.keys())
    scores, weights = [], []
    for c in major_src:
        nodes = src_nodes[c]
        ranked = sorted(nodes, key=lambda i: _node_rank_score(node_records[i], network_type), reverse=True)
        top_nodes = ranked[: min(top_k, len(ranked))]
        target_c = mapping.get(c)
        scores.append(sum(1 for i in top_nodes if m_dst[i] == target_c) / len(top_nodes))
        weights.append(len(nodes))
    return float(np.average(scores, weights=weights))


def landmark_and_tfidf_retention(
    m1: list[int],
    m2: list[int],
    node_records: list[dict[str, Any]],
    network_type: str = "bibliographic_coupling",
    min_cluster_size: int = 5,
) -> tuple[float, float | None]:
    """Symmetrized Top-10 landmark retention + nonzero Top-15 TF-IDF label Jaccard (BC titles only)."""
    lm = 0.5 * (
        _one_way_landmark_retention(m1, m2, node_records, network_type, min_cluster_size, 10)
        + _one_way_landmark_retention(m2, m1, node_records, network_type, min_cluster_size, 10)
    )
    if network_type != "bibliographic_coupling":
        return lm, None

    texts = [
        str(r.get("title") or "").strip() if not str(r.get("title") or "").strip().startswith("http") else ""
        for r in node_records
    ]
    if sum(1 for t in texts if len(t) > 10) < max(10, len(node_records) // 2):
        return lm, None

    _, mapping = hungarian_core_match(m1, m2)
    c1_nodes: dict[int, list[int]] = defaultdict(list)
    c2_nodes: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(m1):
        c1_nodes[c].append(i)
    for i, c in enumerate(m2):
        c2_nodes[c].append(i)
    major_c1 = [c for c, nodes in c1_nodes.items() if len(nodes) >= min_cluster_size] or list(c1_nodes.keys())
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
        x_tf = vec.fit_transform(texts)
        vocab = np.asarray(vec.get_feature_names_out())
        if len(vocab) < 5:
            return lm, None
        scores, weights = [], []
        for c in major_c1:
            tc = mapping.get(c)
            if tc is None or not c2_nodes.get(tc):
                scores.append(0.0)
                weights.append(len(c1_nodes[c]))
                continue
            v1 = np.asarray(x_tf[c1_nodes[c]].mean(axis=0)).ravel()
            v2 = np.asarray(x_tf[c2_nodes[tc]].mean(axis=0)).ravel()
            i1 = [i for i in np.argsort(v1)[-15:] if v1[i] > 0.0]
            i2 = [i for i in np.argsort(v2)[-15:] if v2[i] > 0.0]
            if not i1 and not i2:
                continue
            t1, t2 = set(vocab[i1]), set(vocab[i2])
            scores.append(len(t1 & t2) / max(1, len(t1 | t2)))
            weights.append(len(c1_nodes[c]))
        return lm, (float(np.average(scores, weights=weights)) if scores else None)
    except ValueError:
        return lm, None


def mean_pairwise_metrics(
    parts_a: list[list[int]],
    parts_b: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Compute within-rule or cross-rule AMI, ARI, ECS, AECS, VI, NVI, and per-node mean ECS."""
    is_within = parts_b is None
    b_list = parts_a if is_within else parts_b
    assert b_list is not None
    n_a, n_b = len(parts_a), len(b_list)
    n_nodes = len(parts_a[0]) if parts_a else 0

    amis, aris, ecss, aecss, vis, nvis = [], [], [], [], [], []
    node_ecs_sum = np.zeros(n_nodes, dtype=float)
    pair_count = 0

    for i in range(n_a):
        j_start = (i + 1) if is_within else 0
        for j in range(j_start, n_b):
            pa, pb = parts_a[i], b_list[j]
            amis.append(float(ami(pa, pb)))
            if not is_within:
                aris.append(float(ari(pa, pb)))
                vi_v, nvi_v = variation_of_information(pa, pb)
                vis.append(vi_v)
                nvis.append(nvi_v)
            el = element_centric_scores(pa, pb)
            node_ecs_sum += el
            e_val = float(np.mean(el)) if el.size else 1.0
            ecss.append(e_val)
            e_exp = expected_ecs_fixed_marginals(pa, pb)
            aecss.append(1.0 if (e_exp >= 1.0 - 1e-12 and e_val >= 1.0 - 1e-12) else (0.0 if e_exp >= 1.0 - 1e-12 else (e_val - e_exp) / (1.0 - e_exp)))
            pair_count += 1

    if pair_count == 0:
        return {
            "ami": 1.0,
            "ari": 1.0,
            "ecs": 1.0,
            "aecs": 1.0,
            "vi": 0.0,
            "nvi": 0.0,
            "node_ecs": np.ones(n_nodes, dtype=float),
        }

    return {
        "ami": float(np.mean(amis)),
        "ari": float(np.mean(aris)) if aris else 1.0,
        "ecs": float(np.mean(ecss)),
        "aecs": float(np.mean(aecss)),
        "vi": float(np.mean(vis)) if vis else 0.0,
        "nvi": float(np.mean(nvis)) if nvis else 0.0,
        "node_ecs": node_ecs_sum / float(pair_count),
    }
