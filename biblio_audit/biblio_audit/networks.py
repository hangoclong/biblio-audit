"""Unified bipartite projection engine across all four VOSviewer bibliometric network families.

Implements Perianes-Rodriguez, Waltman & van Eck (2016) Eq. (1)-(15) & Table 1 full vs.
action-normalized fractional counting (`1 / (d_e - 1)`), plus optional Leydesdorff & Park (2017)
document-normalized refined fractional weighting (`2 / (d_e * (d_e - 1))`) across:
- `bibliographic_coupling` (`bc`): documents linked via shared cited references
- `co_citation` (`cc`): cited references linked via citing documents
- `co_authorship` (`ca`): authors linked via multi-authored documents (pre-truncation document degree!)
- `keyword_cooccurrence` (`kw`): keywords linked via co-occurring documents (pre-truncation document degree!)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import InsufficientNetworkError, MissingFieldError

NETWORK_ALIASES = {
    "bc": "bibliographic_coupling",
    "bibliographic_coupling": "bibliographic_coupling",
    "cc": "co_citation",
    "co_citation": "co_citation",
    "ca": "co_authorship",
    "co_authorship": "co_authorship",
    "kw": "keyword_cooccurrence",
    "keyword_cooccurrence": "keyword_cooccurrence",
}


@dataclass
class NetworkBuildArtifact:
    """Complete bipartite & projected edge representation for auditing and null modeling."""

    network_type: str
    frac_scheme: str
    denom_scope: str
    node_records: list[dict[str, Any]]
    full_edges: dict[tuple[int, int], float]
    frac_edges: dict[tuple[int, int], float]
    bipartite_edges: list[tuple[int, int]]  # (node_idx, event_idx)
    event_degrees: dict[int, int]  # full event degree d_e for denominator parity inside nulls
    node_years: dict[int, int]
    event_years: dict[int, int]
    n_eligible_nodes: int
    n_core_selected: int
    event_degree_gini: float
    tls_core_turnover_jaccard: float | None
    warnings: list[str]


def canonical_network_type(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in NETWORK_ALIASES:
        raise ValueError(f"Unsupported network_type '{name}'. Choose from: bc, cc, ca, kw.")
    return NETWORK_ALIASES[key]


def gini_coefficient(values: list[int | float]) -> float:
    """Compute Gini concentration coefficient over positive event degrees (d_e >= 2)."""
    arr = np.asarray([v for v in values if v >= 2], dtype=float)
    if arr.size < 2 or float(np.sum(arr)) <= 0.0:
        return 0.0
    arr.sort()
    n = arr.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1.0) / n)


def reference_degree_gini(works: list[dict[str, Any]]) -> float:
    """Compute Gini concentration of cited-reference frequencies (n_k >= 2) across the corpus."""
    counts: dict[str, int] = defaultdict(int)
    for w in works:
        for r in set(w.get("referenced_works") or []):
            counts[r] += 1
    return gini_coefficient(list(counts.values()))


def _frac_weight(d_e: int, frac_scheme: str) -> float:
    if d_e < 2:
        return 0.0
    if frac_scheme == "vosviewer":
        return 1.0 / (d_e - 1.0)
    if frac_scheme == "refined":
        return 2.0 / (d_e * (d_e - 1.0))
    raise ValueError(f"Unsupported frac_scheme '{frac_scheme}'. Choose 'vosviewer' or 'refined'.")


def build_coupling_matrices(
    c_mat: np.ndarray,
    network_type: str = "bibliographic_coupling",
    frac_scheme: str = "vosviewer",
) -> tuple[np.ndarray, np.ndarray]:
    """Dense matrix implementation of PWvE (2016) Eq. (1)-(15) full vs. fractional counting.

    For `bibliographic_coupling`, `co_authorship`, and `keyword_cooccurrence`: rows are nodes,
    columns are events (references or documents). For `co_citation`: rows are documents and
    columns are cited references (transposed internally so nodes are cited references).
    """
    arr = np.asarray(c_mat, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("c_mat must be a non-empty 2D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("c_mat entries must be finite (no NaN or Inf)")
    if np.any(arr < 0):
        raise ValueError("c_mat entries must be non-negative")
    nt = canonical_network_type(network_type)
    if nt == "co_citation":
        arr = arr.T

    col_totals = arr.sum(axis=0)
    keep = col_totals > 1.0
    if not np.any(keep):
        n = arr.shape[0]
        return np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float)

    c_keep = arr[:, keep]
    dk = col_totals[keep]
    v_full = c_keep @ c_keep.T
    if frac_scheme == "vosviewer":
        inv = 1.0 / (dk - 1.0)
    elif frac_scheme == "refined":
        inv = 2.0 / (dk * (dk - 1.0))
    else:
        raise ValueError(f"Unsupported frac_scheme: {frac_scheme}")
    v_frac = (c_keep * inv) @ c_keep.T
    np.fill_diagonal(v_full, 0.0)
    np.fill_diagonal(v_frac, 0.0)
    return v_full, v_frac


def _apply_thesaurus(items: list[str], thesaurus: dict[str, str | None] | None) -> list[str]:
    if not thesaurus:
        return list(dict.fromkeys(items))
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        low = it.strip().lower()
        if low in thesaurus:
            rep = thesaurus[low]
            if rep is None:
                continue  # explicit VOSviewer ignore
            low = rep
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return out


def build_network_artifact(
    works: list[dict[str, Any]],
    network_type: str = "bibliographic_coupling",
    core_top_n: int | None = 150,
    min_occurrences: int = 1,
    core_selection: str = "tls_within_network",
    denom_scope: str = "auto",
    frac_scheme: str = "vosviewer",
    max_authors: int = 25,
    keyword_source: str = "author",
    thesaurus: dict[str, str | None] | None = None,
) -> NetworkBuildArtifact:
    """Build exact full & fractional edge lists + bipartite incidence metadata across BC, CC, CA, KW.

    Denominator scope (`denom_scope`):
    - `'auto'`: mandates `'full-corpus'` pre-truncation document degree for `CA` and `KW`
      (matching VOSviewer Manual 1.6.20 pp. 34-35), and `'selected-core'` induced subgraph
      degree for `BC` and `CC` (matching Perianes-Rodriguez et al. 2016 Eq. 5/10/15 matrix algebra).
    - `'full-corpus'`: computes event denominator `d_e - 1` across the entire eligible corpus
      BEFORE `min_occurrences`/`core_top_n` node truncation for all four network families.
    - `'selected-core'`: computes `d_e - 1` on the selected core subgraph (emits a warning on CA/KW;
      retained for explicit sensitivity comparison).
    """
    nt = canonical_network_type(network_type)
    if core_top_n is not None:
        if int(core_top_n) < 0:
            raise ValueError(f"core_top_n must be >= 0 (0 or None for unlimited), got {core_top_n}.")
        core_top_n = None if int(core_top_n) == 0 else int(core_top_n)
    if int(max_authors) < 0:
        raise ValueError(f"max_authors must be >= 0 (0 = disabled), got {max_authors}.")
    if denom_scope not in {"auto", "full-corpus", "selected-core"}:
        raise ValueError(f"Unsupported denom_scope '{denom_scope}'. Choose: auto, full-corpus, selected-core.")
    eff_scope = ("full-corpus" if nt in {"co_authorship", "keyword_cooccurrence"} else "selected-core") if denom_scope == "auto" else denom_scope

    warnings: list[str] = []
    if nt in {"co_authorship", "keyword_cooccurrence"} and eff_scope == "selected-core":
        warnings.append(
            f"DenomScopeWarning: `selected-core` denominator on `{nt}` inflates fractional weights for multi-item documents partially truncated by `top_n`; prefer `full-corpus`."
        )

    if nt == "bibliographic_coupling":
        if not any(w.get("referenced_works") for w in works):
            raise MissingFieldError("No cited references found (`referenced_works` / `References` / `CR`). Cannot build bibliographic coupling network.")
        # Full-corpus reference frequencies (needed for full-corpus denom_scope and corpus Gini)
        full_ref_freq: dict[str, int] = defaultdict(int)
        for w in works:
            for r in set(w.get("referenced_works") or []):
                full_ref_freq[r] += 1
        gini_val = gini_coefficient(list(full_ref_freq.values()))

        # Apply min_occurrences citation filter BEFORE eligibility & TLS calculation
        cand_works = [w for w in works if int(w.get("cited_by_count") or 0) >= max(0, int(min_occurrences))] if min_occurrences > 1 else list(works)
        cand_ref_freq: dict[str, int] = defaultdict(int)
        for w in cand_works:
            for r in set(w.get("referenced_works") or []):
                cand_ref_freq[r] += 1
        eligible = [w for w in cand_works if any(cand_ref_freq[r] >= 2 for r in set(w.get("referenced_works") or []))]
        n_eligible = len(eligible)
        if n_eligible < 2:
            raise InsufficientNetworkError(f"Only {n_eligible} eligible documents for bibliographic coupling after filtering.")

        tls_turnover: float | None = None
        if core_top_n is not None and len(eligible) > core_top_n:
            scored_full = []
            scored_frac = []
            for idx, w in enumerate(eligible):
                refs_u = set(w.get("referenced_works") or [])
                tls_f = sum(cand_ref_freq[r] - 1 for r in refs_u if cand_ref_freq[r] >= 2)
                if frac_scheme == "refined":
                    tls_r = sum(2.0 / cand_ref_freq[r] for r in refs_u if cand_ref_freq[r] >= 2)
                else:
                    tls_r = sum(1.0 for r in refs_u if cand_ref_freq[r] >= 2)
                scored_full.append((tls_f, str(w.get("id", "")), idx))
                scored_frac.append((tls_r, str(w.get("id", "")), idx))
            scored_full.sort(key=lambda x: (x[0], x[1]), reverse=True)
            scored_frac.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top_f_set = {x[2] for x in scored_full[:core_top_n]}
            top_r_set = {x[2] for x in scored_frac[:core_top_n]}
            tls_turnover = round(len(top_f_set & top_r_set) / max(1, len(top_f_set | top_r_set)), 4)

            if core_selection == "tls_within_network":
                selected = [eligible[x[2]] for x in scored_full[:core_top_n]]
            elif core_selection == "cited_by_count":
                selected = sorted(
                    eligible,
                    key=lambda w: (int(w.get("cited_by_count", 0) or 0), str(w.get("id", ""))),
                    reverse=True,
                )[:core_top_n]
            else:
                raise ValueError(f"Unsupported core_selection: {core_selection}")
        else:
            selected = list(eligible)

        ref_to_papers: dict[str, list[int]] = defaultdict(list)
        for idx, w in enumerate(selected):
            for r in set(w.get("referenced_works") or []):
                ref_to_papers[r].append(idx)

        full_e: dict[tuple[int, int], float] = defaultdict(float)
        frac_e: dict[tuple[int, int], float] = defaultdict(float)
        bip_edges: list[tuple[int, int]] = []
        event_degrees: dict[int, int] = {}
        event_years: dict[int, int] = {}
        # Store ONLY genuine integer publication years (no silent 2025 imputation!)
        node_years: dict[int, int] = {
            i: int(w["publication_year"])
            for i, w in enumerate(selected)
            if isinstance(w.get("publication_year"), int) and int(w["publication_year"]) > 0
        }

        e_idx = 0
        for ref_key, plist in sorted(ref_to_papers.items()):
            if len(plist) < 2:
                continue
            d_e = full_ref_freq[ref_key] if eff_scope == "full-corpus" else len(plist)
            fw = _frac_weight(d_e, frac_scheme)
            event_degrees[e_idx] = d_e
            yrs_present = [node_years[p] for p in plist if p in node_years]
            if yrs_present:
                event_years[e_idx] = min(yrs_present)
            for p in plist:
                bip_edges.append((p, e_idx))
            e_idx += 1
            for a in range(len(plist)):
                u = plist[a]
                for b in range(a + 1, len(plist)):
                    v = plist[b]
                    key = (u, v) if u < v else (v, u)
                    full_e[key] += 1.0
                    frac_e[key] += fw

        node_records = [
            {
                "id": str(w.get("id", f"doc_{i}")),
                "label": str(w.get("title") or w.get("id") or f"doc_{i}"),
                "title": str(w.get("title") or ""),
                "cited_by_count": int(w.get("cited_by_count") or 0),
                "occurrences": 1,
                "publication_year": w.get("publication_year"),
            }
            for i, w in enumerate(selected)
        ]
        return NetworkBuildArtifact(
            network_type=nt,
            frac_scheme=frac_scheme,
            denom_scope=eff_scope,
            node_records=node_records,
            full_edges=dict(full_e),
            frac_edges=dict(frac_e),
            bipartite_edges=bip_edges,
            event_degrees=event_degrees,
            node_years=node_years,
            event_years=event_years,
            n_eligible_nodes=n_eligible,
            n_core_selected=len(selected),
            event_degree_gini=round(gini_val, 4),
            tls_core_turnover_jaccard=tls_turnover,
            warnings=warnings,
        )

    if nt == "co_citation":
        if not any(w.get("referenced_works") for w in works):
            raise MissingFieldError("No cited references found (`referenced_works` / `References` / `CR`). Cannot build co-citation network.")
        ref_freq: dict[str, int] = defaultdict(int)
        doc_lens: list[int] = []
        for w in works:
            raw_ru = list(dict.fromkeys(w.get("referenced_works") or []))
            ru = _apply_thesaurus(raw_ru, thesaurus) if thesaurus else raw_ru
            if len(ru) >= 2:
                doc_lens.append(len(ru))
            for r in ru:
                ref_freq[r] += 1
        gini_val = gini_coefficient(doc_lens)
        min_c = max(2, int(min_occurrences))
        ranked_refs = sorted(((r, c) for r, c in ref_freq.items() if c >= min_c), key=lambda kv: (-kv[1], kv[0]))
        n_eligible = len(ranked_refs)
        if n_eligible < 2:
            raise InsufficientNetworkError(f"Only {n_eligible} cited references meet min_occurrences >= {min_c}.")
        if core_top_n is not None and len(ranked_refs) > core_top_n:
            ranked_refs = ranked_refs[:core_top_n]

        ref_idx = {r: i for i, (r, _) in enumerate(ranked_refs)}
        node_records = [
            {"id": r, "label": r, "title": "", "cited_by_count": c, "occurrences": c, "publication_year": None}
            for r, c in ranked_refs
        ]
        full_e = defaultdict(float)
        frac_e = defaultdict(float)
        bip_edges = []
        event_degrees = {}
        event_years = {}
        node_earliest: dict[int, int] = {}

        e_idx = 0
        for w in works:
            raw_ru = list(dict.fromkeys(w.get("referenced_works") or []))
            all_refs = _apply_thesaurus(raw_ru, thesaurus) if thesaurus else raw_ru
            present = sorted({ref_idx[r] for r in all_refs if r in ref_idx})
            if len(present) < 2:
                continue
            d_e = len(all_refs) if eff_scope == "full-corpus" else len(present)
            fw = _frac_weight(d_e, frac_scheme)
            event_degrees[e_idx] = d_e
            py = w.get("publication_year")
            if isinstance(py, int) and py > 0:
                event_years[e_idx] = py
                for r_i in present:
                    if r_i not in node_earliest or py < node_earliest[r_i]:
                        node_earliest[r_i] = py
            for r_i in present:
                bip_edges.append((r_i, e_idx))
            e_idx += 1
            for a in range(len(present)):
                u = present[a]
                for b in range(a + 1, len(present)):
                    v = present[b]
                    full_e[(u, v)] += 1.0
                    frac_e[(u, v)] += fw

        return NetworkBuildArtifact(
            network_type=nt,
            frac_scheme=frac_scheme,
            denom_scope=eff_scope,
            node_records=node_records,
            full_edges=dict(full_e),
            frac_edges=dict(frac_e),
            bipartite_edges=bip_edges,
            event_degrees=event_degrees,
            node_years=node_earliest,
            event_years=event_years,
            n_eligible_nodes=n_eligible,
            n_core_selected=len(node_records),
            event_degree_gini=round(gini_val, 4),
            tls_core_turnover_jaccard=None,
            warnings=warnings,
        )

    if nt == "co_authorship":
        if not any(w.get("authors") for w in works):
            raise MissingFieldError("No authors found in input records. Cannot build co-authorship network.")

        doc_authors: list[tuple[list[str], int, int | None]] = []
        id_to_name: dict[str, str] = {}
        au_doc_count: dict[str, int] = defaultdict(int)
        au_cites: dict[str, int] = defaultdict(int)
        n_skipped_hyper = 0
        team_sizes: list[int] = []

        for w in works:
            raw_aus = w.get("authors") or []
            seen_ids: list[str] = []
            seen_set: set[str] = set()
            for a in raw_aus:
                aid = str(a.get("id") or "").strip()
                nm = str(a.get("name") or aid).strip()
                if thesaurus:
                    # Match thesaurus on canonical ID OR lowercased display name
                    if aid.lower() in thesaurus:
                        rep = thesaurus[aid.lower()]
                        if rep is None:
                            continue
                        aid = rep
                        nm = rep
                    elif nm.lower() in thesaurus:
                        rep = thesaurus[nm.lower()]
                        if rep is None:
                            continue
                        aid = rep
                        nm = rep
                if aid and aid not in seen_set:
                    seen_set.add(aid)
                    seen_ids.append(aid)
                    if aid not in id_to_name:
                        id_to_name[aid] = nm
            if not seen_ids:
                continue
            if max_authors > 0 and len(seen_ids) > max_authors:
                n_skipped_hyper += 1
                continue
            cites = int(w.get("cited_by_count") or 0)
            py = w.get("publication_year") if isinstance(w.get("publication_year"), int) and int(w.get("publication_year") or 0) > 0 else None
            for aid in seen_ids:
                au_doc_count[aid] += 1
                au_cites[aid] += cites
            if len(seen_ids) >= 2:
                team_sizes.append(len(seen_ids))
                doc_authors.append((seen_ids, cites, py))

        if n_skipped_hyper > 0:
            warnings.append(f"Excluded {n_skipped_hyper} hyper-authored document(s) exceeding max_authors={max_authors} (VOSviewer parity guard).")
        if any(w.get("authors_truncated") for w in works) and (max_authors == 0 or max_authors >= 100):
            warnings.append("OpenAlexTruncationWarning: corpus includes >=100-author OpenAlex records while max_authors >= 100.")

        gini_val = gini_coefficient(team_sizes)
        ranked = sorted(
            ((aid, cnt) for aid, cnt in au_doc_count.items() if cnt >= max(1, int(min_occurrences))),
            key=lambda kv: (-kv[1], -au_cites[kv[0]], kv[0]),
        )
        n_eligible = len(ranked)
        if n_eligible < 2:
            raise InsufficientNetworkError(f"Only {n_eligible} authors meet min_occurrences >= {min_occurrences}.")
        if core_top_n is not None and len(ranked) > core_top_n:
            ranked = ranked[:core_top_n]

        au_idx = {aid: i for i, (aid, _) in enumerate(ranked)}
        node_records = [
            {
                "id": aid,
                "label": id_to_name.get(aid, aid),
                "title": "",
                "cited_by_count": au_cites[aid],
                "occurrences": cnt,
                "publication_year": None,
            }
            for aid, cnt in ranked
        ]
        full_e = defaultdict(float)
        frac_e = defaultdict(float)
        bip_edges = []
        event_degrees = {}
        event_years = {}

        e_idx = 0
        for seen_ids, _, py in doc_authors:
            present = sorted({au_idx[aid] for aid in seen_ids if aid in au_idx})
            if len(present) < 2:
                continue
            d_e = len(seen_ids) if eff_scope == "full-corpus" else len(present)
            fw = _frac_weight(d_e, frac_scheme)
            event_degrees[e_idx] = d_e
            if py is not None:
                event_years[e_idx] = py
            for u in present:
                bip_edges.append((u, e_idx))
            e_idx += 1
            for a in range(len(present)):
                u = present[a]
                for b in range(a + 1, len(present)):
                    v = present[b]
                    full_e[(u, v)] += 1.0
                    frac_e[(u, v)] += fw

        return NetworkBuildArtifact(
            network_type=nt,
            frac_scheme=frac_scheme,
            denom_scope=eff_scope,
            node_records=node_records,
            full_edges=dict(full_e),
            frac_edges=dict(frac_e),
            bipartite_edges=bip_edges,
            event_degrees=event_degrees,
            node_years={},
            event_years=event_years,
            n_eligible_nodes=n_eligible,
            n_core_selected=len(node_records),
            event_degree_gini=round(gini_val, 4),
            tls_core_turnover_jaccard=None,
            warnings=warnings,
        )

    if nt == "keyword_cooccurrence":
        if keyword_source not in {"author", "index", "all"}:
            raise ValueError(f"Unsupported keyword_source '{keyword_source}'. Choose: author, index, all.")
        doc_kws: list[tuple[list[str], int | None]] = []
        kw_freq: dict[str, int] = defaultdict(int)
        kw_cites: dict[str, int] = defaultdict(int)
        kw_list_lens: list[int] = []

        eff_kw_src = keyword_source
        if eff_kw_src == "author" and not any(w.get("author_keywords") for w in works) and any(w.get("index_keywords") for w in works):
            eff_kw_src = "index"
            warnings.append("KeywordFallbackWarning: `author` keywords absent; fell back to `index`/topic keywords.")

        for w in works:
            if eff_kw_src == "author":
                raw_k = list(w.get("author_keywords") or [])
            elif eff_kw_src == "index":
                raw_k = list(w.get("index_keywords") or [])
            else:
                raw_k = list(w.get("author_keywords") or []) + list(w.get("index_keywords") or [])
            cleaned = _apply_thesaurus(raw_k, thesaurus)
            if not cleaned:
                continue
            cites = int(w.get("cited_by_count") or 0)
            py = w.get("publication_year") if isinstance(w.get("publication_year"), int) and int(w.get("publication_year") or 0) > 0 else None
            for k in cleaned:
                kw_freq[k] += 1
                kw_cites[k] += cites
            if len(cleaned) >= 2:
                kw_list_lens.append(len(cleaned))
                doc_kws.append((cleaned, py))

        if not kw_freq:
            raise MissingFieldError(f"No keywords found under keyword_source='{keyword_source}'. Cannot build keyword co-occurrence network.")

        gini_val = gini_coefficient(kw_list_lens)
        ranked = sorted(
            ((k, cnt) for k, cnt in kw_freq.items() if cnt >= max(1, int(min_occurrences))),
            key=lambda kv: (-kv[1], -kw_cites[kv[0]], kv[0]),
        )
        n_eligible = len(ranked)
        if n_eligible < 2:
            raise InsufficientNetworkError(f"Only {n_eligible} keywords meet min_occurrences >= {max(1, int(min_occurrences))}.")
        if core_top_n is not None and len(ranked) > core_top_n:
            ranked = ranked[:core_top_n]

        kw_idx = {k: i for i, (k, _) in enumerate(ranked)}
        node_records = [
            {
                "id": k,
                "label": k,
                "title": "",
                "cited_by_count": kw_cites[k],
                "occurrences": cnt,
                "publication_year": None,
            }
            for k, cnt in ranked
        ]
        full_e = defaultdict(float)
        frac_e = defaultdict(float)
        bip_edges = []
        event_degrees = {}
        event_years = {}

        e_idx = 0
        for cleaned, py in doc_kws:
            present = sorted({kw_idx[k] for k in cleaned if k in kw_idx})
            if len(present) < 2:
                continue
            d_e = len(cleaned) if eff_scope == "full-corpus" else len(present)
            fw = _frac_weight(d_e, frac_scheme)
            event_degrees[e_idx] = d_e
            if py is not None:
                event_years[e_idx] = py
            for u in present:
                bip_edges.append((u, e_idx))
            e_idx += 1
            for a in range(len(present)):
                u = present[a]
                for b in range(a + 1, len(present)):
                    v = present[b]
                    full_e[(u, v)] += 1.0
                    frac_e[(u, v)] += fw

        return NetworkBuildArtifact(
            network_type=nt,
            frac_scheme=frac_scheme,
            denom_scope=eff_scope,
            node_records=node_records,
            full_edges=dict(full_e),
            frac_edges=dict(frac_e),
            bipartite_edges=bip_edges,
            event_degrees=event_degrees,
            node_years={},
            event_years=event_years,
            n_eligible_nodes=n_eligible,
            n_core_selected=len(node_records),
            event_degree_gini=round(gini_val, 4),
            tls_core_turnover_jaccard=None,
            warnings=warnings,
        )

    raise ValueError(f"Unhandled network_type: {nt}")


def build_sparse_edges_from_works(
    works: list[dict[str, Any]],
    network_type: str = "bibliographic_coupling",
    core_top_n: int | None = 150,
    core_selection: str = "tls_within_network",
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Backward-compatible wrapper returning `(node_records, full_edges, frac_edges)`."""
    art = build_network_artifact(
        works,
        network_type=network_type,
        core_top_n=core_top_n,
        core_selection=core_selection,
        **kwargs,
    )
    return art.node_records, art.full_edges, art.frac_edges
