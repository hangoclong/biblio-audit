"""Layer 2a: Execute author-verified Layer-2 queries via verbatim Scopus API + OpenAlex join.

Hardened specification: - `--excel` mode cleanly re-joins `scopus_id`, `journal_key`, `citescore`, `ipw_weight` by `doi`
  from `layer2_llm_candidates.json` (and `busi_population.parquet`) while honoring all author
  corrections from Excel, AND emits explicit `"journal_key"` in `layer2_manifest.json`.
- `build_scopus_api_query` handles parenthesized field clauses (`(TITLE-ABS-KEY(...) AND ...)`).
- `--mode verify-counts` dispatches Scopus vs OpenAlex by `database_source` (zero WoS->Scopus mistakes).
- `compute_lcc_size` applies `core_top_n` within-network TLS selection BEFORE BC BFS (fast & exact parity).
- `join_scopus_dois_to_openalex` enforces `scopus_to_oa_join_rate >= 0.70` and logs `pct_works_with_refs`.
- `fetch_cited_reference_years` prioritizes all `c >= 2` references inside the `core_top_n` subgraph first.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.layer2_harvest")

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
PROC_DIR = EXP_DIR / "data" / "processed"
RAW_CORP_DIR = EXP_DIR / "data" / "raw" / "repro_corpora"
SCOPUS_CACHE_DIR = EXP_DIR / "data" / "raw" / "scopus_cache"
REF_YEARS_CACHE_PATH = EXP_DIR / "data" / "raw" / "ref_years_cache.json"
RES_DIR = EXP_DIR / "results"


class OpenAlexBudgetExhaustedError(RuntimeError):
    """Raised immediately when OpenAlex reports daily budget exhaustion (HTTP 429)."""


def _check_oa_budget(resp: requests.Response) -> None:
    if resp.status_code == 429 and ("Insufficient budget" in resp.text or "daily_budget" in resp.text):
        raise OpenAlexBudgetExhaustedError(f"OpenAlex daily budget exhausted (Retry-After={resp.headers.get('Retry-After')}s)")


def load_ref_years_cache() -> dict[str, int]:
    """Load global cited-reference publication-year cache (seeded from existing corpora if needed)."""
    cache: dict[str, int] = {}
    if REF_YEARS_CACHE_PATH.exists():
        try:
            cache = {str(k): int(v) for k, v in json.loads(REF_YEARS_CACHE_PATH.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            cache = {}
    if not cache and RAW_CORP_DIR.exists():
        for p in RAW_CORP_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("ref_years"), dict):
                    for k, v in d["ref_years"].items():
                        if k and v is not None:
                            cache[str(k)] = int(v)
            except Exception:  # noqa: BLE001
                continue
        if cache:
            REF_YEARS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            REF_YEARS_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def save_ref_years_cache(cache: dict[str, int]) -> None:
    REF_YEARS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REF_YEARS_CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(REF_YEARS_CACHE_PATH)


def load_env_keys(env_path: Path) -> dict[str, str]:
    """Read API keys from .env without logging secret values."""
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def norm_doi(val: str | None) -> str:
    """Normalize DOI to lowercase bare identifier."""
    if not val:
        return ""
    s = str(val).strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", s).strip()


def _parens_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _normalize_scopus_limit_to(raw: str) -> str:
    """Translate Scopus Web-UI LIMIT-TO(FIELD, "VAL") clauses into REST Search API FIELD(VAL) syntax."""
    def _repl(m: re.Match[str]) -> str:
        fld = m.group(1).strip().upper()
        val = m.group(2).strip().strip('"').strip("'")
        if fld in ("PUBSTAGE", "EXACTKEYWORD", "EXACTSRCTITLE", "AFFILCOUNTRY", "AUTHOR_NAME"):
            return ""
        if fld == "LANGUAGE":
            return f"LANGUAGE({val.lower()})"
        return f"{fld}({val})"

    out = re.sub(r"\bLIMIT-TO\s*\(\s*([A-Z_]+)\s*,\s*[\"']?([^\"')]+)[\"']?\s*\)", _repl, raw, flags=re.I)
    # Clean empty parens and dangling AND/OR around stripped web-UI clauses
    for _ in range(3):
        out = re.sub(r"\(\s*\)", "", out)
        out = re.sub(r"\b(?:AND|OR)\s+(?:AND|OR)\b", "AND", out, flags=re.I)
        out = re.sub(r"\b(?:AND|OR)\s*(?=\)|\s*$)", "", out, flags=re.I)
        out = re.sub(r"(?<=\()\s*(?:AND|OR)\b", "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _balance_trailing_parens(expr: str) -> str:
    """Trim unmatched outer/trailing parentheses caused by PDF line-wrap typos."""
    s = expr.strip()
    while s.count(")") > s.count("(") and s.endswith(")"):
        s = s[:-1].strip()
    while s.count("(") > s.count(")") and s.startswith("("):
        s = s[1:].strip()
    return s


def build_scopus_api_query(spec: dict[str, Any]) -> str:
    """Build exact verbatim Scopus Search API query preserving `*` wildcards & stated filters."""
    raw = re.sub(r"\s+", " ", str(spec.get("verbatim_query") or "")).strip()
    if not raw:
        raise ValueError(f"Empty verbatim_query for {spec.get('doi')}")

    raw = raw.replace("[", "(").replace("]", ")")
    raw = re.sub(r"'([^']{2,100})'", r'"\1"', raw)
    raw = re.sub(r"\bTITLE\s+ABS\s+KEY\b", "TITLE-ABS-KEY", raw, flags=re.I)
    raw = re.sub(r"^(?:TS|TI|TOPIC)\s*=\s*", "", raw, flags=re.I).strip()
    raw = _normalize_scopus_limit_to(raw)
    raw = _balance_trailing_parens(raw)
    # Search anywhere (handles leading '(' like '(TITLE-ABS-KEY(...) AND TITLE-ABS-KEY(...))')
    has_field = bool(re.search(r"\b(?:TITLE-ABS-KEY|TITLE-ABS|ABS-KEY|TITLE|ALL|KEY)\s*\(", raw, re.I))
    if has_field:
        core = raw
    else:
        # Never wrap Scopus metadata filters (SUBJAREA/DOCTYPE/LANGUAGE/SRCTYPE/PUBYEAR) inside TITLE-ABS-KEY(...)!
        m_meta = re.search(r"\s+AND\s+\(?\s*(?:SUBJAREA|DOCTYPE|LANGUAGE|SRCTYPE|PUBYEAR)\s*\(", raw, re.I)
        if m_meta:
            kw_part = raw[: m_meta.start()].strip()
            meta_part = raw[m_meta.start() :].strip()
            core = f"TITLE-ABS-KEY({kw_part}) {meta_part}"
        else:
            core = f"TITLE-ABS-KEY({raw})"

    clauses = [f"({core})"]
    raw_up = raw.upper()

    start_yr = spec.get("start_year")
    cutoff_yr = spec.get("cutoff_year") or spec.get("pub_year")
    if "PUBYEAR" not in raw_up:
        if start_yr and str(start_yr).strip().isdigit():
            clauses.append(f"PUBYEAR > {int(start_yr) - 1}")
        if cutoff_yr and str(cutoff_yr).strip().isdigit():
            clauses.append(f"PUBYEAR < {int(cutoff_yr) + 1}")

    subj = str(spec.get("subjarea_filter") or "").strip().upper()
    if subj and subj not in ("NONE", "NULL", "N/A") and "SUBJAREA" not in raw_up:
        codes = [c.strip() for c in re.split(r"[,;/|\s]+", subj) if re.fullmatch(r"[A-Z]{4}", c.strip())]
        if codes:
            clauses.append("(" + " OR ".join(f"SUBJAREA({c})" for c in codes) + ")")

    if "DOCTYPE" not in raw_up:
        clauses.append("(DOCTYPE(ar) OR DOCTYPE(re))")
    if "LANGUAGE" not in raw_up:
        clauses.append("LANGUAGE(english)")
    return " AND ".join(clauses)


def _uppercase_boolean_ops_outside_quotes(expr: str) -> str:
    """Uppercase standalone Boolean operators AND/OR/NOT outside double-quoted phrases for OpenAlex."""
    parts = re.split(r'("[^"]*")', expr)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
        else:
            out.append(re.sub(r"\b(and|or|not)\b", lambda m: m.group(1).upper(), p, flags=re.I))
    return "".join(out)


def clean_boolean_for_openalex(raw_expr: str) -> str | None:
    """Translate non-Scopus/WoS Boolean core into clean balanced OpenAlex search syntax."""
    expr = re.sub(r"\s+", " ", raw_expr).strip()
    expr = expr.replace("[", "(").replace("]", ")")
    expr = re.sub(r"'([^']{2,80})'", r'"\1"', expr)
    expr = re.sub(r"\b(?:TITLE-ABS-KEY|TITLE-ABS|ABS-KEY|TITLE\s+ABS\s+KEY|TITLE|ALL)\s*\(", "(", expr, flags=re.I)
    expr = re.sub(r"\b(?:TS|TI|TOPIC)\s*=\s*", "", expr, flags=re.I)
    expr = re.split(
        r"\b(?:AND\s+\(?\s*)?(?:LIMIT-TO|EXCLUDE|DOCTYPE|LANGUAGE|SUBJAREA|SRCTYPE|PUBYEAR)\b",
        expr,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ;,.")
    expr = re.sub(r"\b(?:W|PRE|NEAR)/\d+\b", " AND ", expr, flags=re.I)
    # Replace ? wildcards inside words (e.g., organi?ation -> organization) instead of hard-failing
    expr = re.sub(r"organi\?ation", "organization", expr, flags=re.I)
    expr = expr.replace("?", "")
    expr = expr.replace("*", "").replace("$", "").replace("{", '"').replace("}", '"')
    expr = re.sub(r"[,:|]+", " ", expr)
    expr = _uppercase_boolean_ops_outside_quotes(expr)
    expr = _balance_trailing_parens(re.sub(r"\s+", " ", expr).strip())
    while expr.startswith("(") and expr.endswith(")") and _parens_balanced(expr[1:-1]):
        expr = expr[1:-1].strip()
    if not _parens_balanced(expr) or (expr.count('"') % 2 != 0):
        return None
    return expr if len(expr) >= 4 else None


def query_scopus_hit_count(scopus_q: str, skey: str) -> tuple[int | None, int]:
    """Query exact Scopus Search API totalResults count."""
    try:
        r = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            params={"query": scopus_q, "count": 1},
            headers={"X-ELS-APIKey": skey, "Accept": "application/json"},
            timeout=25,
        )
        if r.status_code != 200:
            return None, r.status_code
        cnt = int(r.json().get("search-results", {}).get("opensearch:totalResults", 0))
        return cnt, 200
    except requests.RequestException:
        return None, -1


def query_openalex_hit_count(spec: dict[str, Any], oakey: str) -> tuple[int | None, int]:
    """Query OpenAlex hit count for non-Scopus (WoS/OpenAlex) reviews in verify-counts mode."""
    raw_q = str(spec.get("verbatim_query") or "").strip()
    search_field = "title.search" if re.match(r"^\s*(?:TITLE\s*\(|TI\s*=)", raw_q, re.I) else "title_and_abstract.search"
    oa_q = clean_boolean_for_openalex(raw_q)
    if not oa_q:
        return None, 400
    parts = [f"{search_field}:{oa_q}", "type:article|review", "language:en"]
    start_yr = spec.get("start_year")
    cutoff_yr = spec.get("cutoff_year") or spec.get("pub_year")
    if start_yr and str(start_yr).strip().isdigit():
        parts.append(f"publication_year:>{int(start_yr) - 1}")
    if cutoff_yr and str(cutoff_yr).strip().isdigit():
        parts.append(f"publication_year:<{int(cutoff_yr) + 1}")
    if "BUSI" in str(spec.get("subjarea_filter") or "").upper():
        parts.append("primary_topic.field.id:fields/14")
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"filter": ",".join(parts), "per_page": 1, "api_key": oakey},
            timeout=25,
        )
        if r.status_code != 200:
            return None, r.status_code
        return int(r.json().get("meta", {}).get("count", 0)), 200
    except requests.RequestException:
        return None, -1


def fetch_scopus_records_complete(
    spec: dict[str, Any],
    skey: str,
    max_complete_works: int = 10_000,
) -> tuple[int, list[dict[str, Any]], bool]:
    """Fetch complete Scopus article list via year-partitioned deterministic pagination."""
    base_q = build_scopus_api_query(spec)
    total, status = query_scopus_hit_count(base_q, skey)
    if status != 200 or total is None:
        LOGGER.warning("Scopus count failed (HTTP %d) for %s", status, spec.get("doi"))
        return -1, [], False
    if total < 50 or total > max_complete_works:
        return total, [], False

    url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": skey, "Accept": "application/json"}
    sort_spec = "+coverDate,+publicationName,+citedby-count"

    by_id: dict[str, dict[str, Any]] = {}
    if total <= 4500:
        query_slices = [base_q]
    else:
        raw_lo = spec.get("start_year")
        y_lo = int(raw_lo) if (raw_lo is not None and str(raw_lo).strip().isdigit()) else 1990
        raw_hi = spec.get("cutoff_year")
        y_hi = int(raw_hi) if (raw_hi is not None and str(raw_hi).strip().isdigit()) else 2026
        query_slices = []
        if raw_lo is None or str(raw_lo).strip() == "":
            query_slices.append(f"({base_q}) AND PUBYEAR < {y_lo}")
        query_slices.extend([f"({base_q}) AND PUBYEAR = {yr}" for yr in range(y_lo, y_hi + 1)])

    for q_slice in query_slices:
        start = 0
        slice_expected: int | None = None
        while True:
            resp = requests.get(
                url,
                params={"query": q_slice, "start": start, "count": 25, "sort": sort_spec},
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                LOGGER.warning("Scopus HTTP %d at start=%d for %s", resp.status_code, start, spec.get("doi"))
                return total, [], False
            sr = resp.json().get("search-results", {})
            if slice_expected is None:
                slice_expected = int(sr.get("opensearch:totalResults", 0))
                if slice_expected == 0:
                    break
            entries = sr.get("entry", [])
            if not entries or (len(entries) == 1 and entries[0].get("error")):
                break
            for e in entries:
                if e.get("error"):
                    continue
                sid = str(e.get("dc:identifier", "")).replace("SCOPUS_ID:", "")
                cover = str(e.get("prism:coverDate", ""))
                py = int(cover[:4]) if len(cover) >= 4 and cover[:4].isdigit() else None
                by_id[sid] = {
                    "scopus_id": sid,
                    "doi": norm_doi(e.get("prism:doi")),
                    "title": e.get("dc:title", ""),
                    "publication_year": py,
                    "cited_by_count": int(e.get("citedby-count", 0) or 0),
                }
            start += len(entries)
            if slice_expected is not None and start >= slice_expected:
                break
            time.sleep(0.08)

    records = list(by_id.values())
    is_complete = len(records) >= int(0.95 * total)
    return total, records, is_complete


def join_scopus_dois_to_openalex(
    scopus_records: list[dict[str, Any]], oakey: str
) -> tuple[list[dict[str, Any]], float, float]:
    """Attach OpenAlex `referenced_works` to exact Scopus records; return (merged, join_rate, pct_with_refs)."""
    dois = sorted({r["doi"] for r in scopus_records if r.get("doi")})
    oa_by_doi: dict[str, dict[str, Any]] = {}
    url = "https://api.openalex.org/works"
    for i in range(0, len(dois), 50):
        batch = dois[i : i + 50]
        for attempt in range(3):
            r = requests.get(
                url,
                params={
                    "filter": "doi:" + "|".join(batch),
                    "per_page": 100,
                    "select": "id,doi,title,publication_year,cited_by_count,referenced_works",
                    "api_key": oakey,
                },
                timeout=30,
            )
            if r.status_code == 200:
                for w in r.json().get("results", []):
                    d = norm_doi(w.get("doi"))
                    if d:
                        oa_by_doi[d] = w
                break
            _check_oa_budget(r)
            time.sleep(0.8 * (attempt + 1))
        time.sleep(0.06)

    merged: list[dict[str, Any]] = []
    matched_cnt = 0
    with_refs_cnt = 0
    for rec in scopus_records:
        ow = oa_by_doi.get(rec["doi"])
        if ow:
            matched_cnt += 1
            refs = ow.get("referenced_works") or []
            if refs:
                with_refs_cnt += 1
            merged.append(
                {
                    "id": ow.get("id") or f"SCOPUS:{rec['scopus_id']}",
                    "doi": rec["doi"],
                    "title": rec["title"] or ow.get("title", ""),
                    "publication_year": rec["publication_year"] or ow.get("publication_year"),
                    "cited_by_count": rec["cited_by_count"],
                    "referenced_works": refs,
                }
            )
    n_sc = max(1, len(scopus_records))
    return merged, round(matched_cnt / n_sc, 4), round(with_refs_cnt / max(1, matched_cnt), 4)


def fetch_complete_openalex_corpus(
    spec: dict[str, Any],
    oakey: str,
    max_complete_works: int = 10_000,
) -> tuple[int, list[dict[str, Any]], bool]:
    """Complete OpenAlex corpus harvest for WoS-only or OpenAlex reviews."""
    raw_q = str(spec.get("verbatim_query") or "").strip()
    search_field = "title.search" if re.match(r"^\s*(?:TITLE\s*\(|TI\s*=)", raw_q, re.I) else "title_and_abstract.search"
    oa_q = clean_boolean_for_openalex(raw_q)
    if not oa_q:
        return -1, [], False
    url = "https://api.openalex.org/works"
    parts = [f"{search_field}:{oa_q}", "type:article|review", "language:en"]
    start_yr = spec.get("start_year")
    cutoff_yr = spec.get("cutoff_year") or spec.get("pub_year")
    if start_yr and str(start_yr).strip().isdigit():
        parts.append(f"publication_year:>{int(start_yr) - 1}")
    if cutoff_yr and str(cutoff_yr).strip().isdigit():
        parts.append(f"publication_year:<{int(cutoff_yr) + 1}")
    if "BUSI" in str(spec.get("subjarea_filter") or "").upper():
        parts.append("primary_topic.field.id:fields/14")
    filt = ",".join(parts)

    r0 = requests.get(url, params={"filter": filt, "per_page": 1, "api_key": oakey}, timeout=25)
    _check_oa_budget(r0)
    if r0.status_code != 200:
        return -1, [], False
    total = int(r0.json().get("meta", {}).get("count", 0))
    if total < 50 or total > max_complete_works:
        return total, [], False

    works: list[dict[str, Any]] = []
    cursor = "*"
    while cursor:
        ok = False
        for attempt in range(3):
            r = requests.get(
                url,
                params={
                    "filter": filt,
                    "per_page": 200,
                    "cursor": cursor,
                    "select": "id,doi,title,publication_year,cited_by_count,referenced_works",
                    "api_key": oakey,
                },
                timeout=30,
            )
            if r.status_code == 200:
                ok = True
                break
            _check_oa_budget(r)
            time.sleep(1.0 * (attempt + 1))
        if not ok:
            return total, [], False
        data = r.json()
        batch = data.get("results", [])
        if not batch:
            break
        works.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        if len(works) >= total:
            break
        time.sleep(0.06)

    return total, works, len(works) >= int(0.95 * total)


def _select_bc_tls_core(works: list[dict[str, Any]], core_top_n: int) -> list[dict[str, Any]]:
    """Select top-N BC works by within-corpus TLS (matches counting_engine.py )."""
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


def fetch_cited_reference_years(
    works: list[dict[str, Any]],
    oakey: str,
    network_type: str = "bibliographic_coupling",
    core_top_n: int = 150,
    global_cache: dict[str, int] | None = None,
) -> dict[str, int]:
    """Fetch true OpenAlex publication_year for ALL c>=2 refs inside the exact core_top_n subgraph (cached globally)."""
    if network_type == "bibliographic_coupling":
        sub = _select_bc_tls_core(works, core_top_n)
        c_freq: Counter[str] = Counter()
        for w in sub:
            for r in set(w.get("referenced_works") or []):
                c_freq[r] += 1
        core_refs = [r for r, c in c_freq.items() if c >= 2 and str(r).startswith("https://openalex.org/W")]
    else:
        c_freq = Counter()
        for w in works:
            for r in set(w.get("referenced_works") or []):
                c_freq[r] += 1
        core_refs = [r for r, c in c_freq.most_common(max(core_top_n, 250)) if c >= 2 and str(r).startswith("https://openalex.org/W")]

    cache = global_cache if global_cache is not None else {}
    years: dict[str, int] = {}
    to_fetch: list[str] = []
    for r in core_refs:
        if r in cache:
            years[r] = cache[r]
        else:
            to_fetch.append(r)

    short_ids = [r.rsplit("/", 1)[-1] for r in to_fetch]
    url = "https://api.openalex.org/works"
    for i in range(0, len(short_ids), 50):
        batch = short_ids[i : i + 50]
        for attempt in range(3):
            r = requests.get(
                url,
                params={"filter": "openalex:" + "|".join(batch), "per_page": 100, "select": "id,publication_year", "api_key": oakey},
                timeout=25,
            )
            if r.status_code == 200:
                for item in r.json().get("results", []):
                    wid = item.get("id")
                    py = item.get("publication_year")
                    if wid and py is not None:
                        years[str(wid)] = int(py)
                        cache[str(wid)] = int(py)
                break
            _check_oa_budget(r)
            time.sleep(0.6 * (attempt + 1))
        time.sleep(0.04)
    return years


def compute_lcc_size(works: list[dict[str, Any]], network_type: str, core_top_n: int = 150) -> int:
    """Compute Largest Connected Component node count on the exact core_top_n subgraph."""
    if network_type == "bibliographic_coupling":
        sub = _select_bc_tls_core(works, core_top_n)
        ref_to_p: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(sub):
            for r in set(w.get("referenced_works") or []):
                ref_to_p[r].append(i)
        adj: dict[int, set[int]] = defaultdict(set)
        for plist in ref_to_p.values():
            if len(plist) < 2:
                continue
            for a in range(len(plist)):
                u = plist[a]
                for b in range(a + 1, len(plist)):
                    v = plist[b]
                    adj[u].add(v)
                    adj[v].add(u)
    else:
        ref_freq: Counter[str] = Counter()
        for w in works:
            for r in set(w.get("referenced_works") or []):
                ref_freq[r] += 1
        top_refs = {r for r, c in ref_freq.most_common(core_top_n) if c >= 2}
        r2i = {r: i for i, r in enumerate(sorted(top_refs))}
        adj = defaultdict(set)
        for w in works:
            present = [r2i[r] for r in set(w.get("referenced_works") or []) if r in r2i]
            if len(present) < 2:
                continue
            for a in range(len(present)):
                u = present[a]
                for b in range(a + 1, len(present)):
                    v = present[b]
                    adj[u].add(v)
                    adj[v].add(u)

    visited: set[int] = set()
    best = 0
    for node in list(adj.keys()):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        sz = 0
        while stack:
            cur = stack.pop()
            sz += 1
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if sz > best:
            best = sz
    return best


def load_specs_from_excel(xlsx_path: Path, candidates_json: Path | None = None) -> list[dict[str, Any]]:
    """Load author-verified Layer-2 specs from flat Excel & re-join internal metadata by DOI."""
    import openpyxl

    base_by_doi: dict[str, dict[str, Any]] = {}
    if candidates_json is not None and candidates_json.exists():
        for rec in json.loads(candidates_json.read_text(encoding="utf-8")):
            d = norm_doi(rec.get("doi"))
            if d:
                base_by_doi[d] = rec

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Layer2_Verify_60"]
    headers = [str(ws.cell(1, c).value or "").strip() for c in range(1, ws.max_column + 1)]
    rows: list[dict[str, Any]] = []
    for r in range(2, ws.max_row + 1):
        excel_rec = {headers[c - 1]: ws.cell(r, c).value for c in range(1, len(headers) + 1)}
        verdict = str(excel_rec.get("author_verdict_OK_or_FIX") or "").strip().upper()
        if verdict not in ("OK", "APPROVED", "PASS"):
            continue
        d = norm_doi(excel_rec.get("doi"))
        merged = dict(base_by_doi.get(d, {}))
        # Author Excel values override base candidate fields cleanly (preserving scopus_id, journal_key, citescore, ipw_weight)
        for k, v in excel_rec.items():
            if v is not None and str(v).strip() != "":
                merged[k] = v
        if not merged.get("journal_key"):
            merged["journal_key"] = str(merged.get("journal") or d)
        rows.append(merged)
    return rows


def main() -> None:
    """Verify live Scopus/OpenAlex hit counts (`verify-counts`) or harvest author-approved corpora (`harvest`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify-counts", "harvest"], default="harvest")
    ap.add_argument("--specs", type=str, default=str(RES_DIR / "layer2_llm_candidates.json"))
    ap.add_argument("--excel", type=str, default="", help="Author-approved Excel workbook (`Layer2_Verify_60`)")
    ap.add_argument("--quota-q1", type=int, default=30)
    ap.add_argument("--quota-q24", type=int, default=30)
    ap.add_argument("--per-journal-cap", type=int, default=3)
    ap.add_argument("--max-complete-works", type=int, default=10_000)
    ap.add_argument("--min-oa-join-rate", type=float, default=0.70)
    args = ap.parse_args()

    keys = load_env_keys(REPO_ROOT / ".env")
    skey = keys["SCOPUS_API_KEY"]
    oakey = keys["OPENALEX_API_KEY"]

    if args.excel:
        specs = load_specs_from_excel(Path(args.excel), candidates_json=Path(args.specs))
        LOGGER.info("Loaded %d author-approved specs from %s (re-joined with %s)", len(specs), args.excel, args.specs)
    else:
        specs = json.loads(Path(args.specs).read_text(encoding="utf-8"))
        LOGGER.info("Loaded %d specs from %s", len(specs), args.specs)

    if args.mode == "verify-counts":
        for sp in specs:
            db_src = str(sp.get("database_source") or "scopus").lower()
            try:
                if "scopus" in db_src:
                    sq = build_scopus_api_query(sp)
                    cnt, st = query_scopus_hit_count(sq, skey)
                    sp["repro_api_query"] = sq
                    sp["repro_count_source"] = "scopus"
                else:
                    cnt, st = query_openalex_hit_count(sp, oakey)
                    sp["repro_count_source"] = "openalex"
                sp["n_repro_count"] = cnt
                sp["repro_http_status"] = st
                n_orig = sp.get("n_orig_printed")
                if cnt is not None and n_orig and int(n_orig) > 0:
                    sp["ratio_repro_orig"] = round(cnt / int(n_orig), 3)
                else:
                    sp["ratio_repro_orig"] = None
            except Exception as exc:  # noqa: BLE001
                sp["n_repro_count"] = None
                sp["repro_http_status"] = f"err:{exc}"
                sp["ratio_repro_orig"] = None
            time.sleep(0.10)
        Path(args.specs).write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Updated %d candidate specs with live Scopus/OpenAlex counts at %s", len(specs), args.specs)
        return

    RAW_CORP_DIR.mkdir(parents=True, exist_ok=True)
    SCOPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    global_ref_cache = load_ref_years_cache()
    LOGGER.info("Loaded global ref_years cache: %d known OpenAlex reference years", len(global_ref_cache))

    quotas = {"Q1": args.quota_q1, "Q2_Q4": args.quota_q24}
    got_by_st: Counter[str] = Counter()
    j_counts: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    screen_log: list[dict[str, Any]] = []

    for sp in specs:
        q = str(sp.get("quartile", ""))
        st_name = "Q1" if q == "Q1" else "Q2_Q4"
        if got_by_st[st_name] >= quotas[st_name]:
            continue
        jkey = str(sp.get("journal_key") or sp.get("journal") or "")
        if j_counts[jkey] >= args.per_journal_cap:
            screen_log.append({"doi": sp.get("doi"), "status": "skip_journal_cap"})
            continue

        net_type = str(sp.get("llm_network_type") or sp.get("network_type") or "bibliographic_coupling")
        if net_type not in ("bibliographic_coupling", "co_citation"):
            screen_log.append({"doi": sp.get("doi"), "status": f"skip_unsupported_net:{net_type}"})
            continue

        db_src = str(sp.get("database_source") or "scopus").lower()
        core_n = max(50, min(500, int(sp.get("node_threshold") or sp.get("core_top_n") or 150)))
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sp.get("doi") or sp.get("scopus_id")))[:100] + ".json"
        corp_path = RAW_CORP_DIR / safe_id
        q_sig = f"{db_src}|{sp.get('verbatim_query')}|{sp.get('start_year')}|{sp.get('cutoff_year')}|{sp.get('subjarea_filter')}|{net_type}|{core_n}"

        cached_ok = False
        need_ref_years_backfill = False
        if corp_path.exists():
            try:
                cached = json.loads(corp_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("query_sig") == q_sig and cached.get("works"):
                    works = cached["works"]
                    ref_years = cached.get("ref_years") or {}
                    total_hits = int(cached.get("source_total_hits") or len(works))
                    join_rate = float(cached.get("scopus_to_oa_join_rate") or 1.0)
                    pct_refs = float(cached.get("pct_works_with_refs") or 1.0)
                    harvest_source = str(cached.get("harvest_source") or "cached")
                    is_complete = bool(cached.get("is_complete_corpus", True))
                    cached_ok = True
                    need_ref_years_backfill = len(ref_years) == 0
            except Exception:  # noqa: BLE001
                cached_ok = False

        if not cached_ok:
            join_rate = 1.0
            pct_refs = 1.0
            harvest_source = "scopus_to_openalex"
            if "scopus" in db_src:
                sc_cache_path = SCOPUS_CACHE_DIR / safe_id
                sc_q_sig = f"{sp.get('verbatim_query')}|{sp.get('start_year')}|{sp.get('cutoff_year')}|{sp.get('subjarea_filter')}"
                sc_recs = None
                total_hits = -1
                is_complete = False
                if sc_cache_path.exists():
                    try:
                        sc_cached = json.loads(sc_cache_path.read_text(encoding="utf-8"))
                        if isinstance(sc_cached, dict) and sc_cached.get("sc_q_sig") == sc_q_sig and sc_cached.get("records"):
                            total_hits = int(sc_cached["total_hits"])
                            sc_recs = sc_cached["records"]
                            is_complete = bool(sc_cached.get("is_complete", True))
                    except Exception:  # noqa: BLE001
                        sc_recs = None
                if sc_recs is None:
                    total_hits, sc_recs, is_complete = fetch_scopus_records_complete(sp, skey, args.max_complete_works)
                    if is_complete and len(sc_recs) >= 50:
                        sc_cache_path.write_text(
                            json.dumps({"sc_q_sig": sc_q_sig, "total_hits": total_hits, "is_complete": is_complete, "records": sc_recs}, ensure_ascii=False),
                            encoding="utf-8",
                        )
                if not is_complete or len(sc_recs) < 50:
                    screen_log.append({"doi": sp.get("doi"), "status": f"skip_scopus_size_or_err:{total_hits}"})
                    continue
                works, join_rate, pct_refs = join_scopus_dois_to_openalex(sc_recs, oakey)
                if join_rate < args.min_oa_join_rate or len(works) < 50:
                    screen_log.append({"doi": sp.get("doi"), "status": f"skip_low_oa_join:{join_rate:.3f}/{len(works)}"})
                    continue
            else:
                harvest_source = "openalex_fallback"
                total_hits, works, is_complete = fetch_complete_openalex_corpus(sp, oakey, args.max_complete_works)
                if not is_complete or len(works) < 50:
                    screen_log.append({"doi": sp.get("doi"), "status": f"skip_oa_size_or_err:{total_hits}"})
                    continue
                pct_refs = round(sum(1 for w in works if w.get("referenced_works")) / max(1, len(works)), 4)

        lcc_sz = compute_lcc_size(works, net_type, core_top_n=core_n)
        if lcc_sz < 50:
            screen_log.append({"doi": sp.get("doi"), "status": f"skip_small_lcc:{lcc_sz}"})
            continue

        if (not cached_ok) or need_ref_years_backfill:
            ref_years = fetch_cited_reference_years(
                works, oakey, network_type=net_type, core_top_n=core_n, global_cache=global_ref_cache
            )
            save_ref_years_cache(global_ref_cache)
            tmp_c = corp_path.with_suffix(".json.tmp")
            tmp_c.write_text(
                json.dumps(
                    {
                        "query_sig": q_sig,
                        "harvest_source": harvest_source,
                        "source_total_hits": total_hits,
                        "scopus_to_oa_join_rate": join_rate,
                        "pct_works_with_refs": pct_refs,
                        "is_complete_corpus": is_complete,
                        "works": works,
                        "ref_years": ref_years,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp_c.replace(corp_path)

        j_counts[jkey] += 1
        got_by_st[st_name] += 1
        manifest.append(
            {
                **sp,
                "journal_key": jkey,
                "stratum": st_name,
                "network_type": net_type,
                "core_top_n": core_n,
                "harvest_source": harvest_source,
                "corpus_file": safe_id,
                "source_total_hits": total_hits,
                "n_works_joined": len(works),
                "scopus_to_oa_join_rate": join_rate,
                "pct_works_with_refs": pct_refs,
                "n_ref_years_fetched": len(ref_years),
                "verified_lcc_size": lcc_sz,
                "is_complete_corpus": is_complete,
            }
        )
        LOGGER.info(
            "[%s %d/%d] %s | net=%s | hits=%d | joined=%d (%.1f%%, refs=%.1f%%) | lcc=%d | ref_yrs=%d",
            st_name, got_by_st[st_name], quotas[st_name], sp.get("doi"), net_type[:2], total_hits,
            len(works), 100.0 * join_rate, 100.0 * pct_refs, lcc_sz, len(ref_years),
        )

    (PROC_DIR / "layer2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "layer2_screen_log.json").write_text(json.dumps(screen_log, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Layer-2 manifest saved: %d verified corpora (%d skipped)", len(manifest), len(screen_log))


if __name__ == "__main__":
    main()
