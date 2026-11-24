"""Harvest official Scopus ASJC 14xx journal registry and complete BUSI population.

Implements primary-source anchoring:
1. Active Scopus journals with official ASJC 14xx CiteScore percentiles (Serial Title API).
2. Complete English 2020-2026 journal bibliometric articles/reviews in SUBJAREA(BUSI)
   via deterministic per-year Scopus Search API pagination.
3. OpenAlex batch enrichment (50 DOIs/call) including full `locations[]` Green-OA PDF URLs
   and OA landing URLs (cures Wiley/Emerald/T&F HTTP 403 retrieval loss).
4. Publisher-group classification + article-level OA propensity score & IPW weights stored
   directly in `busi_population.parquet`, plus Layer 0 balance JSON & SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.harvest")

REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[2]
EXP_DIR = PAPER_DIR / "experiments"
RAW_DIR = EXP_DIR / "data" / "raw"
PROC_DIR = EXP_DIR / "data" / "processed"
RES_DIR = EXP_DIR / "results"
MANIFEST_PATH = EXP_DIR / "data" / "data_manifest.json"


def load_env_keys(env_path: Path) -> dict[str, str]:
    """Read API keys from .env without logging secret values."""
    keys: dict[str, str] = {}
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys


def norm_issn(val: str | None) -> str:
    """Normalize ISSN/eISSN to 8 uppercase alphanumeric characters."""
    if not val:
        return ""
    s = re.sub(r"[^0-9X]", "", str(val).upper())
    return s if len(s) == 8 else ""


def norm_doi(val: str | None) -> str:
    """Normalize DOI to lowercase bare identifier (no https://doi.org/ prefix)."""
    if not val:
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return s.strip()


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_quartile(pct: int) -> str:
    """Assign official Scopus CiteScore quartile from subject percentile."""
    if pct >= 75:
        return "Q1"
    if pct >= 50:
        return "Q2"
    if pct >= 25:
        return "Q3"
    return "Q4"


def classify_publisher(reg_publisher: str | None, doi: str | None) -> str:
    """Map Scopus publisher string & DOI prefix into standardized publisher group."""
    d = norm_doi(doi)
    pub = (reg_publisher or "").lower()
    if d.startswith("10.1016/") or "elsevier" in pub or "keai" in pub:
        return "Elsevier"
    if d.startswith(("10.1007/", "10.1038/", "10.1057/", "10.1186/")) or "springer" in pub or "palgrave" in pub or "nature" in pub:
        return "Springer Nature"
    if d.startswith("10.1108/") or "emerald" in pub:
        return "Emerald"
    if d.startswith("10.1080/") or "taylor" in pub or "routledge" in pub or "informa" in pub:
        return "Taylor & Francis"
    if d.startswith(("10.1002/", "10.1111/")) or "wiley" in pub or "blackwell" in pub:
        return "Wiley"
    if d.startswith("10.1177/") or "sage" in pub:
        return "SAGE"
    if d.startswith("10.3390/") or "mdpi" in pub:
        return "MDPI"
    if d.startswith("10.3389/") or "frontiers" in pub:
        return "Frontiers"
    if d.startswith("10.1504/") or "inderscience" in pub:
        return "Inderscience"
    return "Other"


def fetch_scopus_asjc14_registry(skey: str, out_path: Path, refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch complete official Scopus BUSI journal registry with ASJC 14xx CiteScore percentiles."""
    if out_path.exists() and not refresh:
        LOGGER.info("Using cached Scopus ASJC 14xx registry at %s", out_path)
        return json.loads(out_path.read_text(encoding="utf-8"))

    url = "https://api.elsevier.com/content/serial/title"
    headers = {"X-ELS-APIKey": skey, "Accept": "application/json"}
    start = 0
    raw_entries: list[dict[str, Any]] = []
    while True:
        resp = requests.get(
            url,
            params={"subj": "BUSI", "content": "journal", "view": "CITESCORE", "start": start, "count": 200},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Scopus Serial Title API HTTP {resp.status_code} at start={start}")
        entries = resp.json().get("serial-metadata-response", {}).get("entry", [])
        if not entries:
            break
        raw_entries.extend(entries)
        LOGGER.info("Fetched Serial Title page start=%d (%d total)", start, len(raw_entries))
        if len(entries) < 200:
            break
        start += 200
        time.sleep(0.15)

    valid: list[dict[str, Any]] = []
    for e in raw_entries:
        cs_list = e.get("citeScoreYearInfoList", {}).get("citeScoreYearInfo", [])
        best_pct: int | None = None
        best_yr: int | None = None
        best_cs: float | None = None
        best_codes: list[str] = []
        for yinfo in cs_list:
            yr_str = yinfo.get("@year")
            if not yr_str or not str(yr_str).isdigit():
                continue
            yr_int = int(yr_str)
            infolist = yinfo.get("citeScoreInformationList", [])
            if not infolist:
                continue
            for ci in infolist[0].get("citeScoreInfo", []):
                ranks = ci.get("citeScoreSubjectRank", [])
                ranks_14 = [
                    (str(rk.get("subjectCode", "")), int(rk["percentile"]))
                    for rk in ranks
                    if str(rk.get("subjectCode", "")).startswith("14") and rk.get("percentile") is not None
                ]
                if ranks_14:
                    mx = max(p for _, p in ranks_14)
                    if best_yr is None or yr_int > best_yr:
                        best_yr = yr_int
                        best_pct = mx
                        best_cs = float(ci["citeScore"]) if ci.get("citeScore") is not None else None
                        best_codes = sorted({c for c, _ in ranks_14})
        if best_pct is None:
            continue
        valid.append(
            {
                "title": e.get("dc:title"),
                "issn": norm_issn(e.get("prism:issn")),
                "eissn": norm_issn(e.get("prism:eIssn")),
                "publisher": e.get("dc:publisher"),
                "citescore_year": best_yr,
                "citescore": best_cs,
                "max_pct_14xx": best_pct,
                "asjc_14_codes": best_codes,
                "quartile": assign_quartile(best_pct),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d active ASJC 14xx journals to %s", len(valid), out_path)
    return valid


def fetch_scopus_busi_articles(skey: str, out_path: Path, refresh: bool = False) -> list[dict[str, Any]]:
    """Fetch complete Scopus SUBJAREA(BUSI) English 2020-2026 journal bibliometric population."""
    if out_path.exists() and not refresh:
        LOGGER.info("Using cached Scopus BUSI articles at %s", out_path)
        return json.loads(out_path.read_text(encoding="utf-8"))

    base_q = (
        'SUBJAREA(BUSI) AND TITLE("bibliometric*") '
        "AND (DOCTYPE(ar) OR DOCTYPE(re)) AND LANGUAGE(english) AND SRCTYPE(j)"
    )
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": skey, "Accept": "application/json"}
    sort_spec = "+coverDate,+publicationName,+citedby-count"

    articles_by_id: dict[str, dict[str, Any]] = {}
    expected_total = 0
    for yr in range(2020, 2027):
        q = f"{base_q} AND PUBYEAR = {yr}"
        start = 0
        yr_expected: int | None = None
        yr_ids: set[str] = set()
        while True:
            resp = requests.get(
                url,
                params={"query": q, "start": start, "count": 25, "sort": sort_spec},
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Scopus Search API HTTP {resp.status_code} at yr={yr} start={start}: {resp.text[:250]}")
            sr = resp.json().get("search-results", {})
            if yr_expected is None:
                yr_expected = int(sr.get("opensearch:totalResults", 0))
                expected_total += yr_expected
            entries = sr.get("entry", [])
            if not entries or (len(entries) == 1 and entries[0].get("error")):
                break
            for e in entries:
                if e.get("error"):
                    continue
                sid = str(e.get("dc:identifier", "")).replace("SCOPUS_ID:", "")
                cover_date = str(e.get("prism:coverDate", ""))
                pub_year = int(cover_date[:4]) if len(cover_date) >= 4 and cover_date[:4].isdigit() else yr
                yr_ids.add(sid)
                articles_by_id[sid] = {
                    "scopus_id": sid,
                    "eid": e.get("eid", ""),
                    "doi": norm_doi(e.get("prism:doi")),
                    "title": e.get("dc:title", ""),
                    "creator": e.get("dc:creator", ""),
                    "publication_name": e.get("prism:publicationName", ""),
                    "issn": norm_issn(e.get("prism:issn")),
                    "eissn": norm_issn(e.get("prism:eIssn")),
                    "pub_year": pub_year,
                    "query_pubyear": yr,
                    "cover_date": cover_date,
                    "subtype": e.get("subtype", ""),
                    "scopus_cited_by": int(e.get("citedby-count", 0) or 0),
                    "scopus_openaccess": str(e.get("openaccess", "0")) == "1",
                }
            start += len(entries)
            if yr_expected is not None and start >= yr_expected:
                break
            time.sleep(0.10)
        LOGGER.info("PUBYEAR=%d: expected=%s, unique_captured=%d", yr, yr_expected, len(yr_ids))

    articles = list(articles_by_id.values())
    LOGGER.info(
        "Total 2020-2026 expected=%d, unique_captured=%d (%.2f%%)",
        expected_total,
        len(articles),
        100.0 * len(articles) / max(1, expected_total),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
    return articles


def _extract_location_urls(work: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract ordered unique legal OA PDF URLs (Green/repository prioritized) and OA landing URLs."""
    locs = work.get("locations") or []
    best = work.get("best_oa_location") or {}
    prim = work.get("primary_location") or {}

    repo_pdfs: list[str] = []
    other_pdfs: list[str] = []
    landing_urls: list[str] = []

    def _add_loc(loc: dict[str, Any]) -> None:
        if not loc:
            return
        src = loc.get("source") or {}
        src_type = str(src.get("type") or "").lower()
        is_oa = bool(loc.get("is_oa"))
        pdf = str(loc.get("pdf_url") or "").strip()
        land = str(loc.get("landing_page_url") or "").strip()
        if pdf and (is_oa or src_type == "repository"):
            if src_type == "repository":
                if pdf not in repo_pdfs:
                    repo_pdfs.append(pdf)
            elif pdf not in other_pdfs:
                other_pdfs.append(pdf)
        if land and is_oa and land not in landing_urls:
            landing_urls.append(land)

    for loc in locs:
        _add_loc(loc)
    _add_loc(best)
    _add_loc(prim)

    # Deduplicate preserving repository-first priority
    all_pdfs: list[str] = []
    for u in repo_pdfs + other_pdfs:
        if u and u not in all_pdfs:
            all_pdfs.append(u)
    return all_pdfs, landing_urls


def enrich_with_openalex(
    articles: list[dict[str, Any]], oakey: str, out_path: Path, refresh: bool = False
) -> dict[str, dict[str, Any]]:
    """Batch-enrich Scopus articles by DOI via OpenAlex including full `locations[]` Green-OA URLs."""
    oa_map: dict[str, dict[str, Any]] = {}
    if out_path.exists() and not refresh:
        oa_map = json.loads(out_path.read_text(encoding="utf-8"))
        LOGGER.info("Loaded %d cached OpenAlex enrichments from %s", len(oa_map), out_path)

    all_dois = sorted({a["doi"] for a in articles if a.get("doi")})
    # Upgrade any cached record missing `oa_green_pdf_urls`
    dois = [d for d in all_dois if (d not in oa_map) or ("oa_green_pdf_urls" not in oa_map[d])]
    LOGGER.info("Enriching %d DOIs via OpenAlex API with locations[] (of %d total)", len(dois), len(all_dois))
    url = "https://api.openalex.org/works"
    batch_size = 50
    for i in range(0, len(dois), batch_size):
        batch = dois[i : i + batch_size]
        filt = "doi:" + "|".join(batch)
        params = {
            "filter": filt,
            "per_page": 100,
            "select": "id,doi,open_access,best_oa_location,locations,cited_by_count,referenced_works_count,primary_location",
            "api_key": oakey,
        }
        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    break
                LOGGER.warning("OpenAlex HTTP %d on batch %d attempt %d", resp.status_code, i, attempt + 1)
                time.sleep(1.0 * (attempt + 1))
            except requests.RequestException as exc:
                LOGGER.warning("OpenAlex error on batch %d attempt %d: %s", i, attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            continue
        for w in resp.json().get("results", []):
            d = norm_doi(w.get("doi"))
            if not d:
                continue
            oa = w.get("open_access") or {}
            best = w.get("best_oa_location") or {}
            prim = w.get("primary_location") or {}
            pdf_url = best.get("pdf_url") or prim.get("pdf_url") or ""
            oa_url = oa.get("oa_url") or best.get("landing_page_url") or ""
            green_pdfs, land_urls = _extract_location_urls(w)
            oa_map[d] = {
                "openalex_id": w.get("id", ""),
                "oa_is_oa": bool(oa.get("is_oa")),
                "oa_status": oa.get("oa_status") or "closed",
                "oa_pdf_url": pdf_url,
                "oa_green_pdf_urls": green_pdfs,
                "oa_landing_urls": land_urls,
                "oa_url": oa_url,
                "oa_cited_by": int(w.get("cited_by_count", 0) or 0),
                "oa_ref_count": int(w.get("referenced_works_count", 0) or 0),
            }
        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(dois):
            LOGGER.info(
                "OpenAlex enriched %d / %d DOIs (total matched %d)",
                min(i + batch_size, len(dois)),
                len(dois),
                len(oa_map),
            )
        time.sleep(0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(oa_map, ensure_ascii=False), encoding="utf-8")
    return oa_map


def attach_propensity_and_compute_balance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit OA propensity score + IPW weights onto active ASJC 14xx population & compute Layer 0 balance."""
    df = df.copy()
    df["oa_propensity_score"] = np.nan
    df["ipw_weight"] = np.nan

    active_mask = df["quartile"].isin(["Q1", "Q2", "Q3", "Q4"])
    matched = df[active_mask].copy()
    n_tot = len(matched)

    strata: dict[str, Any] = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = matched[matched["quartile"] == q]
        oa = sub[sub["is_oa_combined"]]
        closed = sub[~sub["is_oa_combined"]]
        mw_cites = (
            stats.mannwhitneyu(oa["scopus_cited_by"], closed["scopus_cited_by"], alternative="two-sided")
            if len(oa) > 0 and len(closed) > 0
            else None
        )
        mw_year = (
            stats.mannwhitneyu(oa["pub_year"].dropna(), closed["pub_year"].dropna(), alternative="two-sided")
            if len(oa) > 0 and len(closed) > 0
            else None
        )
        has_any_oa_channel = (
            sub["is_oa_combined"]
            & (
                (sub["oa_pdf_url"].fillna("") != "")
                | (sub["oa_green_pdf_count"].fillna(0) > 0)
                | (sub["is_elsevier_doi"])
                | (sub["oa_url"].fillna("") != "")
            )
        )
        strata[q] = {
            "n_articles": int(len(sub)),
            "weight_Wh": round(len(sub) / n_tot, 6) if n_tot else 0.0,
            "n_journals": int(sub["journal_key"].nunique()),
            "n_oa_combined": int(len(oa)),
            "oa_share": round(len(oa) / len(sub), 4) if len(sub) else 0.0,
            "n_has_pdf_url": int((sub["oa_pdf_url"].fillna("") != "").sum()),
            "n_has_green_pdf": int((sub["oa_green_pdf_count"].fillna(0) > 0).sum()),
            "n_elsevier_oa": int((sub["is_oa_combined"] & sub["is_elsevier_doi"]).sum()),
            "n_legal_oa_retrievable_pool": int(has_any_oa_channel.sum()),
            "oa_status_counts": dict(Counter(sub["oa_status"].fillna("unknown"))),
            "publisher_counts_all": dict(Counter(sub["publisher_group"]).most_common()),
            "publisher_counts_oa": dict(Counter(oa["publisher_group"]).most_common()),
            "median_cites_oa": float(oa["scopus_cited_by"].median()) if len(oa) else None,
            "median_cites_closed": float(closed["scopus_cited_by"].median()) if len(closed) else None,
            "mw_cites_p": round(float(mw_cites.pvalue), 6) if mw_cites is not None else None,
            "mean_year_oa": round(float(oa["pub_year"].mean()), 3) if len(oa) else None,
            "mean_year_closed": round(float(closed["pub_year"].mean()), 3) if len(closed) else None,
            "mw_year_p": round(float(mw_year.pvalue), 6) if mw_year is not None else None,
        }

    # Baseline covariates-only propensity (year, cites, citescore, quartile)
    model_df = matched.dropna(subset=["pub_year", "citescore"]).copy()
    y = model_df["is_oa_combined"].astype(int).to_numpy()
    enc_q = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    q_ohe = enc_q.fit_transform(model_df[["quartile"]])
    x_num = np.column_stack(
        [
            (model_df["pub_year"].to_numpy() - 2023.0) / 2.0,
            np.log1p(model_df["scopus_cited_by"].to_numpy()),
            np.log1p(model_df["citescore"].to_numpy()),
        ]
    )
    x_base = np.hstack([x_num, q_ohe])
    clf_base = LogisticRegression(max_iter=500, random_state=20261122)
    clf_base.fit(x_base, y)
    probs_base = clf_base.predict_proba(x_base)[:, 1]
    auc_base = float(roc_auc_score(y, probs_base))

    # Full IPW propensity with L2-regularized quartile x publisher_group interaction + [0.10, 0.90] trim
    model_df["q_x_pub"] = model_df["quartile"].astype(str) + "__" + model_df["publisher_group"].astype(str)
    enc_full = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_ohe = enc_full.fit_transform(model_df[["quartile", "publisher_group", "q_x_pub"]])
    x_full = np.hstack([x_num, cat_ohe])
    clf_full = LogisticRegression(C=0.5, max_iter=1000, random_state=20261122)
    clf_full.fit(x_full, y)
    probs_full = clf_full.predict_proba(x_full)[:, 1]
    auc_full = float(roc_auc_score(y, probs_full))

    clipped_p = np.clip(probs_full, 0.10, 0.90)
    ipw_all = 1.0 / clipped_p
    df.loc[model_df.index, "oa_propensity_score"] = np.round(probs_full, 6)
    df.loc[model_df.index, "ipw_weight"] = np.round(ipw_all, 6)
    ipw_oa = ipw_all[y == 1]
    kish_ess_ratio = float((np.sum(ipw_oa) ** 2) / (len(ipw_oa) * np.sum(ipw_oa ** 2))) if len(ipw_oa) else 0.0

    # Log structural positivity cells (0% or 100% OA in quartile x publisher_group; disclosure)
    positivity_boundary: list[dict[str, Any]] = []
    for cell, grp in model_df.groupby("q_x_pub"):
        rate = float(grp["is_oa_combined"].mean())
        if rate == 0.0 or rate == 1.0:
            positivity_boundary.append({"cell": str(cell), "n": int(len(grp)), "oa_rate": rate})

    balance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": "1.2",
        "total_scopus_busi_raw": int(len(df)),
        "matched_active_asjc14": int(n_tot),
        "unmatched_non_active_14xx": int(len(df) - n_tot),
        "unique_active_journals_with_biblio": int(matched["journal_key"].nunique()),
        "oa_propensity_auc_covariates": round(auc_base, 4),
        "oa_propensity_auc_with_publisher_interaction": round(auc_full, 4),
        "ipw_trim_bounds": [0.10, 0.90],
        "ipw_kish_ess_ratio_oa": round(kish_ess_ratio, 4),
        "ipw_oa_min_median_max": [
            round(float(np.min(ipw_oa)), 3),
            round(float(np.median(ipw_oa)), 3),
            round(float(np.max(ipw_oa)), 3),
        ],
        "positivity_boundary_cells": positivity_boundary,
        "strata": strata,
    }
    return df, balance


def main() -> None:
    """Execute Step 2 harvest (using cached Scopus files by default) and Layer 0 enrichment."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-scopus", action="store_true", help="Force refresh Scopus API caches")
    ap.add_argument("--refresh-openalex", action="store_true", help="Force refresh OpenAlex DOI cache")
    args = ap.parse_args()

    keys = load_env_keys(REPO_ROOT / ".env")
    skey = keys["SCOPUS_API_KEY"]
    oakey = keys["OPENALEX_API_KEY"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    reg_path = RAW_DIR / "scopus_asjc14_registry.json"
    art_path = RAW_DIR / "scopus_busi_articles_raw.json"
    oa_path = RAW_DIR / "openalex_doi_enrichment.json"

    registry = fetch_scopus_asjc14_registry(skey, reg_path, refresh=args.refresh_scopus)
    articles = fetch_scopus_busi_articles(skey, art_path, refresh=args.refresh_scopus)
    oa_map = enrich_with_openalex(articles, oakey, oa_path, refresh=args.refresh_openalex)

    issn_idx: dict[str, dict[str, Any]] = {}
    for j in registry:
        if j["issn"]:
            issn_idx[j["issn"]] = j
        if j["eissn"]:
            issn_idx[j["eissn"]] = j

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for a in articles:
        sid = a["scopus_id"] or a["eid"]
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        jmeta = issn_idx.get(a["issn"]) or issn_idx.get(a["eissn"])
        ometa = oa_map.get(a["doi"], {})
        is_oa_comb = bool(a["scopus_openaccess"] or ometa.get("oa_is_oa"))
        reg_pub = jmeta["publisher"] if jmeta else None
        green_pdfs = ometa.get("oa_green_pdf_urls") or []
        land_urls = ometa.get("oa_landing_urls") or []
        rows.append(
            {
                **a,
                "journal_key": (jmeta["issn"] or jmeta["eissn"]) if jmeta else (a["issn"] or a["eissn"] or a["publication_name"]),
                "reg_title": jmeta["title"] if jmeta else None,
                "reg_publisher": reg_pub,
                "publisher_group": classify_publisher(reg_pub, a["doi"]),
                "is_elsevier_doi": norm_doi(a["doi"]).startswith("10.1016/"),
                "citescore_year": jmeta["citescore_year"] if jmeta else None,
                "citescore": jmeta["citescore"] if jmeta else None,
                "max_pct_14xx": jmeta["max_pct_14xx"] if jmeta else None,
                "quartile": jmeta["quartile"] if jmeta else "UNMATCHED",
                "openalex_id": ometa.get("openalex_id", ""),
                "oa_is_oa": bool(ometa.get("oa_is_oa", False)),
                "is_oa_combined": is_oa_comb,
                "oa_status": ometa.get("oa_status", "unknown"),
                "oa_pdf_url": ometa.get("oa_pdf_url", ""),
                "oa_green_pdf_urls_json": json.dumps(green_pdfs, ensure_ascii=False),
                "oa_green_pdf_count": len(green_pdfs),
                "oa_landing_urls_json": json.dumps(land_urls, ensure_ascii=False),
                "oa_url": ometa.get("oa_url", ""),
                "oa_cited_by": ometa.get("oa_cited_by", None),
                "oa_ref_count": ometa.get("oa_ref_count", None),
            }
        )

    df = pd.DataFrame(rows)
    df, balance = attach_propensity_and_compute_balance(df)

    parquet_path = PROC_DIR / "busi_population.parquet"
    csv_path = PROC_DIR / "busi_population_summary.csv"
    df.to_parquet(parquet_path, index=False)
    df[
        [
            "scopus_id",
            "doi",
            "pub_year",
            "publication_name",
            "publisher_group",
            "journal_key",
            "quartile",
            "citescore",
            "max_pct_14xx",
            "scopus_cited_by",
            "is_oa_combined",
            "oa_status",
            "oa_pdf_url",
            "oa_green_pdf_count",
            "oa_propensity_score",
            "ipw_weight",
        ]
    ].to_csv(csv_path, index=False)

    bal_path = RES_DIR / "layer0_population_balance.json"
    bal_path.write_text(json.dumps(balance, indent=2), encoding="utf-8")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": "1.1",
        "sources": {
            "scopus_serial_title_api": "https://api.elsevier.com/content/serial/title?subj=BUSI&content=journal&view=CITESCORE",
            "scopus_search_api": 'SUBJAREA(BUSI) AND TITLE("bibliometric*") AND PUBYEAR = 2020..2026 AND (DOCTYPE(ar) OR DOCTYPE(re)) AND LANGUAGE(english) AND SRCTYPE(j)',
            "openalex_api": "https://api.openalex.org/works (DOI batch lookup with locations[])",
        },
        "files": {
            "scopus_asjc14_registry.json": {"sha256": sha256_file(reg_path), "rows": len(registry)},
            "scopus_busi_articles_raw.json": {"sha256": sha256_file(art_path), "rows": len(articles)},
            "openalex_doi_enrichment.json": {"sha256": sha256_file(oa_path), "rows": len(oa_map)},
            "busi_population.parquet": {"sha256": sha256_file(parquet_path), "rows": len(df)},
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Layer 0 complete: raw=%d, active_14xx=%d, auc_cov=%.4f, auc_pub_int=%.4f, ess=%.3f",
        len(df),
        balance["matched_active_asjc14"],
        balance["oa_propensity_auc_covariates"],
        balance["oa_propensity_auc_with_publisher_interaction"],
        balance["ipw_kish_ess_ratio_oa"],
    )


if __name__ == "__main__":
    main()
