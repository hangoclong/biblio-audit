"""Layer 1: Neyman-stratified OA retrieval, cap-adjusted MC HT, Two-Stage IPW & Round-1 Regex Audit.

Hardened specification: - Exact 2,000-rep Monte Carlo cap-adjusted inclusion weights (`1 / pi_cap_j`) reported alongside
  journal-balanced capped HT (`cap <= 5/journal`) so Q1 mega-journal capping (up to 3.47x) is transparent.
- Stage-2 retrieval propensity (`retrieved ~ quartile + publisher_group`, L2-regularized, trimmed [0.10, 0.90])
  multiplied by Stage-1 OA IPW for genuine Two-Stage HT x IPW sensitivity.
- Per-journal cap enforced on `oa_browser_queue.csv` (`j_counts[j] + queued_j_counts[j] <= cap`) so
  blocked mega-journals never flood the browser queue.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = EXP_DIR / "src_audit"
sys.path.insert(0, str(SRC_DIR))

from extract_audit_features import parse_pdf_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("p22.layer1")

PROC_DIR = EXP_DIR / "data" / "processed"
PDF_DIR = EXP_DIR / "data" / "pdfs"
FULLTEXT_DIR = EXP_DIR / "data" / "fulltext_api"
RES_DIR = EXP_DIR / "results"
MASTER_SEED = 20261122

PREFERRED_PDF_HOST_RE = re.compile(
    r"(eprints\.|repository\.|pure\.|arxiv\.org|pmc\.ncbi|ncbi\.nlm|europepmc|hal\.|zenodo\.org|"
    r"core\.ac\.uk|scielo|mdpi\.com|frontiersin\.org|link\.springer\.com|nature\.com|springeropen|"
    r"biomedcentral|plos\.org|doaj\.org|econstor|ssoar|repec|orkg|openrepository)",
    re.I,
)


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


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Exact Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def safe_doc_stem(doi: str, scopus_id: str) -> str:
    """Create a filesystem-safe deterministic stem for a paper PDF or text file."""
    key = doi if doi else f"scopus_{scopus_id}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:120]


def is_valid_pdf(path: Path) -> bool:
    """Check whether a local PDF file exists and starts with a valid %PDF header."""
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    header = path.read_bytes()[:1024]
    return header.startswith(b"%PDF") or (b"%PDF-" in header)


def is_valid_elsevier_fulltext(path: Path) -> bool:
    """Check whether a saved Elsevier API text file contains real full text."""
    if not path.exists() or path.stat().st_size < 8_000:
        return False
    txt = path.read_text(encoding="utf-8", errors="replace")
    return len(txt) >= 8_000 and bool(re.search(r"\b(Introduction|Method|Methodology|Results|Discussion|Conclusion)\b", txt, re.I))


def try_fetch_elsevier_fulltext(doi: str, dest: Path, skey: str) -> tuple[bool, int | str]:
    """Fetch complete plain-text full text for a legal OA Elsevier article via official Elsevier API."""
    if is_valid_elsevier_fulltext(dest):
        return True, 200
    if not doi or not skey:
        return False, "no_key_or_doi"
    try:
        resp = requests.get(
            f"https://api.elsevier.com/content/article/doi/{doi}",
            params={"view": "FULL"},
            headers={"X-ELS-APIKey": skey, "Accept": "text/plain"},
            timeout=25,
        )
        if resp.status_code != 200:
            return False, f"els_{resp.status_code}"
        txt = resp.text or ""
        if len(txt) < 8_000 or not re.search(r"\b(Introduction|Method|Methodology|Results|Discussion)\b", txt, re.I):
            return False, "els_abstract_only"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(txt, encoding="utf-8")
        return True, 200
    except requests.RequestException as exc:
        return False, f"els_err:{type(exc).__name__}"


def try_download_pdf(url: str, dest: Path) -> tuple[bool, int | str]:
    """Download a legally open PDF once; strictly respect HTTP 403/blocks (no anti-bot bypass)."""
    if is_valid_pdf(dest):
        return True, 200
    if not url:
        return False, "no_url"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "AcademicResearchAudit/1.2 (mailto:hnlong@hueuni.edu.vn)"},
            timeout=25,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return False, resp.status_code
        content = resp.content
        if len(content) < 10_000 or (not content.startswith(b"%PDF") and b"%PDF-" not in content[:1024]):
            return False, "not_pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return True, 200
    except (requests.RequestException, UnicodeError, ValueError) as exc:
        return False, f"err:{type(exc).__name__}"


def ordered_candidate_pdf_urls(row: pd.Series) -> list[tuple[str, str]]:
    """Build ordered (channel_label, url) list prioritizing Green-OA repositories over publisher PDFs."""
    green_raw = row.get("oa_green_pdf_urls_json") or "[]"
    try:
        green_list = json.loads(green_raw) if isinstance(green_raw, str) else list(green_raw)
    except json.JSONDecodeError:
        green_list = []
    direct = str(row.get("oa_pdf_url") or "").strip()

    seen: set[str] = set()
    preferred: list[tuple[str, str]] = []
    fallback: list[tuple[str, str]] = []
    for u in green_list:
        u = str(u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        if PREFERRED_PDF_HOST_RE.search(u):
            preferred.append(("green_oa_pdf", u))
        else:
            fallback.append(("green_oa_pdf", u))
    if direct and direct not in seen:
        seen.add(direct)
        if PREFERRED_PDF_HOST_RE.search(direct):
            preferred.append(("direct_oa_pdf", direct))
        else:
            fallback.append(("direct_oa_pdf", direct))
    return preferred + fallback


def estimate_cap_inclusion_weights(
    pool_journals: list[str], target: int, cap: int, n_sim: int = 2000, seed: int = MASTER_SEED
) -> dict[str, float]:
    """Monte Carlo exact inclusion probability pi_cap(journal) under sequential capped random draw."""
    n_pool = len(pool_journals)
    if n_pool == 0 or target <= 0:
        return {}
    j_sizes = Counter(pool_journals)
    j_list = sorted(j_sizes.keys())
    j_to_int = {j: i for i, j in enumerate(j_list)}
    arr = np.array([j_to_int[j] for j in pool_journals], dtype=np.int32)
    hits = np.zeros(len(j_list), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for _ in range(n_sim):
        perm = rng.permutation(arr)
        counts = np.zeros(len(j_list), dtype=np.int32)
        got = 0
        for jid in perm:
            if counts[jid] < cap:
                counts[jid] += 1
                got += 1
                if got >= target:
                    break
        hits += counts

    # Per-article inclusion probability in journal j = E[drawn_j] / N_pool_j
    w_map: dict[str, float] = {}
    for idx, j in enumerate(j_list):
        pi_i = max(1e-4, (hits[idx] / n_sim) / j_sizes[j])
        w_map[j] = round(1.0 / pi_i, 6)
    return w_map


def fit_stage2_retrieval_weights(attempt_log: list[dict[str, Any]]) -> dict[str, float]:
    """Fit L2-regularized Stage-2 retrieval propensity among attempted legal-OA records (trimmed [0.10, 0.90])."""
    if len(attempt_log) < 20:
        return {}
    adf = pd.DataFrame(attempt_log)
    if adf["retrieved"].nunique() < 2:
        return {str(r["scopus_id"]): 1.0 for r in attempt_log}
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x = enc.fit_transform(adf[["quartile", "publisher_group"]])
    y = adf["retrieved"].astype(int).to_numpy()
    clf = LogisticRegression(C=0.5, max_iter=500, random_state=MASTER_SEED)
    clf.fit(x, y)
    probs = np.clip(clf.predict_proba(x)[:, 1], 0.10, 0.90)
    return {str(sid): round(float(1.0 / p), 6) for sid, p in zip(adf["scopus_id"], probs)}


def journal_clustered_ht_bootstrap_prop(
    records: list[dict[str, Any]],
    metric: str,
    wh: dict[str, float],
    weight_field: str | None = None,
    n_reps: int = 2000,
    seed: int = MASTER_SEED,
) -> dict[str, float | list[float]]:
    """Journal-clustered stratified HT bootstrap 95% CI with optional cap/IPW sample weights."""
    st_j_vals: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        q = r["quartile"]
        j = r["journal_key"]
        y = 1.0 if r.get(metric) else 0.0
        w = float(r.get(weight_field) or 1.0) if weight_field else 1.0
        st_j_vals[q][j].append((y, w))

    active_q = sorted(st_j_vals.keys())
    if not active_q:
        return {"rate": 0.0, "se": 0.0, "ci95": [0.0, 0.0]}
    w_tot = sum(wh.get(q, 0.0) for q in active_q)
    norm_wh = {q: wh[q] / w_tot for q in active_q}

    def _stratum_mean(pairs: list[tuple[float, float]]) -> float:
        num = sum(y * w for y, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den if den > 0 else 0.0

    point = sum(
        norm_wh[q] * _stratum_mean([p for lst in st_j_vals[q].values() for p in lst])
        for q in active_q
    )

    rng = np.random.default_rng(seed)
    st_keys = {q: sorted(st_j_vals[q].keys()) for q in active_q}
    boots = np.empty(n_reps, dtype=float)
    for b in range(n_reps):
        est_b = 0.0
        for q in active_q:
            jk = st_keys[q]
            g_h = len(jk)
            drawn = rng.integers(0, g_h, size=g_h)
            pairs_h = [p for idx in drawn for p in st_j_vals[q][jk[int(idx)]]]
            est_b += norm_wh[q] * _stratum_mean(pairs_h)
        boots[b] = est_b

    se = float(np.std(boots, ddof=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "rate": round(float(point), 4),
        "se": round(se, 4),
        "ci95": [round(max(0.0, float(lo)), 4), round(min(1.0, float(hi)), 4)],
    }


def parse_targets(spec: str) -> dict[str, int]:
    """Parse 'Q1=100,Q2=50,Q3=50,Q4=50' into quartile target dict."""
    out: dict[str, int] = {}
    for tok in spec.split(","):
        k, v = tok.strip().split("=")
        out[k.strip()] = int(v.strip())
    return out


def main() -> None:
    """Execute stratified Layer-1 retrieval, Round-1 regex audit, browser queue & QA export."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default="Q1=100,Q2=50,Q3=50,Q4=50", help="Neyman targets")
    ap.add_argument("--per-journal-cap", type=int, default=5, help="cap <= 5 articles/journal")
    ap.add_argument("--parse-only", action="store_true", help="Parse local PDFs + Elsevier .txt in locked order without hitting network")
    ap.add_argument("--max-attempts-factor", type=int, default=4, help="Max non-capped attempts multiplier per quartile")
    ap.add_argument("--n-boot", type=int, default=2000, help="Journal-clustered bootstrap replicates")
    args = ap.parse_args()

    targets = parse_targets(args.targets)
    keys = load_env_keys(REPO_ROOT / ".env")
    skey = keys.get("SCOPUS_API_KEY", "")

    pop_path = PROC_DIR / "busi_population.parquet"
    if not pop_path.exists():
        raise FileNotFoundError(f"Run 01_harvest_scopus_busi_universe.py first; missing {pop_path}")
    df = pd.read_parquet(pop_path)
    active = df[df["quartile"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()

    n_active = len(active)
    wh = {q: float((active["quartile"] == q).sum() / n_active) for q in ["Q1", "Q2", "Q3", "Q4"]}

    parsed_rows: list[dict[str, Any]] = []
    attempt_log: list[dict[str, Any]] = []
    retrieval_log: dict[str, Any] = {}
    browser_queue: list[dict[str, Any]] = []

    for q_idx, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        target = targets.get(q, 50)
        has_channel = (
            (active["oa_pdf_url"].fillna("") != "")
            | (active.get("oa_green_pdf_count", pd.Series(0, index=active.index)).fillna(0) > 0)
            | (active.get("is_elsevier_doi", pd.Series(False, index=active.index)).fillna(False))
            | (active["oa_url"].fillna("") != "")
        )
        pool = active[(active["quartile"] == q) & (active["is_oa_combined"]) & has_channel].copy()
        pool = pool.sort_values(["scopus_id", "doi"]).reset_index(drop=True)
        q_rng = np.random.default_rng(MASTER_SEED + 100 + q_idx)
        pool = pool.iloc[q_rng.permutation(len(pool))].reset_index(drop=True)

        # Compute exact Monte Carlo 1 / pi_cap(journal) inclusion weights for this stratum
        cap_w_map = estimate_cap_inclusion_weights(
            pool["journal_key"].astype(str).tolist(), target, args.per_journal_cap, n_sim=2000, seed=MASTER_SEED + 500 + q_idx
        )

        j_counts: Counter[str] = Counter()
        queued_j_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        channel_counts: Counter[str] = Counter()
        got = 0
        attempted = 0
        skipped_cap = 0
        max_attempts = min(len(pool), target * args.max_attempts_factor)

        for perm_rank, (_, row) in enumerate(pool.iterrows()):
            if got >= target or attempted >= max_attempts:
                break
            jkey = str(row["journal_key"])
            if j_counts[jkey] >= args.per_journal_cap:
                skipped_cap += 1
                continue

            attempted += 1
            doi = str(row["doi"] or "")
            sid = str(row["scopus_id"] or "")
            pub_grp = str(row.get("publisher_group") or "Other")
            stem = safe_doc_stem(doi, sid)
            pdf_dest = PDF_DIR / q / f"{stem}.pdf"
            txt_dest = FULLTEXT_DIR / q / f"{stem}.txt"

            doc_path: Path | None = None
            channel = ""
            last_st: int | str = "not_tried"

            if is_valid_pdf(pdf_dest):
                doc_path = pdf_dest
                channel = "local_pdf"
                last_st = 200
            elif is_valid_elsevier_fulltext(txt_dest):
                doc_path = txt_dest
                channel = "elsevier_fulltext_api"
                last_st = 200
            elif not args.parse_only:
                if bool(row.get("is_elsevier_doi")) and skey:
                    ok, st = try_fetch_elsevier_fulltext(doi, txt_dest, skey)
                    last_st = st
                    if ok:
                        doc_path = txt_dest
                        channel = "elsevier_fulltext_api"
                    time.sleep(0.08)

                if doc_path is None:
                    for ch_lbl, url in ordered_candidate_pdf_urls(row):
                        ok, st = try_download_pdf(url, pdf_dest)
                        last_st = st
                        if ok:
                            doc_path = pdf_dest
                            channel = ch_lbl
                            break
                        time.sleep(0.06)

            if doc_path is None:
                status_counts[str(last_st)] += 1
                attempt_log.append({"scopus_id": sid, "quartile": q, "publisher_group": pub_grp, "retrieved": 0})
                # Enforce per-journal cap on browser queue too so blocked mega-journals never flood queue
                if (
                    not args.parse_only
                    and (j_counts[jkey] + queued_j_counts[jkey] < args.per_journal_cap)
                    and str(last_st) in ("403", "202", "not_pdf", "429", "401", "no_url", "els_abstract_only")
                ):
                    cand_urls = ordered_candidate_pdf_urls(row)
                    land_raw = row.get("oa_landing_urls_json") or "[]"
                    try:
                        land_list = json.loads(land_raw) if isinstance(land_raw, str) else list(land_raw)
                    except json.JSONDecodeError:
                        land_list = []
                    queued_j_counts[jkey] += 1
                    browser_queue.append(
                        {
                            "quartile": q,
                            "perm_rank": perm_rank,
                            "doi": doi,
                            "scopus_id": sid,
                            "journal_key": jkey,
                            "publication_name": row["publication_name"],
                            "publisher_group": pub_grp,
                            "oa_status": row["oa_status"],
                            "oa_pdf_url": cand_urls[0][1] if cand_urls else "",
                            "oa_landing_url": land_list[0] if land_list else str(row.get("oa_url") or ""),
                            "dest_pdf_path": str(pdf_dest),
                            "last_http_status": str(last_st),
                        }
                    )
                continue

            try:
                feats = parse_pdf_features(doc_path, title=str(row["title"] or ""))
            except Exception as exc:  # noqa: BLE001
                status_counts[f"parse_err:{type(exc).__name__}"] += 1
                attempt_log.append({"scopus_id": sid, "quartile": q, "publisher_group": pub_grp, "retrieved": 0})
                continue

            status_counts["200"] += 1
            channel_counts[channel] += 1
            j_counts[jkey] += 1
            got += 1
            attempt_log.append({"scopus_id": sid, "quartile": q, "publisher_group": pub_grp, "retrieved": 1})

            w_ipw1 = float(row["ipw_weight"]) if pd.notna(row.get("ipw_weight")) else 1.0
            parsed_rows.append(
                {
                    "scopus_id": sid,
                    "doi": doi,
                    "quartile": q,
                    "perm_rank": perm_rank,
                    "citescore": float(row["citescore"]) if pd.notna(row["citescore"]) else None,
                    "max_pct_14xx": int(row["max_pct_14xx"]) if pd.notna(row["max_pct_14xx"]) else None,
                    "pub_year": int(row["pub_year"]) if pd.notna(row["pub_year"]) else None,
                    "publication_name": row["publication_name"],
                    "publisher_group": pub_grp,
                    "journal_key": jkey,
                    "oa_status": row["oa_status"],
                    "oa_propensity_score": float(row["oa_propensity_score"]) if pd.notna(row.get("oa_propensity_score")) else None,
                    "ipw_weight": w_ipw1,
                    "cap_ht_weight": float(cap_w_map.get(jkey, 1.0)),
                    "retrieval_channel": channel,
                    "doc_rel_path": str(doc_path.relative_to(EXP_DIR)).replace("\\", "/"),
                    "title": row["title"],
                    "scopus_cited_by": int(row["scopus_cited_by"]),
                    **feats,
                }
            )
            if got % 20 == 0:
                LOGGER.info("[%s] Retrieved & parsed %d / %d (attempted=%d, journals=%d)", q, got, target, attempted, len(j_counts))

        max_j_cnt = max(j_counts.values()) if j_counts else 0
        retrieval_log[q] = {
            "target": target,
            "pool_legal_oa": int(len(pool)),
            "attempted": attempted,
            "skipped_journal_cap": skipped_cap,
            "retrieved_parsed": got,
            "unique_journals": len(j_counts),
            "max_per_journal": max_j_cnt,
            "max_journal_share": round(max_j_cnt / max(1, got), 4),
            "retrieval_rate": round(got / attempted, 4) if attempted else 0.0,
            "channel_counts": dict(channel_counts),
            "status_counts": dict(status_counts),
            "publisher_counts": dict(Counter(r["publisher_group"] for r in parsed_rows if r["quartile"] == q).most_common()),
        }
        LOGGER.info(
            "[%s] Done: got=%d/%d | journals=%d (max=%d) | channels=%s",
            q, got, target, len(j_counts), max_j_cnt, dict(channel_counts),
        )

    # Fit Stage-2 retrieval propensity & attach combined Two-Stage IPW weight
    stage2_w_map = fit_stage2_retrieval_weights(attempt_log)
    for r in parsed_rows:
        w2 = float(stage2_w_map.get(r["scopus_id"], 1.0))
        r["stage2_retrieval_ipw"] = w2
        r["two_stage_ipw_weight"] = round(float(r["ipw_weight"]) * w2, 6)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    full_json = RES_DIR / "layer1_parsed_records.json"
    full_json.write_text(json.dumps(parsed_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.parse_only:
        bq_path = PROC_DIR / "oa_browser_queue.csv"
        pd.DataFrame(browser_queue).to_csv(bq_path, index=False)
        LOGGER.info("Saved %d capped blocked legal-OA candidates to %s", len(browser_queue), bq_path)

    emp = [r for r in parsed_rows if not r["is_guide"]]
    metrics = [
        "uses_scopus",
        "uses_wos",
        "uses_openalex",
        "exact_query_printed",
        "snapshot_date_reported",
        "prisma_reported",
        "counting_method_disclosed",
        "fractional_disclosed",
        "author_pub_fractionalized",
        "null_model_reported",
        "stability_check_reported",
        "data_or_code_shared",
        "sw_vosviewer",
        "sw_bibliometrix",
        "sw_citespace",
        "mentions_bc",
        "mentions_cc",
    ]
    by_q: dict[str, Any] = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = [r for r in emp if r["quartile"] == q]
        n = len(sub)
        m_stats: dict[str, Any] = {}
        for m in metrics:
            k = sum(1 for r in sub if r.get(m))
            lo, hi = wilson_ci(k, n)
            m_stats[m] = {"k": k, "n": n, "rate": round(k / n, 4) if n else 0.0, "ci95": [lo, hi]}
        by_q[q] = {
            "n_empirical": n,
            "unique_journals": len({r["journal_key"] for r in sub}),
            "n_guides_excluded": sum(1 for r in parsed_rows if r["quartile"] == q and r["is_guide"]),
            "metrics": m_stats,
        }

    ht_capped: dict[str, Any] = {}
    ht_cap_adj: dict[str, Any] = {}
    ht_two_stage_ipw: dict[str, Any] = {}
    for m_idx, m in enumerate(metrics):
        ht_capped[m] = journal_clustered_ht_bootstrap_prop(emp, m, wh, weight_field=None, n_reps=args.n_boot, seed=MASTER_SEED + 1000 + m_idx)
        ht_cap_adj[m] = journal_clustered_ht_bootstrap_prop(emp, m, wh, weight_field="cap_ht_weight", n_reps=args.n_boot, seed=MASTER_SEED + 2000 + m_idx)
        ht_two_stage_ipw[m] = journal_clustered_ht_bootstrap_prop(emp, m, wh, weight_field="two_stage_ipw_weight", n_reps=args.n_boot, seed=MASTER_SEED + 3000 + m_idx)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": "1.2",
        "audit_round": "round1_regex_cross_audit",
        "master_seed": MASTER_SEED,
        "targets": targets,
        "per_journal_cap": args.per_journal_cap,
        "total_retrieved_parsed": len(parsed_rows),
        "total_empirical": len(emp),
        "total_unique_journals": len({r["journal_key"] for r in emp}),
        "total_guides_excluded": len(parsed_rows) - len(emp),
        "stratum_weights_Wh": {k: round(v, 5) for k, v in wh.items()},
        "retrieval_log": retrieval_log,
        "ht_capped_clustered": ht_capped,
        "ht_article_weighted_cap_adjusted": ht_cap_adj,
        "ht_two_stage_ipw_sensitivity": ht_two_stage_ipw,
        "by_quartile": by_q,
    }
    sum_path = RES_DIR / "layer1_reporting_census.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Layer 1 complete: empirical=%d (%d journals), summary=%s", len(emp), len({r['journal_key'] for r in emp}), sum_path)


if __name__ == "__main__":
    main()
