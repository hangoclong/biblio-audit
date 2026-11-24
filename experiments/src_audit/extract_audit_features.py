"""Hardened deterministic Round-1 regex cross-audit parser (PDF + Elsevier API full text).

Hardened specification: 1. Supports BOTH PDF files (PyMuPDF) and Elsevier Full-Text API plain-text (.txt) files.
2. Unwraps intra-paragraph soft line breaks (\\n) before sentence splitting so multi-line
   negations ('Scopus\\nor other repositories were excluded') are never severed.
3. Anchors References stripping past 55% of document and skips Biblioshiny Table 1 rows
   ('7 References 59367') that previously amputated mid-paper body text.
4. Eliminates data_or_code_shared boilerplate FPs ('upon request', 'no associated data',
   'available from Scopus') by requiring explicit repository URLs/DOIs or attached datasets.
5. Separates network_fractional_disclosed (VOSviewer/link fractional counting) from
   author_pub_fractionalized ('articles fractionalized') and blocks binary-counting FPs.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz

GUIDE_TITLE_RE = re.compile(
    r"\b(how to (conduct|design|perform|write)|guidelines? for|caveats for|eight tips|"
    r"tutorial|editorial|overview and guidelines|best practices for bibliometric)\b",
    re.I,
)

NEGATION_WINDOW_RE = re.compile(
    r"\b(omit(?:ted|ting)?|exclud(?:ed|ing)|rather than|instead of|alternative(?:s)?|"
    r"not (?:used|chosen|selected|included|employed|applied|conducted|performed)|"
    r"did not|do not|never|without|unlike|compared (?:to|with)|other databases? such as|"
    r"databases? such as|more coverage than|future (?:research|stud(?:y|ies)|work)|"
    r"recommended for future|limitations?|binary counting)\b",
    re.I,
)

DATA_NEGATION_RE = re.compile(
    r"\b(upon\s+(?:reasonable\s+)?request|on\s+(?:reasonable\s+)?request|no\s+associated\s+data|"
    r"not\s+(?:openly\s+)?available|not\s+applicable|no\s+data\s+(?:were|was)\s+(?:used|generated)|"
    r"available\s+from\s+(?:the\s+)?(?:scopus|web\s+of\s+science|wos|corresponding\s+author)|"
    r"sourced\s+from\s+(?:the\s+)?(?:scopus|web\s+of\s+science))\b",
    re.I,
)

SCOPUS_RE = re.compile(r"\bScopus\b")
WOS_RE = re.compile(r"\b(Web of Science|WoS|Clarivate|SSCI|Social Sciences Citation Index|SCIE)\b")
OPENALEX_RE = re.compile(r"\bOpenAlex\b", re.I)

# Expanded case-insensitive Boolean & field-tag query syntax (closes 9 FNs in QA-50)
STRICT_QUERY_RE = re.compile(
    r"(TITLE-ABS-KEY|TITLE-ABS|ABS-KEY|\bTITLE\s*\(|\bALL\s*\(|\bTS\s*=|\bTI\s*=|\bAB\s*=|\bTOPIC\s*:|"
    r'["\'][^"\']{3,65}["\']\s+(?:AND|OR|and|or)\s+["\'][^"\']{3,65}["\'])'
)

PRISMA_RE = re.compile(r"\bPRISMA(?:-S|-ScR)?\b")
NETWORK_FRAC_RE = re.compile(r"\bfractional[- ]count(?:ing|ed|s)?\b", re.I)
AUTHOR_FRAC_RE = re.compile(r"\b(?:articles?\s+fractionalized|fractionalized\s+(?:articles?|publications?|authorship))\b", re.I)
FULL_COUNT_RE = re.compile(r"\bfull[- ]count(?:ing|ed|s)?\b", re.I)
NULL_MODEL_RE = re.compile(
    r"\b(null\s+model|configuration\s+model|maslov[- ]sneppen|degree[- ]preserving\s+(?:null|rewir)|"
    r"bipartite\s+null|randomized\s+network\s+null)\b",
    re.I,
)
STABILITY_RE = re.compile(
    r"\b(seed\s+stability|multi[- ]seed|normalized\s+mutual\s+information|\bNMI\b|"
    r"adjusted\s+rand\s+index|\bARI\b|cluster\s+stability|consensus\s+clustering)\b",
    re.I,
)
SOFTWARE_PATTERNS = {
    "vosviewer": re.compile(r"\bVOSviewer\b", re.I),
    "bibliometrix": re.compile(r"\b(bibliometrix|biblioshiny)\b", re.I),
    "citespace": re.compile(r"\bCiteSpace\b", re.I),
    "gephi": re.compile(r"\bGephi\b", re.I),
    "scimat": re.compile(r"\bSciMAT\b", re.I),
}
# Require explicit public repository URL/DOI or explicit attached dataset/code file (not bare "Data availability")
DATA_CODE_RE = re.compile(
    r"(github\.com/[A-Za-z0-9_.-]+|zenodo\.org/record|10\.5281/zenodo|osf\.io/[A-Za-z0-9]+|"
    r"figshare\.com|10\.6084/m9\.figshare|data\.mendeley\.com|dataverse\.|kaggle\.com/datasets|"
    r"(?:dataset|data|code|scripts?)\s+(?:is|are)\s+(?:publicly\s+)?(?:available|deposited|archived)\s+(?:at|in|on)\s+https?://|"
    r"replication\s+(?:package|data|code)\s+(?:is|are)\s+available)",
    re.I,
)
SNAPSHOT_DATE_RE = re.compile(
    r"\b(retrieved|downloaded|extracted|searched|collected|conducted)\s+(?:on|in)\s+"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+20\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2})",
    re.I,
)
BC_RE = re.compile(r"\bbibliographic\s+coupling\b", re.I)
CC_RE = re.compile(r"\bco-?citation\b", re.I)

PRESERVE_HYPHEN_PREFIX_RE = re.compile(
    r"(TITLE|ABS|KEY|co|full|fractional|multi|non|re|pre|sub|data|faith|woke|self|cross|inter|micro|macro|socio|decision)$",
    re.I,
)
# Standalone References heading (rejects table rows like '7 References 59367')
REF_HEADING_RE = re.compile(r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:References|Bibliography|Works\s+Cited)\s*\n(?!\s*\d{2,6}\b)", re.I)
APP_HEADING_RE = re.compile(r"(?:^|\n)\s*Appendix\b", re.I)
ABBREV_DOT_RE = re.compile(r"(?:\b(?:e\.g|i\.e|et\s+al|vs|Fig|Tab|Eq|No|Vol|pp|Dr|Prof)\.$|\b[ei]\.$)", re.I)


def normalize_pdf_text(raw: str) -> str:
    """Normalize ligatures, soft hyphens, Unicode dashes, curly quotes, and unwrap soft line breaks."""
    txt = unicodedata.normalize("NFKC", raw)
    txt = txt.replace("\xad", "")
    txt = txt.replace("\u00a0", " ")
    txt = re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", txt)
    txt = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", txt)
    txt = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', txt)

    def _join_wrap(match: re.Match[str]) -> str:
        prefix = match.group(1)
        if PRESERVE_HYPHEN_PREFIX_RE.search(prefix):
            return prefix + "-"
        return prefix

    txt = re.sub(r"([A-Za-z]{2,})-\s*\n\s*(?=[A-Za-z])", _join_wrap, txt)
    return txt


def unwrap_soft_newlines(txt: str) -> str:
    """Collapse single intra-paragraph line wraps to spaces while preserving paragraph breaks (\\n\\n)."""
    txt = re.sub(r"\r\n?", "\n", txt)
    # Protect double newlines as paragraph boundaries
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    # Replace single line wraps with a single space so sentence clauses stay intact
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)
    return txt


def extract_body_pages(doc_path: Path) -> list[tuple[int, str]]:
    """Extract normalized page/chunk texts from either a PDF or Elsevier plain-text fulltext file."""
    if doc_path.suffix.lower() == ".txt":
        raw = normalize_pdf_text(doc_path.read_text(encoding="utf-8", errors="replace"))
        # Strip References past 55% of text
        cut_start = int(len(raw) * 0.55)
        m = REF_HEADING_RE.search(raw[cut_start:])
        body = raw[: cut_start + m.start()] if m else raw
        # Split into pseudo-pages of ~4,000 chars for locator reporting
        chunks = [(i + 1, body[pos : pos + 4000]) for i, pos in enumerate(range(0, len(body), 4000))]
        return chunks or [(1, "")]

    doc = fitz.open(doc_path)
    pages: list[tuple[int, str]] = []
    try:
        for i, page in enumerate(doc, start=1):
            pages.append((i, normalize_pdf_text(page.get_text("text"))))
    finally:
        doc.close()

    total = len(pages)
    ref_page_idx: int | None = None
    # Start scanning past 55% of pages to avoid amputating mid-paper Biblioshiny Table 1 rows (Fix F3)
    for idx in range(max(0, int(total * 0.55)), total):
        pno, txt = pages[idx]
        m = REF_HEADING_RE.search(txt)
        if m:
            ref_page_idx = idx
            pages[idx] = (pno, txt[: m.start()])
            break

    if ref_page_idx is None:
        return pages

    kept = list(pages[: ref_page_idx + 1])
    in_appendix = False
    for idx in range(ref_page_idx + 1, total):
        pno, txt = pages[idx]
        if APP_HEADING_RE.search(txt):
            in_appendix = True
        if in_appendix:
            kept.append((pno, txt))
    return kept


def _sentence_clause(txt: str, start: int, end: int) -> str:
    """Return the sentence/clause around [start, end] on unwrapped prose without splitting on abbreviations."""
    left_slice = txt[max(0, start - 180) : start]
    right_slice = txt[end : min(len(txt), end + 130)]

    l_cut = 0
    for m in re.finditer(r"(?:\n\n|[.;])", left_slice):
        ch = m.group(0)
        if ch == ".":
            prefix = left_slice[: m.end()]
            if ABBREV_DOT_RE.search(prefix):
                continue
            if m.end() < len(left_slice) and left_slice[m.end()].isdigit():
                continue
        l_cut = m.end()

    r_cut = len(right_slice)
    for m in re.finditer(r"(?:\n\n|[.;])", right_slice):
        ch = m.group(0)
        if ch == ".":
            prefix = right_slice[: m.end()]
            if ABBREV_DOT_RE.search(prefix):
                continue
            if m.end() < len(right_slice) and right_slice[m.end()].isdigit():
                continue
        r_cut = m.start()
        break

    return left_slice[l_cut:] + txt[start:end] + right_slice[:r_cut]


def _first_positive_match(
    pages: list[tuple[int, str]],
    pattern: re.Pattern[str],
    apply_negation_guard: bool = False,
    neg_pattern: re.Pattern[str] = NEGATION_WINDOW_RE,
) -> tuple[bool, int | None, str]:
    """Return (matched, page_num, snippet) skipping negation/comparison contexts within the unwrapped sentence."""
    for pno, raw_txt in pages:
        txt = unwrap_soft_newlines(raw_txt)
        for m in pattern.finditer(txt):
            start = max(0, m.start() - 90)
            end = min(len(txt), m.end() + 110)
            snippet = re.sub(r"\s+", " ", txt[start:end]).strip()
            if apply_negation_guard:
                clause = _sentence_clause(txt, m.start(), m.end())
                if neg_pattern.search(clause):
                    continue
            return True, pno, snippet
    return False, None, ""


def parse_pdf_features(doc_path: Path, title: str = "") -> dict[str, Any]:
    """Deterministically extract Round-1 regex cross-audit features with verbatim snippets & page locators."""
    pages = extract_body_pages(doc_path)
    is_guide = bool(GUIDE_TITLE_RE.search(title))

    scopus_hit, scopus_pg, scopus_sn = _first_positive_match(pages, SCOPUS_RE, apply_negation_guard=True)
    wos_hit, wos_pg, wos_sn = _first_positive_match(pages, WOS_RE, apply_negation_guard=True)
    oa_hit, oa_pg, oa_sn = _first_positive_match(pages, OPENALEX_RE, apply_negation_guard=True)
    q_hit, q_pg, q_sn = _first_positive_match(pages, STRICT_QUERY_RE, apply_negation_guard=False)
    snap_hit, snap_pg, snap_sn = _first_positive_match(pages, SNAPSHOT_DATE_RE, apply_negation_guard=False)
    prisma_hit, prisma_pg, prisma_sn = _first_positive_match(pages, PRISMA_RE, apply_negation_guard=False)
    net_frac_hit, frac_pg, frac_sn = _first_positive_match(pages, NETWORK_FRAC_RE, apply_negation_guard=True)
    auth_frac_hit, _, _ = _first_positive_match(pages, AUTHOR_FRAC_RE, apply_negation_guard=False)
    full_hit, full_pg, full_sn = _first_positive_match(pages, FULL_COUNT_RE, apply_negation_guard=True)
    null_hit, null_pg, null_sn = _first_positive_match(pages, NULL_MODEL_RE, apply_negation_guard=True)
    stab_hit, stab_pg, stab_sn = _first_positive_match(pages, STABILITY_RE, apply_negation_guard=True)
    dc_hit, dc_pg, dc_sn = _first_positive_match(pages, DATA_CODE_RE, apply_negation_guard=True, neg_pattern=DATA_NEGATION_RE)
    bc_hit, _, _ = _first_positive_match(pages, BC_RE, apply_negation_guard=False)
    cc_hit, _, _ = _first_positive_match(pages, CC_RE, apply_negation_guard=False)

    sw_hits: dict[str, bool] = {}
    for sw_name, sw_pat in SOFTWARE_PATTERNS.items():
        hit, _, _ = _first_positive_match(pages, sw_pat, apply_negation_guard=False)
        sw_hits[sw_name] = hit

    return {
        "doc_file": doc_path.name,
        "source_format": doc_path.suffix.lower().lstrip("."),
        "n_body_pages": len(pages),
        "is_guide": is_guide,
        "uses_scopus": scopus_hit,
        "uses_wos": wos_hit,
        "uses_openalex": oa_hit,
        "exact_query_printed": q_hit,
        "snapshot_date_reported": snap_hit,
        "prisma_reported": prisma_hit,
        "fractional_disclosed": net_frac_hit,
        "author_pub_fractionalized": auth_frac_hit,
        "full_counting_disclosed": full_hit,
        "counting_method_disclosed": bool(net_frac_hit or full_hit),
        "null_model_reported": null_hit,
        "stability_check_reported": stab_hit,
        "data_or_code_shared": dc_hit,
        "mentions_bc": bc_hit,
        "mentions_cc": cc_hit,
        **{f"sw_{k}": v for k, v in sw_hits.items()},
        "snippets": {
            "scopus": {"page": scopus_pg, "text": scopus_sn},
            "wos": {"page": wos_pg, "text": wos_sn},
            "query": {"page": q_pg, "text": q_sn},
            "snapshot": {"page": snap_pg, "text": snap_sn},
            "prisma": {"page": prisma_pg, "text": prisma_sn},
            "fractional": {"page": frac_pg, "text": frac_sn},
            "full_counting": {"page": full_pg, "text": full_sn},
            "null_model": {"page": null_pg, "text": null_sn},
            "stability": {"page": stab_pg, "text": stab_sn},
            "data_code": {"page": dc_pg, "text": dc_sn},
        },
    }
