"""Clean-room multi-database parsers for bibliometric exports.

Supports six standard bibliometric export formats with automatic format detection,
multi-file ingestion with deduplication, and VOSviewer 2-column thesaurus loading:
1. Scopus CSV (both legacy comma-author and 2023-2026 semicolon-author `References` formats)
2. Web of Science Tab-Delimited (`.txt` / `.tsv`)
3. Web of Science Plain Text (`FN Clarivate Analytics Web of Science` / `VR 1.0`)
4. OpenAlex JSON (API list / `{"results": [...]}` / `{"works": [...]}`) and OpenAlex CSV (`|`-delimited)
5. Dimensions CSV (with or without 1-line `"About the data..."` preamble; pipe-delimited `Cited references`)
6. Lens.org CSV (natural-order `Author/s` with surname-particle handling and 15-digit Lens ID `References`)
"""

from __future__ import annotations

import codecs
import csv
import html
import io
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from .exceptions import FormatDetectionError

# Guard against Windows 32-bit C-long overflow (`OverflowError` if passed sys.maxsize on 64-bit Windows)
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

SUPPORTED_FORMATS = ("auto", "scopus", "wos", "wos_plaintext", "openalex", "openalex_csv", "dimensions", "lens")

# SICI-safe DOI regex: preserves `<`, `>`, `;`, `:` inside legacy Wiley/Springer SICI DOIs
_DOI_EXTRACT_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"']+)", re.IGNORECASE)
_LENS_ID_RE = re.compile(r"^\d{3}-\d{3}-\d{3}-\d{3}-\d{3}$")
_OPENALEX_WORK_RE = re.compile(r"^(?:https?://openalex\.org/)?(w\d+)$", re.IGNORECASE)
_OPENALEX_AUTH_RE = re.compile(r"^(?:https?://openalex\.org/)?(a\d+)$", re.IGNORECASE)

_YEAR_CLOSE_RE = re.compile(r"\((?:18|19|20)\d{2}[a-z]?\)\s*(?:,\s*(?:pp?\.\s*[\w\-–]+|https?://\S+|doi:\s*\S+))*\.?\s*$", re.IGNORECASE)
_PARTICLE_PAT = r"(?:(?:van(?:\s+der|\s+den)?|von(?:\s+der)?|de(?:\s+la|\s+los)?|del|della|da|di|du|la|le|les|al|el|ben|st\.?)\s+)"
_WORD_PAT = r"[^\W\d_](?:[^\W\d_]|['’‘\-])*"
_INIT_PAT = r"(?:[^\W\d_]\.(?:\s*-\s*[^\W\d_]\.)*|[^\W\d_]\.?|\([^\W\d_]\.?\)\.?)"
_BARE_AUTHOR_TOK_RE = re.compile(
    rf"^(?:{_PARTICLE_PAT})?{_WORD_PAT}(?:\s+{_WORD_PAT}){{0,2}}\s+(?:{_INIT_PAT}\s*){{1,4}}$",
    re.UNICODE | re.IGNORECASE,
)
_BARE_AUTHOR_COMMA_RE = re.compile(
    rf"^(?:{_PARTICLE_PAT})?{_WORD_PAT}(?:\s+{_WORD_PAT}){{0,2}},\s*(?:{_INIT_PAT}\s*){{1,4}}$",
    re.UNICODE | re.IGNORECASE,
)
_SURNAME_PARTICLE_START_RE = re.compile(rf"^{_PARTICLE_PAT}", re.UNICODE | re.IGNORECASE)
_UNICODE_INITIALS_RE = re.compile(r"^(?:[^\W\d_]\.(?:\s*-\s*[^\W\d_]\.)*\s*|[^\W\d_]\.?\s*){1,4}$", re.UNICODE)
_BARE_INITIALS_STRICT_RE = re.compile(
    r"^(?:[^\W\d_]\.(?:\s*-\s*[^\W\d_]\.)*\s*|\([^\W\d_]\.?\)\.?\s*){1,4}$",
    re.UNICODE,
)
_TITLE_NAME_WORD = r"[^\W\d_a-z](?:[^\W\d_A-Z]|['’‘\-](?=[^\W\d_]))*(?:['’‘\-][^\W\d_a-z][^\W\d_A-Z]*)*"
_ALLCAPS_NAME_WORD = r"[A-ZÀ-ÖØ-Þ]{2,}(?:['’‘\-][A-ZÀ-ÖØ-Þ]{2,})*"
_BARE_FULLNAME_TITLE_RE = re.compile(
    rf"^(?:{_PARTICLE_PAT})?{_TITLE_NAME_WORD}(?:\s+(?:{_PARTICLE_PAT})?{_TITLE_NAME_WORD}){{1,3}}$",
    re.UNICODE,
)
_BARE_FULLNAME_CAPS_RE = re.compile(
    rf"^{_ALLCAPS_NAME_WORD}(?:\s+{_ALLCAPS_NAME_WORD}){{1,2}}$",
    re.UNICODE,
)
_SINGLE_NAME_WORD_RE = re.compile(rf"^{_TITLE_NAME_WORD}$", re.UNICODE)
_ET_AL_OR_EDS_RE = re.compile(r"^(?:et\s+al\.?|\(?eds?\.?\)?)$", re.IGNORECASE)
_BARE_YEAR_STUB_RE = re.compile(r"^(?:1[5-9]|20)\d{2}[a-z]?$")
_BARE_NUMERIC_STUB_RE = re.compile(r"^[\d\s]+$")
_BARE_INITIALS_COMPACT_RE = re.compile(r"^(?:[a-z]\s+){1,5}[a-z]$")
_BARE_BIB_NOISE_RE = re.compile(
    r"^(?:pp?|vol|no|ch|sec|art|fig|tab|et al|eds?|ibid|op cit|loc cit|anonymous|unknown)(?:\s+[\d\s]+)?$"
)
_NON_AUTHOR_LEAD_WORDS = frozenset({
    "the", "a", "an", "in", "on", "of", "for", "and", "or", "to", "by", "at", "from",
    "journal", "proceedings", "report", "annual", "global", "world", "united", "european",
    "international", "national", "act", "section", "chapter", "part", "table", "figure",
    "appendix", "supplement", "compare", "see", "cf", "hello", "review", "survey", "guide",
    "manual", "handbook", "commission", "committee", "council", "ministry", "department",
    "association", "institute", "university", "foundation", "organization", "organisation",
    "bank", "fund", "board", "agency", "authority", "office", "center", "centre",
})

_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "phd", "ph.d.", "md", "m.d."})
_PARTICLES = frozenset({
    "van", "von", "de", "del", "della", "da", "di", "du", "la", "le", "les",
    "der", "den", "ter", "ten", "al", "el", "ben", "bin", "ibn", "st", "st.", "dos", "das",
})
_MISSING_SENTINELS = frozenset({"", "nan", "none", "null", "n/a", "na", "[no author name available]", "[anonymous]"})


def _strip_unbalanced_trailing(doi: str) -> str:
    """Strip trailing citation punctuation without corrupting balanced brackets/parens inside SICI DOIs."""
    s = doi.strip()
    while s:
        last = s[-1]
        if last in ".,;":
            s = s[:-1]
        elif last == ")" and s.count(")") > s.count("("):
            s = s[:-1]
        elif last == "]" and s.count("]") > s.count("["):
            s = s[:-1]
        elif last == ">" and s.count(">") > s.count("<"):
            s = s[:-1]
        else:
            break
    return s


def canonicalize_doi(raw: Any) -> str:
    """Normalize a DOI string cleanly, preserving SICI angle brackets and internal semicolons."""
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if s.lower() in _MISSING_SENTINELS:
        return ""
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)+", "", s, flags=re.IGNORECASE).strip()
    m = _DOI_EXTRACT_RE.search(s)
    if not m:
        return ""
    return _strip_unbalanced_trailing(m.group(1)).lower()


def _parse_int(val: Any, default: int = 0) -> int:
    try:
        s = str(val or "").strip().replace(",", "")
        if not s or s.lower() in _MISSING_SENTINELS:
            return default
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _protect_html_entities_and_sici(text: str) -> tuple[str, list[str]]:
    """Replace HTML entities (`&#x0141;`, malformed `&#X.008;`) and SICI DOIs containing `;` with temporary sentinels."""
    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        stashed.append(m.group(0))
        return f"@@TOK{len(stashed) - 1}@@"

    out = re.sub(r"&(?:#[xX]\.?[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);", _stash, text)
    # Require >= 1 non-space char AFTER `;` so trailing reference-separator `;` is NEVER swallowed!
    out = re.sub(r"\b10\.\d{4,9}/[^\s\"']+;[^\s\"']+", _stash, out, flags=re.IGNORECASE)
    return out, stashed


_MALFORMED_HEX_ENTITY_RE = re.compile(r"&#[xX]\.[0-9a-fA-F]+;\.?")


def _restore_stashed(text: str, stashed: list[str]) -> str:
    out = text
    for idx, val in enumerate(stashed):
        out = out.replace(f"@@TOK{idx}@@", val)
    return _MALFORMED_HEX_ENTITY_RE.sub("", out)


def _is_bare_initials_part(text: str) -> bool:
    return (
        len(text) <= 15
        and "," not in text
        and not any(ch.isdigit() for ch in text)
        and bool(_BARE_INITIALS_STRICT_RE.match(text))
    )


def _is_bare_initials_author(text: str, allow_lower_start: bool = False) -> bool:
    if not text or len(text) > 55 or "10." in text or any(ch.isdigit() for ch in text):
        return False
    if any(ch in text for ch in ":?!/\"|—"):
        return False
    if not allow_lower_start and not (text[0].isupper() or bool(_SURNAME_PARTICLE_START_RE.match(text))):
        return False
    return bool(_BARE_AUTHOR_TOK_RE.match(text) or _BARE_AUTHOR_COMMA_RE.match(text))


def _is_bare_fullname_author(text: str) -> bool:
    if not text or len(text) > 50 or "," in text or "10." in text or any(ch.isdigit() for ch in text):
        return False
    if any(ch in text for ch in ":?!/\"|—()[]"):
        return False
    first_word = text.split()[0].lower().rstrip(".")
    if first_word in _NON_AUTHOR_LEAD_WORDS:
        return False
    return bool(_BARE_FULLNAME_TITLE_RE.match(text) or _BARE_FULLNAME_CAPS_RE.match(text))


def split_scopus_references(raw: str) -> list[str]:
    """Split Scopus `References` safely across both legacy and 2023-2026 semicolon-author formats.

    In 2023-2026 Scopus CSV exports, semicolons separate co-authors *within* a single reference
    (`Cheng M.; Xue Y.; Cui M., Title..., (2023)` or full-name `Devlin Jacob; Chang Ming-Wei, ...`),
    so naive `.split(';')` shreds references into author fragments. This parser protects HTML
    entities (including malformed `&#X.008;`) and SICI DOIs, splits on `;`, preserves parentheticals
    containing semicolons, and re-joins bare-author/initial/mononym tokens onto their parent reference.
    """
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    protected, stashed = _protect_html_entities_and_sici(text)
    raw_parts = [p.strip() for p in protected.split(";") if p.strip()]
    if not raw_parts:
        return []

    # Fast path: OpenAlex URIs, pure DOIs, or Lens IDs already cleanly semicolon-separated
    first_restored = _restore_stashed(raw_parts[0], stashed)
    if (
        _OPENALEX_WORK_RE.match(first_restored)
        or _LENS_ID_RE.match(first_restored)
        or re.match(r"^(?:doi:)?10\.\d{4,9}/", first_restored, flags=re.IGNORECASE)
    ):
        return [_restore_stashed(p, stashed) for p in raw_parts]

    parts = [s for p in raw_parts if (s := html.unescape(_restore_stashed(p, stashed)).strip())]
    merged: list[str] = []
    pending_authors: list[str] = []
    n_parts = len(parts)

    for idx, restored in enumerate(parts):
        has_next = (idx + 1) < n_parts
        nxt = parts[idx + 1] if has_next else ""

        # Re-join semicolons inside an unclosed parenthetical e.g. "(Release Nos. 33-9106; 34-61469; FR-82), (2010)"
        if not pending_authors and merged and merged[-1].count("(") > merged[-1].count(")"):
            merged[-1] = f"{merged[-1]}; {restored}"
            continue

        # Attach stray punctuation-only fragments (e.g., "." after malformed "Madsen D.&#X.008;.;")
        if not any(ch.isalnum() for ch in restored):
            if pending_authors:
                pending_authors[-1] = f"{pending_authors[-1]}{restored}"
            elif merged:
                merged[-1] = f"{merged[-1]}{restored}"
            continue

        # Attach bare `et al.` / `(Eds.)` tokens cleanly (never onto bare numeric/page stubs)
        if _ET_AL_OR_EDS_RE.match(restored):
            if pending_authors:
                pending_authors.append(restored)
            elif merged and normalize_reference_key(merged[-1]):
                merged[-1] = f"{merged[-1]}; {restored}"
            continue

        # Attach bare initials split off from surname (`Barendregt; W.` or `Hastie T.; T. R.`)
        if _is_bare_initials_part(restored):
            if pending_authors:
                pending_authors[-1] = f"{pending_authors[-1]} {restored}"
            elif has_next:
                pending_authors.append(restored)
            continue

        # Standard bare author with initials (`Cheng M.`, `van Eck N.J.`, `Cho H.(.K.).`, `sik K.W.`)
        if _is_bare_initials_author(restored, allow_lower_start=bool(pending_authors)):
            pending_authors.append(restored)
            continue

        # Full-name semicolon co-author (`Devlin Jacob`, `Chang Ming-Wei`, `Le Scao Teven`, `CLAXTON GARY`)
        if has_next and _is_bare_fullname_author(restored):
            nxt_continues_ref = (
                bool(pending_authors)
                or "," in nxt
                or "(" in nxt
                or _is_bare_initials_author(nxt, allow_lower_start=True)
                or _is_bare_fullname_author(nxt)
                or _is_bare_initials_part(nxt)
                or bool(_ET_AL_OR_EDS_RE.match(nxt))
            )
            if nxt_continues_ref:
                pending_authors.append(restored)
                continue

        # Split surname before bare initial (`Barendregt; W.`) or mononym/particle inside active author list (`Zhang A.; Danish;`, `Araujo R.S.; da;`)
        if has_next and len(restored) <= 30 and not any(ch.isdigit() for ch in restored):
            if _SINGLE_NAME_WORD_RE.match(restored) and restored.lower() not in _NON_AUTHOR_LEAD_WORDS:
                if _is_bare_initials_part(nxt):
                    pending_authors.append(restored)
                    continue
                if pending_authors and (
                    "," in nxt
                    or "(" in nxt
                    or _is_bare_initials_author(nxt, allow_lower_start=True)
                    or _is_bare_fullname_author(nxt)
                    or _is_bare_initials_part(nxt)
                ):
                    pending_authors.append(restored)
                    continue
            if pending_authors and restored.lower() in _PARTICLES:
                pending_authors.append(restored)
                continue

        # Detect mid-title split like "... COVID-19 (2020); an empirical study, IJPE, (2022)"
        # Exclude single-letter fragments, bare page/bib-noise tokens, and merging onto numeric stubs
        if (
            not pending_authors
            and merged
            and restored[0].islower()
            and len(restored) >= 4
            and not _BARE_BIB_NOISE_RE.match(re.sub(r"[^a-z0-9]+", " ", restored.lower()).strip())
            and not _BARE_NUMERIC_STUB_RE.match(re.sub(r"[^a-z0-9]+", " ", merged[-1].lower()).strip())
            and not re.match(r"^(?:van|von|de|del|da|di|la|le)\s+", restored)
        ):
            merged[-1] = f"{merged[-1]}; {restored}"
            continue

        if pending_authors:
            merged.append("; ".join(pending_authors + [restored]))
            pending_authors = []
        else:
            merged.append(restored)

    if pending_authors:
        merged.append("; ".join(pending_authors))
    return merged


def normalize_reference_key(raw: str) -> str:
    """Canonicalize a reference string to DOI, OpenAlex ID, Lens ID, or normalized bibliographic key.

    Drops unidentifiable reference stubs (bare years such as `(2024)`, purely numeric/page tokens,
    bare initials, and `< 4`-character fragments) by returning `""` so they never form artificial
    coupling super-hubs.
    """
    s = html.unescape(unicodedata.normalize("NFKC", str(raw or ""))).strip()
    if not s or s.lower() in _MISSING_SENTINELS:
        return ""
    s_clean = re.sub(r"\s+", " ", s)

    # OpenAlex work ID
    m_oa = _OPENALEX_WORK_RE.match(s_clean)
    if m_oa:
        return f"https://openalex.org/{m_oa.group(1).upper()}"

    # Lens.org 15-digit ID
    if _LENS_ID_RE.match(s_clean):
        return f"lens:{s_clean}"

    # Clean repeated WoS "DOI DOI 10..." or "DOI [10.xxx, 10.yyy]"
    s_doi_scan = re.sub(r"\bDOI\s+\[?\s*(?:DOI\s+)*", "DOI ", s_clean, flags=re.IGNORECASE)
    m_doi = _DOI_EXTRACT_RE.search(s_doi_scan)
    if m_doi:
        d = _strip_unbalanced_trailing(m_doi.group(1)).lower()
        if d:
            return f"doi:{d}"

    compact = re.sub(r"[^a-z0-9]+", " ", s_clean.lower()).strip()
    if not compact:
        return ""
    if _BARE_YEAR_STUB_RE.match(compact):
        return ""
    if _BARE_NUMERIC_STUB_RE.match(compact):
        return ""
    if len(compact) < 4 or len(compact.replace(" ", "")) < 4:
        return ""
    if _BARE_INITIALS_COMPACT_RE.match(compact):
        return ""
    if _BARE_BIB_NOISE_RE.match(compact):
        return ""
    return compact[:100]


def _extract_normalized_refs(raw_chunks: list[str]) -> tuple[list[str], int]:
    """Normalize a list of raw reference strings, returning `(sorted_unique_keys, n_dropped_stubs)`."""
    keys: set[str] = set()
    n_dropped = 0
    for chunk in raw_chunks:
        s = str(chunk or "").strip()
        if not s or s.lower() in _MISSING_SENTINELS or not any(ch.isalnum() for ch in s):
            continue
        k = normalize_reference_key(s)
        if k:
            keys.add(k)
        else:
            n_dropped += 1
    return sorted(keys), n_dropped


def _stub_warning(n_dropped: int) -> str:
    return (
        f"UnidentifiableReferenceStubWarning: Dropped {n_dropped} unidentifiable reference stub(s) "
        "(e.g., bare years or single tokens) to prevent artificial coupling hubs."
    )


def normalize_natural_author_name(name: str) -> str:
    """Normalize a natural-order author name (`First [Middle] [Particle] Last [Suffix]`) to `last initials`."""
    s = html.unescape(unicodedata.normalize("NFKC", str(name or ""))).strip()
    if not s or s.lower() in _MISSING_SENTINELS:
        return ""
    # Strip embedded Scopus/ORCID parenthetical IDs e.g. "(60410297700)"
    s = re.sub(r"\s*\(\d{5,}\)\s*", " ", s).strip()
    if not s:
        return ""

    # If already "Surname, Given/Initials"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        # Drop trailing generational suffixes
        while len(parts) > 1 and parts[-1].lower().rstrip(".") in {x.rstrip(".") for x in _SUFFIXES}:
            parts.pop()
        surname = re.sub(r"\s+", " ", parts[0]).strip().lower()
        given = " ".join(parts[1:]) if len(parts) > 1 else ""
        initials = "".join(tok[0].lower() for tok in re.findall(r"[^\W\d_]+", given, flags=re.UNICODE))
        return f"{surname}, {initials}" if initials else surname

    tokens = [t.strip(",") for t in s.split() if t.strip(",")]
    while len(tokens) > 1 and tokens[-1].lower().rstrip(".") in {x.rstrip(".") for x in _SUFFIXES}:
        tokens.pop()
    if len(tokens) == 1:
        return tokens[0].lower()

    # Check if already "Surname Initials" (Unicode-aware, e.g., "van Eck N.J.", "Kowalski Ł.", "Richard M.-O.")
    last_tok = tokens[-1]
    last_letters = re.sub(r"[.\-\s]+", "", last_tok)
    if (
        1 <= len(last_letters) <= 4
        and all(ch.isalpha() and ch.isupper() for ch in last_letters)
        and (len(last_letters) == 1 or "." in last_tok or "-" in last_tok or last_tok.isupper())
    ):
        initials = last_letters.lower()
        surname = " ".join(tokens[:-1]).lower()
        return f"{surname}, {initials}"

    # Natural order: absorb nobiliary particles before the final surname token
    idx = len(tokens) - 1
    while idx > 1 and tokens[idx - 1].lower() in _PARTICLES:
        idx -= 1
    surname = " ".join(tokens[idx:]).lower()
    given_tokens = tokens[:idx]
    initials = "".join(t[0].lower() for t in given_tokens if t and t[0].isalpha())
    return f"{surname}, {initials}" if initials else surname


def _split_semicolon_or_comma_authors(raw: str) -> list[str]:
    """Parse author strings cleanly across semicolon-delimited or legacy comma-paired Scopus formats."""
    if not raw or str(raw).strip().lower() in _MISSING_SENTINELS:
        return []
    text = html.unescape(str(raw)).strip()
    if ";" in text:
        return [a.strip() for a in text.split(";") if a.strip() and a.strip().lower() not in _MISSING_SENTINELS]
    if "|" in text:
        return [a.strip() for a in text.split("|") if a.strip() and a.strip().lower() not in _MISSING_SENTINELS]
    if "," not in text:
        return [text]
    parts = [p.strip() for p in text.split(",") if p.strip()]
    # Merge generational suffixes back onto preceding token
    merged: list[str] = []
    for tok in parts:
        if merged and tok.lower().rstrip(".") in {x.rstrip(".") for x in _SUFFIXES}:
            merged[-1] = f"{merged[-1]}, {tok}"
        else:
            merged.append(tok)
    if len(merged) == 2:
        # Solo "Surname, Initials" vs "Alice Smith, Bob Jones"
        if _UNICODE_INITIALS_RE.match(merged[1]) or len(merged[0].split()) == 1:
            return [f"{merged[0]}, {merged[1]}"]
        return merged
    if len(merged) >= 4 and len(merged) % 2 == 0:
        if all(_UNICODE_INITIALS_RE.match(merged[i]) for i in range(1, len(merged), 2)):
            return [f"{merged[i]}, {merged[i + 1]}" for i in range(0, len(merged), 2)]
    return merged


def _split_keywords(raw: str) -> list[str]:
    """Split semicolon- or pipe-delimited keywords, lowercasing and deduplicating cleanly."""
    if not raw or str(raw).strip().lower() in _MISSING_SENTINELS:
        return []
    text = html.unescape(unicodedata.normalize("NFKC", str(raw))).strip()
    sep = "|" if ("|" in text and ";" not in text) else ";"
    out: list[str] = []
    seen: set[str] = set()
    for tok in text.split(sep):
        k = re.sub(r"\s+", " ", tok).strip().lower()
        if k and k not in _MISSING_SENTINELS and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _open_text_with_encoding(path: Path) -> tuple[str, str]:
    raw_bytes = path.read_bytes()
    if raw_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw_bytes.decode("utf-16"), "utf-16"
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise FormatDetectionError(f"Unable to decode {path} as UTF-8, UTF-16, Windows-1252, or Latin-1.")


def load_thesaurus(path: Path | str) -> dict[str, str | None]:
    """Load a VOSviewer-format 2-column TSV/CSV thesaurus (`label` -> `replace_by`; empty = ignore)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Thesaurus file not found: {p}")
    text, _ = _open_text_with_encoding(p)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    delim = "\t" if "\t" in lines[0] else (";" if ";" in lines[0] else ",")
    reader = csv.reader(lines, delimiter=delim)
    mapping: dict[str, str | None] = {}
    for row_idx, row in enumerate(reader):
        if not row:
            continue
        lbl = re.sub(r"\s+", " ", row[0]).strip().lower()
        rep = re.sub(r"\s+", " ", row[1]).strip().lower() if len(row) > 1 else ""
        if row_idx == 0 and lbl in {"label", "term", "keyword", "from"} and rep in {"replace by", "replace_by", "to", "replacement"}:
            continue
        if lbl:
            mapping[lbl] = rep if rep else None
    return mapping


def _clean_cell(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val or "")).strip()


def _fallback_dedup_key(source_prefix: str, native_id: str, title: str, year: int | None, authors: list[dict[str, str]], file_stem: str, idx: int) -> str:
    if native_id:
        return f"{source_prefix}:{native_id.lower()}"
    t_norm = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    if t_norm:
        first_au = authors[0]["id"] if authors else ""
        return f"tya:{t_norm[:80]}|{year or ''}|{first_au}"
    return f"pos:{file_stem}:{idx}"


def parse_scopus_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Scopus CSV (legacy & 2023-2026 exports with semicolon-author references & Author(s) ID)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty Scopus CSV: {path}")
    lines = text.splitlines()
    first = lines[0]
    delim = ";" if first.count(";") > first.count(",") and first.count(";") > first.count("\t") else ","
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delim)

    works: list[dict[str, Any]] = []
    warnings: list[str] = []
    any_disambig = False
    n_dropped_stubs = 0
    for idx, row in enumerate(reader):
        if not any(_clean_cell(v) for v in row.values()):
            continue
        refs_raw = _clean_cell(row.get("References") or row.get("references"))
        refs, dropped = _extract_normalized_refs(split_scopus_references(refs_raw))
        n_dropped_stubs += dropped

        # Authors: prefer canonical Scopus Author(s) ID when aligned, else extract from Author full names / Authors
        auth_names_raw = _clean_cell(row.get("Author full names") or row.get("Authors") or row.get("authors"))
        auth_ids_raw = _clean_cell(row.get("Author(s) ID") or row.get("author_ids"))
        parsed_names = _split_semicolon_or_comma_authors(auth_names_raw)
        parsed_ids = [t.strip() for t in str(auth_ids_raw).split(";") if t.strip() and t.strip().lower() not in _MISSING_SENTINELS]

        authors: list[dict[str, str]] = []
        if parsed_ids and len(parsed_ids) == len(parsed_names):
            any_disambig = True
            for aid, nm in zip(parsed_ids, parsed_names):
                clean_nm = re.sub(r"\s*\(\d{5,}\)\s*", "", nm).strip()
                authors.append({"id": f"scopus_auid:{aid}", "name": clean_nm or aid})
        elif parsed_names:
            for nm in parsed_names:
                m_id = re.search(r"\((\d{5,})\)", nm)
                clean_nm = re.sub(r"\s*\(\d{5,}\)\s*", "", nm).strip()
                if m_id:
                    any_disambig = True
                    authors.append({"id": f"scopus_auid:{m_id.group(1)}", "name": clean_nm or m_id.group(1)})
                else:
                    norm = normalize_natural_author_name(clean_nm)
                    if norm:
                        authors.append({"id": norm, "name": clean_nm})
        elif parsed_ids:
            any_disambig = True
            for aid in parsed_ids:
                authors.append({"id": f"scopus_auid:{aid}", "name": aid})

        akw = _split_keywords(_clean_cell(row.get("Author Keywords") or row.get("author_keywords")))
        ikw = _split_keywords(_clean_cell(row.get("Index Keywords") or row.get("Indexed Keywords") or row.get("index_keywords")))
        doi = canonicalize_doi(row.get("DOI") or row.get("doi") or "")
        native_eid = _clean_cell(row.get("EID") or row.get("eid"))
        title = _clean_cell(row.get("Title") or row.get("title"))
        year = _parse_int(row.get("Year") or row.get("year"), default=0) or None
        rec_id = doi or native_eid or f"scopus_{path.stem}_{idx}"
        works.append({
            "id": rec_id,
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("eid", native_eid, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(row.get("Cited by") or row.get("cited_by_count"), default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": any_disambig,
            "source_format": "scopus",
        })
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if works and not any_disambig:
        warnings.append("HomonymCollapseWarning: Scopus export lacks `Author(s) ID`; co-authorship falls back to normalized author strings.")
    return works, warnings


def parse_wos_tsv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Web of Science Tab-Delimited export (`TI`, `PY`, `DI`, `TC`, `CR`, `AU`/`AF`, `DE`, `ID`)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty Web of Science TSV: {path}")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    works: list[dict[str, Any]] = []
    n_dropped_stubs = 0
    for idx, row in enumerate(reader):
        if not any(_clean_cell(v) for v in row.values()):
            continue
        cr_raw = _clean_cell(row.get("CR") or row.get("cr"))
        prot, stashed = _protect_html_entities_and_sici(cr_raw)
        refs, dropped = _extract_normalized_refs([_restore_stashed(tok, stashed) for tok in prot.split(";")])
        n_dropped_stubs += dropped
        raw_au = _clean_cell(row.get("AF") or row.get("AU") or row.get("af") or row.get("au"))
        authors = []
        for nm in _split_semicolon_or_comma_authors(raw_au):
            norm = normalize_natural_author_name(nm)
            if norm:
                authors.append({"id": norm, "name": nm})
        akw = _split_keywords(_clean_cell(row.get("DE") or row.get("de")))
        ikw = _split_keywords(_clean_cell(row.get("ID") or row.get("id")))
        doi = canonicalize_doi(row.get("DI") or row.get("di") or "")
        native_ut = _clean_cell(row.get("UT") or row.get("ut"))
        title = _clean_cell(row.get("TI") or row.get("ti"))
        year = _parse_int(row.get("PY") or row.get("py"), default=0) or None
        works.append({
            "id": doi or native_ut or f"wos_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("ut", native_ut, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(row.get("TC") or row.get("tc"), default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": False,
            "source_format": "wos",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if works:
        warnings.append("HomonymCollapseWarning: Web of Science tab-delimited exports lack numeric author IDs; co-authorship uses normalized author names.")
    return works, warnings


def parse_wos_plaintext(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Web of Science Plain-Text (`FN ... VR 1.0`) multi-line tag-value exports (3-space continuations)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty Web of Science plain-text file: {path}")
    records: list[dict[str, list[str]]] = []
    cur: dict[str, list[str]] = {}
    cur_tag = ""

    for raw_line in text.splitlines():
        if raw_line.startswith("EF"):
            break
        if raw_line.startswith("ER"):
            if cur:
                records.append(cur)
                cur = {}
            cur_tag = ""
            continue
        if len(raw_line) >= 3 and raw_line[:2].strip() and raw_line[2] == " ":
            tag = raw_line[:2].strip()
            val = raw_line[3:].strip()
            if tag in {"FN", "VR"}:
                cur_tag = ""
                continue
            cur_tag = tag
            cur.setdefault(tag, []).append(val)
        elif raw_line.startswith("   ") and cur_tag:
            cur.setdefault(cur_tag, []).append(raw_line[3:].strip())

    if cur:
        records.append(cur)

    works: list[dict[str, Any]] = []
    n_dropped_stubs = 0
    for idx, rec in enumerate(records):
        refs, dropped = _extract_normalized_refs(rec.get("CR", []))
        n_dropped_stubs += dropped
        raw_authors = rec.get("AF") or rec.get("AU") or []
        authors = []
        for nm in raw_authors:
            norm = normalize_natural_author_name(nm)
            if norm:
                authors.append({"id": norm, "name": nm})
        # Join wrapped DE/ID keyword continuation lines with space (keywords are already semicolon-separated)
        akw = _split_keywords(" ".join(rec.get("DE", [])))
        ikw = _split_keywords(" ".join(rec.get("ID", [])))
        doi = canonicalize_doi(" ".join(rec.get("DI", [])))
        native_ut = " ".join(rec.get("UT", [])).strip()
        title = " ".join(rec.get("TI", [])).strip()
        year = _parse_int(" ".join(rec.get("PY", [])), default=0) or None
        works.append({
            "id": doi or native_ut or f"wos_pt_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("ut", native_ut, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(" ".join(rec.get("TC", [])), default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": False,
            "source_format": "wos_plaintext",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if works:
        warnings.append("HomonymCollapseWarning: Web of Science plain-text exports lack numeric author IDs; co-authorship uses normalized author names.")
    return works, warnings


def parse_dimensions_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Dimensions CSV (auto-skipping 1-line preamble; right-anchored pipe-delimited `Cited references`)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty Dimensions CSV: {path}")
    lines = text.splitlines(keepends=True)

    # Auto-detect 1-line or 2-line preamble (`About the data...`)
    header_idx = 0
    dim_markers = {"publication id", "pub_year", "pub year", "times cited", "times_cited", "cited references", "cited_references"}
    for probe in range(min(3, len(lines))):
        cols = {c.strip().strip('"').lower() for c in next(csv.reader([lines[probe]]))}
        if cols & dim_markers:
            header_idx = probe
            break

    reader = csv.DictReader(io.StringIO("".join(lines[header_idx:]), newline=""))
    works: list[dict[str, Any]] = []
    n_dropped_stubs = 0
    for idx, row in enumerate(reader):
        if not any(_clean_cell(v) for v in row.values()):
            continue
        low_row = {str(k).strip().lower(): _clean_cell(v) for k, v in row.items() if k is not None}
        cr_raw = low_row.get("cited references") or low_row.get("cited_references") or ""
        refs_set: set[str] = set()
        if cr_raw.strip():
            # Split bracketed pipe-tuples `[Author|Title|...|DOI];[...]` safely even if `|` occurs inside title
            blocks = re.split(r"\]\s*;\s*\[", cr_raw.strip().lstrip("[").rstrip("]"))
            for blk in blocks:
                if not blk.strip():
                    continue
                fields = [f.strip() for f in blk.split("|")]
                # Anchor DOI from right (`fields[-1]`) to resist stray pipes in titles
                doi_cand = canonicalize_doi(fields[-1]) if fields else ""
                if doi_cand:
                    refs_set.add(f"doi:{doi_cand}")
                else:
                    k = normalize_reference_key(blk)
                    if k:
                        refs_set.add(k)
                    elif any(ch.isalnum() for ch in blk):
                        n_dropped_stubs += 1

        raw_au = low_row.get("authors") or ""
        authors = []
        for nm in _split_semicolon_or_comma_authors(raw_au):
            norm = normalize_natural_author_name(nm)
            if norm:
                authors.append({"id": norm, "name": nm})

        akw = _split_keywords(low_row.get("concepts") or low_row.get("keywords") or "")
        ikw = _split_keywords(low_row.get("mesh terms") or low_row.get("for (anzsrc 2020)") or "")
        doi = canonicalize_doi(low_row.get("doi") or "")
        native_id = low_row.get("publication id") or low_row.get("id") or ""
        title = low_row.get("title") or ""
        year = _parse_int(low_row.get("pub year") or low_row.get("pub_year"), default=0) or None
        works.append({
            "id": doi or native_id or f"dim_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("dim", native_id, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(low_row.get("times cited") or low_row.get("times_cited"), default=0),
            "referenced_works": sorted(refs_set),
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": False,
            "source_format": "dimensions",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if works:
        warnings.append("HomonymCollapseWarning: Dimensions CSV exports lack disambiguated author IDs; co-authorship uses normalized author strings.")
    return works, warnings


def parse_lens_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Lens.org CSV (`Lens ID`, natural-order `Author/s`, 15-digit Lens-ID `References`)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty Lens.org CSV: {path}")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    works: list[dict[str, Any]] = []
    n_dropped_stubs = 0
    for idx, row in enumerate(reader):
        if not any(_clean_cell(v) for v in row.values()):
            continue
        refs_raw = _clean_cell(row.get("References"))
        refs, dropped = _extract_normalized_refs(refs_raw.split(";"))
        n_dropped_stubs += dropped
        raw_au = _clean_cell(row.get("Author/s") or row.get("Authors"))
        authors = []
        for nm in _split_semicolon_or_comma_authors(raw_au):
            norm = normalize_natural_author_name(nm)
            if norm:
                authors.append({"id": norm, "name": nm})
        akw = _split_keywords(_clean_cell(row.get("Keywords")))
        ikw = _split_keywords(_clean_cell(row.get("Fields of Study") or row.get("MeSH Terms")))
        doi = canonicalize_doi(row.get("DOI") or "")
        native_id = _clean_cell(row.get("Lens ID"))
        title = _clean_cell(row.get("Title"))
        year = _parse_int(row.get("Publication Year") or row.get("Year"), default=0) or None
        cites = row.get("Citing Works Count") or row.get("Scholarly Citation Count") or 0
        works.append({
            "id": doi or native_id or f"lens_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("lens", native_id, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(cites, default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": False,
            "source_format": "lens",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if works:
        warnings.append("HomonymCollapseWarning: Lens.org CSV exports lack disambiguated author IDs; co-authorship normalizes natural-order names (particle/suffix aware).")
    return works, warnings


def parse_openalex_json(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse OpenAlex JSON (list, `{"results": [...]}`, or `{"works": [...]}`) with BOM/UTF-16 & truncation guards."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty OpenAlex JSON: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatDetectionError(f"Malformed OpenAlex JSON in {path.name}: {exc}") from exc
    if isinstance(payload, dict):
        raw_list = (payload.get("works") if "works" in payload else payload.get("results")) or []
    elif isinstance(payload, list):
        raw_list = payload
    else:
        raise FormatDetectionError(f"Unsupported OpenAlex JSON structure in {path}")
    if not isinstance(raw_list, list):
        raise FormatDetectionError(f"Expected list of works in OpenAlex JSON {path.name}")

    works: list[dict[str, Any]] = []
    n_truncated = 0
    n_dropped_stubs = 0
    for idx, w in enumerate(raw_list):
        if not isinstance(w, dict):
            continue
        refs, dropped = _extract_normalized_refs([str(r) for r in (w.get("referenced_works") or [])])
        n_dropped_stubs += dropped
        authorships = w.get("authorships") or []
        if len(authorships) >= 100 or bool(w.get("is_authors_truncated")):
            n_truncated += 1
        authors: list[dict[str, str]] = []
        disambig = False
        for au in authorships:
            if not isinstance(au, dict):
                continue
            aobj = au.get("author")
            if not isinstance(aobj, dict):
                aobj = {}
            aid = str(aobj.get("id") or "").strip()
            nm = str(aobj.get("display_name") or au.get("raw_author_name") or "").strip()
            if aid:
                disambig = True
                m_a = _OPENALEX_AUTH_RE.match(aid)
                canon_id = f"openalex:{m_a.group(1).upper()}" if m_a else aid.lower()
                authors.append({"id": canon_id, "name": nm or canon_id})
            elif nm:
                norm = normalize_natural_author_name(nm)
                if norm:
                    authors.append({"id": norm, "name": nm})

        akw = sorted({
            str(k.get("display_name") or k.get("keyword") or "").strip().lower()
            for k in (w.get("keywords") or [])
            if isinstance(k, dict) and str(k.get("display_name") or k.get("keyword") or "").strip()
        })
        ikw = sorted({
            str(t.get("display_name") or "").strip().lower()
            for t in (w.get("topics") or [])
            if isinstance(t, dict) and str(t.get("display_name") or "").strip()
        })
        doi = canonicalize_doi(w.get("doi") or "")
        native_id = _clean_cell(w.get("id"))
        title = _clean_cell(w.get("title") or w.get("display_name"))
        year = _parse_int(w.get("publication_year"), default=0) or None
        works.append({
            "id": native_id or doi or f"oa_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("oa", native_id, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(w.get("cited_by_count"), default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": disambig,
            "authors_truncated": len(authorships) >= 100 or bool(w.get("is_authors_truncated")),
            "source_format": "openalex",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if n_truncated > 0:
        warnings.append(
            f"OpenAlexTruncationWarning: {n_truncated} work(s) hit OpenAlex's 100-author API cap (`len(authorships) >= 100` or `is_authors_truncated`)."
        )
    return works, warnings


def parse_openalex_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse OpenAlex web CSV export (`|`-delimited multi-values with positional `zip_longest` alignment)."""
    text, _ = _open_text_with_encoding(path)
    if not text.strip():
        raise FormatDetectionError(f"Empty OpenAlex CSV: {path}")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    works: list[dict[str, Any]] = []
    n_truncated = 0
    n_dropped_stubs = 0
    for idx, row in enumerate(reader):
        if not any(_clean_cell(v) for v in row.values()):
            continue
        refs_raw = _clean_cell(row.get("referenced_works"))
        refs, dropped = _extract_normalized_refs(re.split(r"[|;]", refs_raw))
        n_dropped_stubs += dropped

        # Preserve empty pipe slots so `author.id` and `display_name` never drift out of alignment
        raw_ids = (row.get("authorships.author.id") or "").split("|") if row.get("authorships.author.id") else []
        raw_nms = (row.get("authorships.author.display_name") or "").split("|") if row.get("authorships.author.display_name") else []
        authors: list[dict[str, str]] = []
        disambig = False
        for aid_raw, nm_raw in itertools.zip_longest(raw_ids, raw_nms, fillvalue=""):
            aid = aid_raw.strip()
            nm = nm_raw.strip()
            if aid:
                disambig = True
                m_a = _OPENALEX_AUTH_RE.match(aid)
                canon = f"openalex:{m_a.group(1).upper()}" if m_a else aid.lower()
                authors.append({"id": canon, "name": nm or canon})
            elif nm:
                norm = normalize_natural_author_name(nm)
                if norm:
                    authors.append({"id": norm, "name": nm})
        if len(authors) >= 100:
            n_truncated += 1

        akw = _split_keywords(_clean_cell(row.get("keywords.display_name")))
        ikw = _split_keywords(_clean_cell(row.get("topics.display_name") or row.get("primary_topic.display_name")))
        doi = canonicalize_doi(row.get("doi") or "")
        native_id = _clean_cell(row.get("id"))
        title = _clean_cell(row.get("display_name") or row.get("title"))
        year = _parse_int(row.get("publication_year"), default=0) or None
        works.append({
            "id": native_id or doi or f"oa_csv_{path.stem}_{idx}",
            "dedup_key": f"doi:{doi}" if doi else _fallback_dedup_key("oa", native_id, title, year, authors, path.stem, idx),
            "doi": doi,
            "title": title,
            "publication_year": year,
            "cited_by_count": _parse_int(row.get("cited_by_count"), default=0),
            "referenced_works": refs,
            "authors": authors,
            "author_keywords": akw,
            "index_keywords": ikw,
            "authors_disambiguated": disambig,
            "authors_truncated": len(authors) >= 100,
            "source_format": "openalex_csv",
        })
    warnings: list[str] = []
    if n_dropped_stubs > 0:
        warnings.append(_stub_warning(n_dropped_stubs))
    if n_truncated > 0:
        warnings.append(f"OpenAlexTruncationWarning: {n_truncated} row(s) contain >= 100 authors.")
    return works, warnings


def detect_format(path: Path) -> str:
    """Auto-detect export format cleanly from file content and header signatures."""
    ext = path.suffix.lower()
    if ext == ".json":
        return "openalex"
    text, _ = _open_text_with_encoding(path)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FormatDetectionError(f"Input file is empty: {path}")

    # Web of Science Plain Text (`FN Clarivate...` / `VR 1.0` / `PT J`)
    if lines[0].startswith("FN ") or (len(lines) > 1 and lines[1].startswith("VR ")) or lines[0].startswith("PT "):
        return "wos_plaintext"

    # Inspect up to first 3 lines for header columns (handles Dimensions 1-line preamble)
    for probe in lines[:3]:
        if "\t" in probe:
            tabs = {c.strip().strip('"') for c in probe.split("\t")}
            if {"TI", "PY", "CR"} & tabs or "UT" in tabs:
                return "wos"
        delim = ";" if probe.count(";") > probe.count(",") else ","
        cols_low = {c.strip().strip('"').lower() for c in next(csv.reader([probe], delimiter=delim))}
        if "lens id" in cols_low or "citing works count" in cols_low:
            return "lens"
        if {"publication id", "times cited", "times_cited", "cited references", "cited_references"} & cols_low:
            return "dimensions"
        if "display_name" in cols_low and "cited_by_count" in cols_low:
            return "openalex_csv"
        if {"eid", "source title", "author(s) id", "author keywords", "references"} & cols_low:
            return "scopus"

    raise FormatDetectionError(
        f"Could not auto-detect bibliometric format for '{path.name}'. Pass --format explicitly ({', '.join(SUPPORTED_FORMATS[1:])})."
    )


def _record_richness(rec: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    """Score record metadata completeness so duplicate DOIs retain the richest export row."""
    return (
        len(rec.get("referenced_works") or ()),
        len(rec.get("authors") or ()),
        len(rec.get("author_keywords") or ()) + len(rec.get("index_keywords") or ()),
        int(rec.get("cited_by_count") or 0),
        int(rec.get("publication_year") is not None),
        len(str(rec.get("title") or "")),
    )


def load_corpus_files(
    paths: list[Path | str] | Path | str,
    fmt: str = "auto",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load one or more bibliometric exports, deduplicating records across multi-file batches (retaining the richest record)."""
    path_list = [Path(paths)] if isinstance(paths, (Path, str)) else [Path(p) for p in paths]
    if not path_list:
        raise ValueError("At least one input file path is required.")

    all_works: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    seen_idx: dict[str, int] = {}
    n_dups = 0

    for p in path_list:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        resolved = detect_format(p) if fmt == "auto" else fmt.lower()
        if resolved == "scopus":
            wks, warns = parse_scopus_csv(p)
        elif resolved == "wos":
            wks, warns = parse_wos_tsv(p)
        elif resolved == "wos_plaintext":
            wks, warns = parse_wos_plaintext(p)
        elif resolved == "openalex":
            wks, warns = parse_openalex_json(p)
        elif resolved == "openalex_csv":
            wks, warns = parse_openalex_csv(p)
        elif resolved == "dimensions":
            wks, warns = parse_dimensions_csv(p)
        elif resolved == "lens":
            wks, warns = parse_lens_csv(p)
        else:
            raise FormatDetectionError(f"Unsupported format '{fmt}'. Choose from: {', '.join(SUPPORTED_FORMATS)}.")

        for w in warns:
            if w not in all_warnings:
                all_warnings.append(w)
        for rec in wks:
            dk = rec.get("dedup_key") or rec["id"]
            if dk in seen_idx:
                n_dups += 1
                prev_i = seen_idx[dk]
                if _record_richness(rec) > _record_richness(all_works[prev_i]):
                    all_works[prev_i] = rec
                continue
            seen_idx[dk] = len(all_works)
            all_works.append(rec)

    if n_dups > 0:
        all_warnings.append(f"Deduplicated {n_dups} duplicate record(s) across {len(path_list)} input file(s).")
    return all_works, all_warnings


def load_corpus(path: Path | str, fmt: str = "auto") -> list[dict[str, Any]]:
    """Backward-compatible single-file loader returning standardized work dicts."""
    works, _ = load_corpus_files([path], fmt=fmt)
    return works
