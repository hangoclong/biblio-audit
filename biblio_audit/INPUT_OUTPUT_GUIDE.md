# `biblio-audit` Input & Output Reference Guide

This guide documents how to export data cleanly from all five supported bibliometric databases (seven export formats) and how to interpret every output artifact produced by `biblio-audit`.

---

## 1. Supported Input Formats (`--format auto`)

`biblio-audit` auto-detects file formats from content headers and supports passing multiple files simultaneously (e.g., Web of Science 500-record batches), automatically deduplicating records by canonical DOI, EID, UT, Lens ID, or OpenAlex ID.

| Database | Export Steps | Required Fields by Network | Parser Protections Built In |
|---|---|---|---|
| **Scopus CSV** (`scopus`) | Search $\to$ **Export** $\to$ **CSV** $\to$ check *Citation information*, *Bibliographical information*, *Abstract & keywords*, and *Include references*. | `BC`/`CC`: `References`<br>`CA`: `Author(s) ID` / `Authors`<br>`KW`: `Author Keywords`, `Index Keywords` | Boundary-aware splitting of 2023–2026 semicolon-author `References`; protects HTML entities and SICI DOIs (`<`, `>`, `;`); prefers canonical `Author(s) ID` over raw names. |
| **Web of Science TSV** (`wos`) | Search $\to$ **Export** $\to$ **Tab delimited file** $\to$ Record Content: **Full Record and Cited References**. | `BC`/`CC`: `CR`<br>`CA`: `AF` / `AU`<br>`KW`: `DE`, `ID` | Cleans doubled `DOI DOI` and bracketed multi-DOI strings; supports multi-file batch deduplication via `UT`/`DI`. |
| **Web of Science Plain Text** (`wos_plaintext`) | Search $\to$ **Export** $\to$ **Plain text file** (`FN Clarivate...` / `VR 1.0`) $\to$ **Full Record and Cited References**. | `BC`/`CC`: `CR`<br>`CA`: `AF` / `AU`<br>`KW`: `DE`, `ID` | Handles 3-space continuation lines accurately (`AU`/`AF`/`CR` joined by item; `TI`/`DE`/`ID` joined smoothly). |
| **OpenAlex JSON** (`openalex`) | OpenAlex API `/works` JSON (`{"results": [...]}` or raw list) or saved JSON corpus. | `BC`/`CC`: `referenced_works`<br>`CA`: `authorships`<br>`KW`: `keywords` / `topics` | Prefers disambiguated OpenAlex Author IDs (`A...`); emits `OpenAlexTruncationWarning` if any record hits the 100-author API truncation cap. |
| **OpenAlex CSV** (`openalex_csv`) | OpenAlex web portal CSV export (`|`-delimited multi-values). | Note: default web CSV omits `referenced_works`; `CA`/`KW` work directly. | Uses positional `zip_longest` so empty `||` slots never misalign author IDs vs. display names; raises `MissingFieldError` cleanly if `BC`/`CC` lacks references. |
| **Dimensions CSV** (`dimensions`) | Dimensions UI $\to$ **Export** $\to$ **Export for bibliometric mapping** (or full CSV). | `BC`/`CC`: `Cited references`<br>`CA`: `Authors`<br>`KW`: `Concepts` / `MeSH terms` | Auto-skips the 1-line `"About the data..."` preamble; parses bracketed pipe tuples with right-anchored DOI extraction (`fields[-1]`) resistant to `|` inside titles. |
| **Lens.org CSV** (`lens`) | Lens.org Scholarly Works $\to$ **Export** $\to$ **CSV**. | `BC`/`CC`: `References` (15-digit Lens IDs)<br>`CA`: `Author/s`<br>`KW`: `Keywords` / `Fields of Study` | Preserves 15-digit Lens reference IDs; normalizes natural-order author names while preserving nobiliary particles (`van`, `de la`, `von`) and stripping suffixes (`Jr.`, `III`). |

### Optional VOSviewer Thesaurus (`--thesaurus thesaurus.txt`)
Pass any standard VOSviewer 2-column tab-delimited thesaurus file:
```tsv
label	replace by
ai	artificial intelligence
smes	small and medium enterprises
literature review	
```
Rows with an empty `replace by` cell instruct `biblio-audit` to exclude that term/node identically to VOSviewer.

---

## 2. Output Artifacts & Column Schemas

### A. Ready-to-Paste Supplementary Disclosure Table (`--md-out`, `--latex-out`)
Prints directly to stdout and optionally saves to Markdown (`.md`) or Booktabs LaTeX (`.tex`). Includes:
- Corpus metadata, active normalization, denominator scope, and LCC coverage (`n_lcc / n_core_selected`).
- Matched cluster counts ($K_{\mathrm{full}} \to K_{\mathrm{frac}}$) and exact on-grid midpoint resolution $\gamma^*$.
- Within-rule AMI baselines, cross-rule AMI/ARI, **Excess Counting Instability ($\Delta_{\mathrm{count}}$)**, and reverse matching check.
- **Chance-Adjusted Element-Centric Similarity (`AECS` / $\Delta_{\mathrm{AECS}}$)** and metric **Variation of Information (`VI` / `NVI`)**.
- **Core-Node Reassignment (`1 - Hungarian`)**, symmetrized **Top-10 Landmark Retention**, and **BC TF-IDF Label Jaccard**.
- Optional **Bipartite Null Benchmark** (fixed-$N$ RBConfiguration $p$-value and VOS quality $p$-value) and **Editorial Triage Badge** with guardrail warnings. For offline citation corpora lacking external cited-reference publication years, the causal temporal MCMC null uses the conservative upper-bound proxy $t_{\mathrm{cited}} \le \min(t_{\mathrm{citing}})$ within the core graph (emitting `LowMixingWarning` on narrow time spans if swap acceptance falls below 5%; corpora with incomplete citing years automatically fall back to unbiased Curveball randomization).

### B. Per-Node Boundary-Shifter Table (`--nodes-csv`)
Exports one row per shared-LCC node, sorted from most unstable (`delta_ecs_node` highest) to most stable:

| Column | Type | Meaning |
|---|---|---|
| `node_id` | string | Canonical DOI, EID, author ID, reference key, or keyword |
| `label` | string | Human-readable title, author name, reference, or keyword |
| `cited_by_count` | int | Citations (or summed citations for authors/keywords) |
| `occurrences` | int | Document count or occurrence count |
| `full_cluster_medoid` | int | Consensus Medoid cluster ID under Full Counting (`0`-indexed) |
| `frac_cluster_aligned` | int | Consensus Medoid Fractional cluster ID optimally aligned to Full IDs via Hungarian matching |
| `switched_cluster` | bool | `True` if Medoid full cluster $\neq$ aligned fractional cluster |
| `ecs_within_full` | float | Mean per-node ECS stability across all full-counting seeds (`[0, 1]`) |
| `ecs_within_frac` | float | Mean per-node ECS stability across all matched fractional seeds (`[0, 1]`) |
| `ecs_cross` | float | Mean per-node ECS similarity across all $S^2$ full-vs.-fractional pairs (`[0, 1]`) |
| `delta_ecs_node` | float | $\min(\text{ecs\_within\_full}, \text{ecs\_within\_frac}) - \text{ecs\_cross}$; isolates counting-induced shifting from stochastic solver noise |
| `is_boundary_shifter` | bool | `True` if `switched_cluster` is `True` or `delta_ecs_node >= 0.20` |

### C. VOSviewer Verification Export (`--vos-export-dir`)
Writes three tab-delimited files loadable directly into VOSviewer (`Create -> Create a map based on network data -> VOSviewer`):
- `<network>_vos_map.txt`: Contains `id`, `label`, `cluster` (Medoid full cluster, 1-indexed), `score<frac_cluster>`, `score<ecs_stability>`, `score<delta_ecs_node>`, `weight<Occurrences>`, and `weight<Citations>`. Open **Overlay Visualization** on `score<delta_ecs_node>` inside VOSviewer to color every boundary shifter directly on your map!
- `<network>_vos_network_full.txt` and `<network>_vos_network_frac.txt`: Exact 1-indexed weighted edge lists (`u  v  weight`) for direct visual inspection inside VOSviewer.

### D. Machine-Readable Provenance Ledger (`--json-out`)
Stores software version, UTC timestamp, SHA-256 hashes of every input file, full CLI parameters, parser/guardrail warnings, and every scalar metric under `audits[network_type]` (plus top-level `audit` shortcut for single-network runs).
