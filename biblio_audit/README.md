# `biblio-audit`

**VOSviewer-Parity Full-vs.-Fractional Counting Sensitivity & Clustering Stability Auditor for Bibliometric Science Maps**

`biblio-audit` is a lightweight, deterministic, offline Python package and command-line utility that verifies whether bibliometric clusters produced under default **full counting** survive switching to **fractional counting** (Perianes-Rodriguez, Waltman, & van Eck, 2016) at matched cluster granularity under exact VOSviewer 1.6.20/1.6.21 normalization and clustering parity.

## Why `biblio-audit`?

Existing bibliometric tools either construct maps under one counting rule at a time (`VOSviewer`, `CiteSpace`, `SciMAT`), implement full/binary counting only (`bibliometrix`, `pyBibX`, `litstudy`, `metaknowledge`), or output raw edge tables without partition-stability diagnostics (`biblionetwork`). `biblio-audit` closes this verification gap in one command:

- **7 Bibliographic Export Formats Across 5 Databases (Zero API keys, multi-file auto-dedup):** Scopus CSV (both legacy and 2023–2026 semicolon-author `References` exports), Web of Science Tab-Delimited (`.txt`/`.tsv`), Web of Science Plain-Text (`FN ... VR 1.0`), OpenAlex JSON & CSV, Dimensions CSV, and Lens.org CSV.
- **All 4 VOSviewer Bibliometric Network Families:** Bibliographic Coupling (`bc`), Co-Citation (`cc`), Co-Authorship (`ca`), and Keyword Co-Occurrence (`kw`) — run individually or together via `--network all`.
- **All 3 VOSviewer 1.6.20/1.6.21 Normalizations:**
  - `VOS_AS_CPM`: Association Strength $s_{ij} = \frac{2m\, w_{ij}}{k_i k_j}$ + Leiden `CPMVertexPartition` (default)
  - `VOS_Frac_CPM`: VOSviewer Fractionalization $s_{ij} = \frac{n}{2}\left(\frac{w_{ij}}{k_i} + \frac{w_{ij}}{k_j}\right)$ + Leiden `CPMVertexPartition`
  - `Leiden_RBConfig`: LinLog / Modularity (`RBConfigurationVertexPartition` on raw weights)
- **Mathematically Rigorous Partition-Comparison Suite:**
  - Matched-$K$ on-grid resolution plateau sweep isolating **excess counting-rule instability** ($\Delta_{\mathrm{count}}$ via Adjusted Mutual Information; Vinh et al., 2010) plus reverse direction check;
  - Exact $O(N)$ **Element-Centric Similarity (`ECS`)** (Gates et al., 2019) + exact closed-form hypergeometric **Chance-Adjusted ECS (`AECS` / $\Delta_{\mathrm{AECS}}$)** eliminating small-$K$ baseline inflation + per-node boundary-shifter detection (`nodes.csv` using consensus **Medoid** partitions);
  - Metric **Variation of Information (`VI` / `NVI`)** (Meilă, 2007), Cross-Rule **ARI**, optimal **Hungarian Core-Node Reassignment**, symmetrized **Top-10 Landmark Retention**, and **BC TF-IDF Label Jaccard**;
  - Optional exchangeable **bipartite null models** (`--null-rewires 99`): causal temporal MCMC swap ($t_{\mathrm{cited}} \le t_{\mathrm{citing}}$; offline exports proxy $t_{\mathrm{cited}}$ by the conservative upper bound $\min t_{\mathrm{citing}}$ within the core graph, falling back automatically to Curveball when years are incomplete) for citation networks and minimum-dimension **Curveball** (Strona et al., 2014) for `ca`/`kw`.
- **Publication-Ready Outputs:** Ready-to-paste Markdown (`--md-out`), Booktabs LaTeX (`--latex-out`), full SHA-256 provenance JSON (`--json-out`), per-node shifter CSV (`--nodes-csv`), and VOSviewer `map.txt` / `network.txt` exports (`--vos-export-dir`).

---

## Installation

Requires Python `>= 3.10` and four core scientific libraries (`numpy`, `scipy`, `igraph`, `scikit-learn`) — using `igraph`'s native C Leiden implementation (`beta = 0.01`, Traag et al., 2019) with zero C++ extension build hurdles.

```bash
pip install -e .
# Verify installation and inspect all CLI options:
biblio-audit --help
# Or invoke as a Python module if the scripts directory is not on PATH:
python -m biblio_audit --help
```

---

## Plug-and-Play Quickstart

### 0. 5-Second Bundled Demo (No External Data Needed)
```bash
biblio-audit --input examples/demo_scopus.csv --network all \
  --thesaurus examples/demo_thesaurus.txt --seeds 6 --starts 4 --min-cluster-size 3 \
  --md-out demo_audit.md --nodes-csv demo_nodes/ --vos-export-dir demo_vos/
```

### 1. Standard Bibliographic Coupling Audit (Scopus CSV)
```bash
biblio-audit --input scopus.csv --network bc --top-n 150 \
  --md-out sensitivity_bc.md --latex-out sensitivity_bc.tex \
  --nodes-csv bc_shifters.csv --json-out audit_bc.json
```

### 2. Audit All Four Networks Across Multi-File Web of Science Exports
```bash
biblio-audit --input savedrecs_1.txt savedrecs_2.txt --network all \
  --thesaurus vos_thesaurus.txt --md-out audit_all.md --json-out audit_all.json
```

### 3. Keyword Co-Occurrence with VOSviewer GUI Defaults + VOSviewer Export
```bash
biblio-audit --input scopus.csv --network kw --vos-gui-defaults \
  --thesaurus vos_thesaurus.txt --vos-export-dir vos_verify/
```

### 4. Citation Audit with 99 Exchangeable Causal Temporal Null Rewires
```bash
biblio-audit --input openalex.json --network cc --null-rewires 99 --json-out audit_cc.json
```

---

## Mathematical Summary

1. **Action-Normalized Fractional Counting (`--frac-scheme vosviewer`, default; Perianes-Rodriguez et al., 2016 Eq. 1–15):**
   $$W^{\mathrm{full}}_{ij} = \sum_{e:\, d_e \ge 2} c_{ie} c_{je}, \qquad W^{\mathrm{frac}}_{ij} = \sum_{e:\, d_e \ge 2} \frac{c_{ie} c_{je}}{d_e - 1} \quad (i \neq j,\; W_{ii} = 0).$$
   For `co_authorship` and `keyword_cooccurrence`, $d_e$ is strictly computed on each full document *before* `min_occurrences` / `top_n` node filtering (`--denom-scope auto` enforces `full-corpus` for `ca`/`kw` and `selected-core` for `bc`/`cc`, configurable via `--denom-scope`). Note that VOSviewer's desktop binary is closed-source; while `ca`/`bc`/`cc` follow the explicit equations of Perianes-Rodriguez et al. (2016) and the VOSviewer 1.6.20 Manual (pp. 34–35), an opt-in document-normalized comparator `--frac-scheme refined` ($2/(d_e(d_e - 1))$; Leydesdorff & Park, 2017) is also included for sensitivity checks.

2. **Excess Counting-Rule Instability ($\Delta_{\mathrm{count}}$):**
   $$\Delta_{\mathrm{count}} = \min\Big(\overline{\mathrm{AMI}}_{\mathrm{within, full}},\; \overline{\mathrm{AMI}}_{\mathrm{within, frac}}\Big) - \overline{\mathrm{AMI}}_{\mathrm{full\ vs.\ frac}},$$
   evaluated at matched cluster granularity ($K_{\mathrm{frac}}(\gamma^*) = K_{\mathrm{full}}(\gamma=1.0)$) across $S = 20$ converged best-of-10 Leiden runs with CWTS greedy small-cluster merging.

3. **Guarded Three-Zone Editorial Triage (`ROPE = [0.10, 0.20]`):**
   - `LOW_SENSITIVITY_PASS` ($\Delta_{\mathrm{count}} \le 0.10$, zero guardrail flags)
   - `MODERATE_SENSITIVITY_CHECK_BOTH` ($0.10 < \Delta_{\mathrm{count}} < 0.20$)
   - `HIGH_SENSITIVITY_REPORT_FRACTIONAL` ($\Delta_{\mathrm{count}} \ge 0.20$)
   - Guardrails automatically flag fragmented co-authorship LCCs (`LCC / core < 50%`) and high-dispersion keyword landscapes (`within-AMI < 0.85` or `reassignment >= 25%`) so no unstable map receives an unqualified pass.

See [`INPUT_OUTPUT_GUIDE.md`](INPUT_OUTPUT_GUIDE.md) for complete database export instructions and output column specifications.

## License

Released under the [MIT License](LICENSE).
