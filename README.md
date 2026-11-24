# Replication Package: Counting-Rule Instability in Bibliometric Science Maps

This archive accompanies the manuscript *"Counting-Rule Instability in Bibliometric Science Maps: Normalization Analysis and Stratified Empirical Audit"* (N. Long Ha, University of Economics, Hue University, `hnlong@hueuni.edu.vn`, ORCID `0000-0002-5673-2812`).

## Contents

- `biblio_audit/` — Clean-room MIT-licensed Python package (`biblio-audit` v1.0.0) implementing VOSviewer 1.6.20/1.6.21 normalization parity (`VOS_AS_CPM`, `VOS_Frac_CPM`, `Leiden_RBConfig`), matched-granularity resolution sweeps, `Delta_count`, exact `O(N)` `ECS` / hypergeometric `AECS`, metric `VI`, Hungarian core-node reassignment, causal temporal bipartite null models, and multi-database parsers.
- `experiments/src_audit/` — Experiment counting engine (`counting_engine.py`) and full-text reporting feature parser (`extract_audit_features.py`).
- `experiments/scripts/` — Deterministic data-harvesting, corpus-reconstruction, Layer-0/1/2 analysis scripts (`01_*.py` to `08_*.py`), and the symbolic/rational formal verification suite (`verify_propositions_sympy.py`).
- `experiments/tests/` — Automated unit and parity tests (`test_box1_parity.py`, `test_counting_engine.py`, `test_extract_audit_features.py`).
- `experiments/results/` — Anonymized JSON analysis summaries (`layer0_population_balance.json`, `layer1_final_adjudicated_census.json`, `layer1_llm_provisional_census.json`, `layer2_counting_audit_production.json`, `layer2_v1_expanded_benchmark.json`, `layer2_s1_and_rb_diagnostics.json`, `layer2_s2_s3_sensitivity.json`) and the structured verification workbook (`author_verification_pack.xlsx`).
- `requirements.txt` — Verified Python dependency list across all archived modules and scripts.
- `LICENSE` — MIT License (Copyright (c) 2026 N. Long Ha).

> **Note on licensed/copyrighted third-party materials:** Proprietary Scopus metadata exports and copyrighted full-text publisher PDFs/XMLs cannot be redistributed. Scripts `01_*.py`–`03_*.py` document the exact Scopus/OpenAlex retrieval pipeline for reproduction with institutional API credentials (`SCOPUS_API_KEY`, `OPENALEX_API_KEY`). All Layer-0, Layer-1, and Layer-2 empirical summary outputs, formal symbolic proofs, and package/engine test suites run fully offline.

## Quick Verification

### 1. Install dependencies and `biblio-audit`

```bash
python -m pip install -r requirements.txt
pip install -e ./biblio_audit
```

### 2. Run the automated test suites

```bash
pytest biblio_audit/tests experiments/tests -v
```

*(Note: `test_box1_parity.py` skips gracefully when raw local Scopus/OpenAlex corpus caches are omitted from the clean archive.)*

### 3. Run the formal symbolic and numerical proof verification

```bash
python experiments/scripts/verify_propositions_sympy.py
```

This verifies the simultaneous invariance condition, closed-form hub-to-topical distortion ratios, sign-flip thresholds, and `Q_RB` vs. CPM partition comparisons (Section 2.2 and Online Resource ESM 1.1), the disjoint-partition Personalized PageRank `ECS` collapse across `alpha in {0.1, 0.5, 0.85, 0.9, 1.0}`, the closed-form hypergeometric `AECS` expectation against exhaustive `S_N` permutations, `igraph` C CPM/RBConfig objective scaling (including the exact diagonal constant offset), and the 59-corpus Layer-2 HT4 Jensen gap against `experiments/results/layer2_counting_audit_production.json`.

### 4. Re-run Layer-2 sensitivity summaries from archived JSON results

```bash
python experiments/scripts/04_run_counting_audit.py --help
python experiments/scripts/06_run_s2_s3_and_sensitivity.py --help
python experiments/scripts/07_run_v1_expanded_benchmark.py --help
python experiments/scripts/08_run_s1_and_rb_diagnostics.py --help
```

## License

Released under the [MIT License](LICENSE).
