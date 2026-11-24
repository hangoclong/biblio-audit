"""Plug-and-play command-line interface for `biblio-audit` v1.0."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from . import __version__
from .engine import SUPPORTED_OBJECTIVES, CountingAuditResult, audit_artifact
from .exceptions import BiblioAuditError
from .networks import build_network_artifact, canonical_network_type
from .parsers import SUPPORTED_FORMATS, load_corpus_files, load_thesaurus
from .reporting import (
    build_provenance_payload,
    export_vosviewer_files,
    render_batch_markdown_table,
    render_latex_table,
    render_markdown_table,
    write_nodes_csv,
)

CLI_EPILOG = textwrap.dedent(
    """\
    Examples:
      1) Standard VOSviewer-parity Bibliographic Coupling audit on a Scopus CSV:
         biblio-audit --input scopus.csv --network bc --top-n 150 --md-out sensitivity.md

      2) Audit ALL feasible network families (BC, CC, CA, KW) across split Web of Science files:
         biblio-audit --input savedrecs_1.txt savedrecs_2.txt --network all \\
                      --thesaurus vos_thesaurus.txt --latex-out audit_table.tex --json-out audit.json

      3) Keyword co-occurrence with VOSviewer GUI defaults + node instability CSV + VOSviewer export:
         biblio-audit --input scopus.csv --network kw --vos-gui-defaults \\
                      --thesaurus thesaurus.txt --nodes-csv kw_shifters.csv --vos-export-dir vos_out/

      4) Full citation audit with 99 exchangeable causal temporal bipartite null rewires:
         biblio-audit --input openalex.json --network cc --null-rewires 99 --json-out cc_audit.json

    Supported formats (--format auto):
      Scopus CSV (legacy & 2023-2026), Web of Science TSV & Plain-Text, OpenAlex JSON & CSV,
      Dimensions CSV, and Lens.org CSV. See INPUT_OUTPUT_GUIDE.md for exact export steps.
    """
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="biblio-audit",
        description="VOSviewer-parity full-vs.-fractional counting sensitivity and clustering stability auditor.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more bibliometric export files (auto-deduplicated across multi-file batches).",
    )
    ap.add_argument(
        "--format",
        choices=list(SUPPORTED_FORMATS),
        default="auto",
        help="Input database format (default: auto).",
    )
    ap.add_argument(
        "--network",
        choices=["bc", "cc", "ca", "kw", "bibliographic_coupling", "co_citation", "co_authorship", "keyword_cooccurrence", "all"],
        default="bc",
        help="Network family to audit: bc (bibliographic coupling), cc (co-citation), ca (co-authorship), kw (keyword co-occurrence), or all.",
    )
    ap.add_argument(
        "--objective",
        choices=list(SUPPORTED_OBJECTIVES),
        default="VOS_AS_CPM",
        help="Clustering normalization/objective: VOS_AS_CPM (default Association Strength), VOS_Frac_CPM (VOSviewer Fractionalization), or Leiden_RBConfig (LinLog/Modularity).",
    )
    ap.add_argument(
        "--frac-scheme",
        choices=["vosviewer", "refined"],
        default="vosviewer",
        help="Fractional weight formula: vosviewer = 1/(d_e - 1) (Perianes-Rodriguez et al. 2016 default); refined = 2/(d_e*(d_e - 1)) (Leydesdorff & Park 2017).",
    )
    ap.add_argument(
        "--denom-scope",
        choices=["auto", "full-corpus", "selected-core"],
        default="auto",
        help="Event denominator scope: auto (full-corpus for CA/KW, selected-core for BC/CC), full-corpus, or selected-core.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Core node cap after min-occurrences filter (default: 150, or 1000 under --vos-gui-defaults; set 0 for unlimited).",
    )
    ap.add_argument(
        "--min-occurrences",
        type=int,
        default=None,
        help="Minimum document/citation/occurrence threshold per node (default: 1; under --vos-gui-defaults: BC=1, CC=20, CA=5, KW=5).",
    )
    ap.add_argument(
        "--core-selection",
        choices=["tls_within_network", "cited_by_count"],
        default="tls_within_network",
        help="Core document ranking rule for BC (default: tls_within_network, zero post-cutoff citation look-ahead).",
    )
    ap.add_argument(
        "--max-authors",
        type=int,
        default=25,
        help="Maximum authors per document for CA (VOSviewer default: 25; set 0 to disable).",
    )
    ap.add_argument(
        "--keyword-source",
        choices=["author", "index", "all"],
        default="author",
        help="Keyword source field for KW (default: author, with transparent fallback to index/topics if absent).",
    )
    ap.add_argument(
        "--thesaurus",
        type=str,
        default="",
        help="Optional VOSviewer 2-column TSV/CSV thesaurus (`label<TAB>replace_by`; empty replacement ignores term).",
    )
    ap.add_argument(
        "--vos-gui-defaults",
        action="store_true",
        help="Apply VOSviewer GUI Step-4/clustering defaults (`BC min=1, CC min=20, CA/KW min=5, top_n=1000, min_cluster_size=1`) for any unset flag.",
    )
    ap.add_argument("--seeds", type=int, default=20, help="Outer clustering replicates (default: 20).")
    ap.add_argument("--starts", type=int, default=10, help="Random starts per run (VOSviewer default: 10).")
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum cluster size before CWTS greedy merge (default: 5, or 1 under --vos-gui-defaults).",
    )
    ap.add_argument("--no-merge", action="store_true", help="Disable CWTS small-cluster merging.")
    ap.add_argument(
        "--null-rewires",
        type=int,
        default=0,
        help="Optional exchangeable bipartite null rewires (0 = skip for fast run; 99 = Phipson-Smyth p >= 0.01).",
    )
    ap.add_argument("--null-starts", type=int, default=3, help="Leiden starts per null evaluation (default: 3).")
    ap.add_argument("--json-out", type=str, default="", help="Path to write full provenance JSON report.")
    ap.add_argument("--md-out", type=str, default="", help="Path to write ready-to-paste Markdown table.")
    ap.add_argument("--latex-out", type=str, default="", help="Path to write ready-to-paste Booktabs LaTeX table.")
    ap.add_argument("--nodes-csv", type=str, default="", help="Path (or directory if --network all) to write per-node ECS stability CSV.")
    ap.add_argument("--vos-export-dir", type=str, default="", help="Directory to export VOSviewer `map.txt` and `network.txt` files.")
    return ap


def run_cli(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    # Explicit CLI bounds validation (zero silent assertions / negative slicing / NaN JSON)
    if args.seeds < 2:
        raise ValueError("--seeds must be >= 2.")
    if args.starts < 1:
        raise ValueError("--starts must be >= 1.")
    if args.top_n is not None and args.top_n < 0:
        raise ValueError("--top-n must be >= 0 (0 = unlimited).")
    if args.min_occurrences is not None and args.min_occurrences < 1:
        raise ValueError("--min-occurrences must be >= 1.")
    if args.min_cluster_size is not None and args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1.")
    if args.max_authors < 0:
        raise ValueError("--max-authors must be >= 0 (0 = disabled).")
    if args.null_rewires < 0:
        raise ValueError("--null-rewires must be >= 0.")
    if args.null_starts < 1:
        raise ValueError("--null-starts must be >= 1.")

    input_paths = [Path(p) for p in args.input]
    works, parse_warnings = load_corpus_files(input_paths, fmt=args.format)
    thesaurus = load_thesaurus(args.thesaurus) if args.thesaurus else None

    eff_top_n_raw = args.top_n if args.top_n is not None else (1000 if args.vos_gui_defaults else 150)
    top_n: int | None = None if eff_top_n_raw == 0 else eff_top_n_raw
    min_cs = args.min_cluster_size if args.min_cluster_size is not None else (1 if args.vos_gui_defaults else 5)
    _vos_gui_min_occ = {
        "bibliographic_coupling": 1,
        "co_citation": 20,
        "co_authorship": 5,
        "keyword_cooccurrence": 5,
    }

    target_nets = (
        ["bibliographic_coupling", "co_citation", "co_authorship", "keyword_cooccurrence"]
        if args.network == "all"
        else [canonical_network_type(args.network)]
    )

    results: list[CountingAuditResult] = []
    skipped: dict[str, str] = {}
    min_occ_by_net: dict[str, int] = {}
    for nt in target_nets:
        min_occ = args.min_occurrences if args.min_occurrences is not None else (_vos_gui_min_occ[nt] if args.vos_gui_defaults else 1)
        min_occ_by_net[nt] = min_occ
        try:
            art = build_network_artifact(
                works,
                network_type=nt,
                core_top_n=top_n,
                min_occurrences=min_occ,
                core_selection=args.core_selection,
                denom_scope=args.denom_scope,
                frac_scheme=args.frac_scheme,
                max_authors=args.max_authors,
                keyword_source=args.keyword_source,
                thesaurus=thesaurus,
            )
            res = audit_artifact(
                art,
                objective=args.objective,
                n_seeds=args.seeds,
                n_random_starts=args.starts,
                merge_small_clusters=not args.no_merge,
                min_cluster_size=min_cs,
                null_rewires=args.null_rewires,
                null_starts=args.null_starts,
            )
            results.append(res)
        except BiblioAuditError as exc:
            if args.network == "all":
                skipped[nt] = str(exc)
            else:
                raise

    if not results:
        raise BiblioAuditError(f"No feasible networks could be audited across {target_nets}. Skipped: {skipped}")

    input_label = input_paths[0].name if len(input_paths) == 1 else f"{input_paths[0].name} (+{len(input_paths) - 1} files)"
    md_text = (
        render_markdown_table(input_label, len(works), results[0])
        if len(results) == 1
        else render_batch_markdown_table(input_label, len(works), results)
    )
    if skipped:
        md_text += "\n**Skipped Networks (`--network all`):**\n" + "\n".join(f"- `{k}`: {v}" for k, v in skipped.items()) + "\n"
    print(md_text)

    if args.md_out:
        out_md = Path(args.md_out)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text, encoding="utf-8")

    if args.latex_out:
        out_tex = Path(args.latex_out)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(render_latex_table(input_label, len(works), results), encoding="utf-8")

    if args.nodes_csv:
        nc = Path(args.nodes_csv)
        if args.network == "all" or nc.is_dir():
            nc.mkdir(parents=True, exist_ok=True)
            for r in results:
                write_nodes_csv(nc / f"nodes_{r.network_type}.csv", r)
        else:
            write_nodes_csv(nc, results[0])

    vos_exports: dict[str, Any] = {}
    if args.vos_export_dir:
        vdir = Path(args.vos_export_dir)
        for r in results:
            vos_exports[r.network_type] = export_vosviewer_files(vdir, r)

    params = {
        "format": args.format,
        "network": args.network,
        "objective": args.objective,
        "frac_scheme": args.frac_scheme,
        "denom_scope": args.denom_scope,
        "top_n": top_n,
        "min_occurrences": (min_occ_by_net if len(min_occ_by_net) > 1 and args.min_occurrences is None and args.vos_gui_defaults else min_occ),
        "core_selection": args.core_selection,
        "max_authors": args.max_authors,
        "keyword_source": args.keyword_source,
        "thesaurus": bool(args.thesaurus),
        "vos_gui_defaults": args.vos_gui_defaults,
        "seeds": args.seeds,
        "starts": args.starts,
        "min_cluster_size": min_cs,
        "merge_small_clusters": not args.no_merge,
        "null_rewires": args.null_rewires,
    }
    payload = build_provenance_payload(
        version=__version__,
        input_paths=input_paths,
        n_works=len(works),
        params=params,
        results=results,
        parse_warnings=parse_warnings,
        vos_exports=vos_exports,
    )
    if skipped:
        payload["skipped_networks"] = skipped

    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
