"""Publication-ready Markdown, Booktabs LaTeX, JSON provenance, node CSV, and VOSviewer `.txt` reporters."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import CountingAuditResult


def compute_files_sha256(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    name_counts: dict[str, int] = {}
    for p in paths:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        key = str(p) if name_counts[p.name] > 1 else p.name
        out[key] = h.hexdigest()
    return out


def _md_escape_inline(s: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", str(s)).replace("|", r"\|").strip()


def render_markdown_table(
    input_label: str,
    n_works: int,
    res: CountingAuditResult,
) -> str:
    """Render a ready-to-paste Supplementary Sensitivity Disclosure Table in Markdown."""
    safe_label = _md_escape_inline(input_label)
    rows = [
        f"### Counting-Rule Sensitivity Disclosure (`biblio-audit`, VOSviewer Parity)",
        "",
        "| Diagnostic | Value |",
        "|---|---|",
        f"| Corpus input | `{safe_label}` ({n_works} records) |",
        f"| Network & objective | `{res.network_type}` / `{res.objective}` (frac=`{res.frac_scheme}`, denom=`{res.denom_scope}`, starts={res.n_random_starts}, min_size={res.min_cluster_size}) |",
        f"| Shared LCC / Core nodes (Coverage) | {res.n_lcc} / {res.n_core_selected} ({100.0 * res.lcc_fraction:.1f}%) |",
        f"| Event-degree Gini | {res.event_degree_gini:.4f} |",
        f"| Matched major clusters ($K_{{\\mathrm{{full}}}} \\to K_{{\\mathrm{{frac}}}}$) | {res.k_full_major} $\\to$ {res.k_frac_matched_major} (exact: {res.exact_k_matched}; $\\gamma^* = {res.gamma_plateau_mid:.4f}$) |",
        f"| Within-rule AMI (Full / Fractional) | {res.ami_within_full:.4f} / {res.ami_within_frac:.4f} |",
        f"| Cross-rule AMI / ARI | {res.ami_cross_mid:.4f} / {res.ari_cross_mid:.4f} |",
        f"| **Excess counting instability ($\\Delta_{{\\mathrm{{count}}}}$)** | **{res.delta_count_mid:.4f}** (reverse: {res.delta_count_reverse:.4f}) |",
        f"| Chance-adjusted ECS (cross / $\\Delta_{{\\mathrm{{AECS}}}}$) | {res.aecs_cross_mid:.4f} / {res.delta_aecs_mid:.4f} (raw ECS: {res.ecs_cross_mid:.4f}) |",
        f"| Variation of Information (VI bits / NVI) | {res.vi_cross_bits:.4f} / {res.nvi_cross:.4f} |",
        f"| Core-node reassignment ($1 - \\mathrm{{Hungarian}}$) | {100.0 * res.core_reassignment:.1f}% |",
        f"| Top-10 landmark retention | {100.0 * res.landmark_top10_retention:.1f}% |",
    ]
    if res.tfidf_term_jaccard is not None:
        rows.append(f"| Top-15 TF-IDF label Jaccard (BC titles) | {res.tfidf_term_jaccard:.4f} |")
    if res.null_benchmark is not None:
        nb = res.null_benchmark
        rows.append(
            f"| Bipartite null ({nb['method']}, R={nb['n_rewires']}) | "
            f"RB $Q_{{obs}}={nb['q_obs_rb_fixed_n']:.3f}$ vs $p_{{95}}={nb['q_null_rb_p95']:.3f}$ ($p={nb['p_value_rb']:.3f}$); "
            f"VOS $p={nb['p_value_vos']:.3f}$ |"
        )
    rows.append(f"| **Editorial triage badge (`ROPE [0.10, 0.20]`)** | **`{res.triage_band}`** |")
    if res.warnings:
        rows.append("")
        rows.append("**Guardrail Warnings:**")
        for w in res.warnings:
            rows.append(f"- {w}")
    rows.append("")
    return "\n".join(rows)


def render_batch_markdown_table(input_label: str, n_works: int, results: list[CountingAuditResult]) -> str:
    """Render a compact multi-network Markdown summary table (`--network all`)."""
    safe_label = _md_escape_inline(input_label)
    lines = [
        f"### Multi-Network Counting-Rule Sensitivity Audit (`biblio-audit`, `{safe_label}`, {n_works} records)",
        "",
        "| Network | LCC / Core | $K_{\\mathrm{full}} \\to K_{\\mathrm{frac}}$ | $\\Delta_{\\mathrm{count}}$ | $\\Delta_{\\mathrm{AECS}}$ | NVI | Reassign. | Landmarks | Triage Badge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.network_type}` | {r.n_lcc}/{r.n_core_selected} ({100.0*r.lcc_fraction:.0f}%) | "
            f"{r.k_full_major} $\\to$ {r.k_frac_matched_major} | **{r.delta_count_mid:.3f}** | "
            f"{r.delta_aecs_mid:.3f} | {r.nvi_cross:.3f} | {100.0*r.core_reassignment:.1f}% | "
            f"{100.0*r.landmark_top10_retention:.1f}% | `{r.triage_band}` |"
        )
    lines.append("")
    return "\n".join(lines)


_TEX_TRANS = str.maketrans({
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
})


def _tex_escape(s: str) -> str:
    return str(s).translate(_TEX_TRANS)


def render_latex_table(input_label: str, n_works: int, results: list[CountingAuditResult]) -> str:
    """Render a publication-ready Booktabs LaTeX table for manuscript Supplementary Material."""
    if len(results) == 1:
        r = results[0]
        body = [
            f"Corpus & \\texttt{{{_tex_escape(input_label)}}} ({n_works} records) \\\\",
            f"Network / Objective & \\texttt{{{_tex_escape(r.network_type)}}} / \\texttt{{{_tex_escape(r.objective)}}} \\\\",
            f"Shared LCC / Core nodes & {r.n_lcc} / {r.n_core_selected} ({100.0*r.lcc_fraction:.1f}\\%) \\\\",
            f"Event-degree Gini & {r.event_degree_gini:.4f} \\\\",
            f"Matched clusters ($K_{{\\mathrm{{full}}}} \\to K_{{\\mathrm{{frac}}}}$) & {r.k_full_major} $\\to$ {r.k_frac_matched_major} ($\\gamma^* = {r.gamma_plateau_mid:.4f}$) \\\\",
            f"Within-rule AMI (Full / Fractional) & {r.ami_within_full:.4f} / {r.ami_within_frac:.4f} \\\\",
            f"Cross-rule AMI / ARI & {r.ami_cross_mid:.4f} / {r.ari_cross_mid:.4f} \\\\",
            f"\\textbf{{Excess counting instability ($\\Delta_{{\\mathrm{{count}}}}$)}} & \\textbf{{{r.delta_count_mid:.4f}}} (rev.: {r.delta_count_reverse:.4f}) \\\\",
            f"Chance-adjusted ECS (cross / $\\Delta_{{\\mathrm{{AECS}}}}$) & {r.aecs_cross_mid:.4f} / {r.delta_aecs_mid:.4f} \\\\",
            f"Variation of Information (bits / NVI) & {r.vi_cross_bits:.4f} / {r.nvi_cross:.4f} \\\\",
            f"Core-node reassignment ($1 - \\mathrm{{Hungarian}}$) & {100.0*r.core_reassignment:.1f}\\% \\\\",
            f"Top-10 landmark retention & {100.0*r.landmark_top10_retention:.1f}\\% \\\\",
        ]
        if r.tfidf_term_jaccard is not None:
            body.append(f"Top-15 TF-IDF label Jaccard & {r.tfidf_term_jaccard:.4f} \\\\")
        if r.null_benchmark is not None:
            nb = r.null_benchmark
            body.append(
                f"Bipartite null (\\texttt{{{_tex_escape(nb['method'])}}}, $R={nb['n_rewires']}$) & "
                f"RB $p={nb['p_value_rb']:.3f}$; VOS $p={nb['p_value_vos']:.3f}$ \\\\"
            )
        body.append(f"\\textbf{{Editorial triage badge}} & \\texttt{{{_tex_escape(r.triage_band)}}} \\\\")
        notes_lines: list[str] = []
        if r.warnings:
            joined = "; ".join(_tex_escape(w) for w in r.warnings)
            notes_lines.append(f"\\multicolumn{{2}}{{p{{0.92\\linewidth}}}}{{\\footnotesize \\textit{{Guardrail warnings:}} {joined}}} \\\\")
        return "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Counting-rule sensitivity disclosure generated by \texttt{biblio-audit}.}",
            r"\label{tab:biblio_audit_sensitivity}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Diagnostic & Audited Value \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            *notes_lines,
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    rows_tex = []
    warn_notes: list[str] = []
    for r in results:
        rows_tex.append(
            f"\\texttt{{{_tex_escape(r.network_type)}}} & {r.n_lcc}/{r.n_core_selected} & "
            f"{r.k_full_major}$\\to${r.k_frac_matched_major} & {r.delta_count_mid:.3f} & "
            f"{r.delta_aecs_mid:.3f} & {r.nvi_cross:.3f} & {100.0*r.core_reassignment:.1f}\\% & "
            f"\\texttt{{{_tex_escape(r.triage_band)}}} \\\\"
        )
        for w in r.warnings:
            warn_notes.append(f"\\texttt{{{_tex_escape(r.network_type)}}}: {_tex_escape(w)}")
    notes_lines = []
    if warn_notes:
        notes_lines.append(
            f"\\multicolumn{{8}}{{p{{0.95\\linewidth}}}}{{\\footnotesize \\textit{{Guardrail warnings:}} {'; '.join(warn_notes)}}} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-network counting-rule sensitivity audit generated by \texttt{biblio-audit}.}",
        r"\label{tab:biblio_audit_multinet}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Network & LCC/Core & $K$ & $\Delta_{\mathrm{count}}$ & $\Delta_{\mathrm{AECS}}$ & NVI & Reassign. & Triage Badge \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        *notes_lines,
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def write_nodes_csv(path: Path, res: CountingAuditResult) -> None:
    """Export per-node consensus Medoid clusters and ECS stability sorted by instability descending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(res.node_rows, key=lambda r: (-r.delta_ecs_node, r.ecs_cross, r.node_id))
    fieldnames = [
        "node_id",
        "label",
        "cited_by_count",
        "occurrences",
        "full_cluster_medoid",
        "frac_cluster_aligned",
        "switched_cluster",
        "ecs_within_full",
        "ecs_within_frac",
        "ecs_cross",
        "delta_ecs_node",
        "is_boundary_shifter",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: getattr(row, k) for k in fieldnames})


def export_vosviewer_files(out_dir: Path, res: CountingAuditResult, prefix: str = "") -> dict[str, str]:
    """Export VOSviewer-compatible `map.txt` and `network_full.txt` / `network_frac.txt` for visual verification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{res.network_type}" if prefix else res.network_type
    map_path = out_dir / f"{stem}_vos_map.txt"
    net_full_path = out_dir / f"{stem}_vos_network_full.txt"
    net_frac_path = out_dir / f"{stem}_vos_network_frac.txt"

    with map_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "id",
            "label",
            "cluster",
            "score<frac_cluster>",
            "score<ecs_stability>",
            "score<delta_ecs_node>",
            "weight<Occurrences>",
            "weight<Citations>",
        ])
        for idx, nr in enumerate(res.node_rows, start=1):
            clean_lbl = re.sub(r"[\r\n\t]+", " ", nr.label).strip()[:120]
            if not clean_lbl:
                clean_lbl = re.sub(r"[\r\n\t]+", " ", nr.node_id).strip()[:120] or f"node_{idx}"
            w.writerow([
                idx,
                clean_lbl,
                nr.full_cluster_medoid + 1,
                nr.frac_cluster_aligned + 1,
                nr.ecs_cross,
                nr.delta_ecs_node,
                nr.occurrences,
                nr.cited_by_count,
            ])

    for npath, edges in ((net_full_path, res.lcc_full_edges), (net_frac_path, res.lcc_frac_edges)):
        with npath.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            for (u, v), wt in sorted(edges.items()):
                w.writerow([u + 1, v + 1, f"{wt:.6f}"])

    return {
        "vos_map": str(map_path),
        "vos_network_full": str(net_full_path),
        "vos_network_frac": str(net_frac_path),
    }


def build_provenance_payload(
    version: str,
    input_paths: list[Path],
    n_works: int,
    params: dict[str, Any],
    results: list[CountingAuditResult],
    parse_warnings: list[str],
    vos_exports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble full machine-readable JSON provenance report."""
    return {
        "software": "biblio-audit",
        "version": version,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": compute_files_sha256(input_paths),
        "n_works_loaded": n_works,
        "parameters": params,
        "parser_warnings": parse_warnings,
        "audits": {r.network_type: r.to_summary_dict() for r in results},
        "audit": results[0].to_summary_dict() if len(results) == 1 else None,
        "vosviewer_exports": vos_exports or {},
    }
