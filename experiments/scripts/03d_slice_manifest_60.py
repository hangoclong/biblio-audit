"""Slice harvested `layer2_manifest.json` deterministically in locked order to primary n=60 (30 Q1 + 30 Q2-Q4)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    full = json.loads((PROC_DIR / "layer2_manifest.json").read_text(encoding="utf-8"))
    q1 = [m for m in full if m["quartile"] == "Q1"]
    q24 = [m for m in full if m["quartile"] != "Q1"]

    sel_q1 = q1[:30]
    sel_q24 = q24[:30]
    sel = sel_q1 + sel_q24

    out_p = PROC_DIR / "layer2_manifest_60.json"
    out_p.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")

    qc = Counter(m["quartile"] for m in sel)
    qj = {q: len({m["journal_key"] for m in sel if m["quartile"] == q}) for q in ("Q1", "Q2", "Q3", "Q4")}
    nc = Counter(m["network_type"] for m in sel)
    hs = Counter(m["harvest_source"] for m in sel)
    print(f"Full harvested: {len(full)} (Q1={len(q1)}, Q2-4={len(q24)})")
    print(f"Sliced primary: {len(sel)} -> {dict(qc)} | unique journals={len({m['journal_key'] for m in sel})} ({qj})")
    print(f"Network types: {dict(nc)} | Harvest sources: {dict(hs)}")
    print(f"Min OA join={min(m['scopus_to_oa_join_rate'] for m in sel):.4f} | Min LCC={min(m['verified_lcc_size'] for m in sel)} | Min ref_yrs={min(m['n_ref_years_fetched'] for m in sel)}")


if __name__ == "__main__":
    main()
