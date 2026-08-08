#!/usr/bin/env python
"""Check that every gold citation actually resolves in the built corpus.

    python scripts/verify_goldset.py

A gold set is only useful if its labels are checkable. If a gold citation does
not exist in the index, then a retrieval "miss" on that item is measuring a hole
in the corpus, not a failure of retrieval — and reporting it as the latter would
overstate the problem. Run this before quoting any retrieval numbers.

It does NOT verify that the legal claims are correct. The gold verdicts were
written by hand and should be checked against the linked sources by a human
before the numbers go in a report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.config import load_config  # noqa: E402
from foga.evaluation.metrics import citation_match, normalize_citation  # noqa: E402
from foga.index import HybridIndex  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def main() -> int:
    cfg = load_config()
    index = HybridIndex.load(Path(cfg.path("index")))
    gold = [json.loads(l) for l in
            open(Path(cfg.path("eval")) / "goldset.jsonl", encoding="utf-8") if l.strip()]

    all_citations = sorted({c.citation for c in index.chunks})
    norm_index = {normalize_citation(c) for c in all_citations}

    ok = missing = abstention = 0
    print(f"index: {len(index):,} chunks, {len(all_citations):,} distinct citations")
    print(f"gold set: {len(gold)} claims\n")

    for item in gold:
        gc = item.get("gold_citation", "").strip()
        if not gc:
            abstention += 1
            print(f"  {DIM}SKIP{RESET} {item['id']:32s} (abstention item, no gold citation)")
            continue
        exact = normalize_citation(gc) in norm_index
        fuzzy = [c for c in all_citations if citation_match(c, gc)]
        if exact or fuzzy:
            ok += 1
            how = "exact" if exact else f"{len(fuzzy)} matching chunks"
            print(f"  {GREEN}OK  {RESET} {item['id']:32s} {gc:34s} {DIM}({how}){RESET}")
        else:
            missing += 1
            print(f"  {RED}MISS{RESET} {item['id']:32s} {gc:34s} "
                  f"{RED}not found in corpus{RESET}")

    print(f"\n{ok} resolvable, {missing} missing, {abstention} abstention items")
    if missing:
        print(f"\n{YELLOW}Missing citations mean the corpus lacks that provision.{RESET}")
        print("If they are from the USCIS Policy Manual, run the full download:")
        print("  python scripts/download.py --full")
        print("Otherwise the gold label may have a typo — fix it before reporting metrics.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
