"""Print every reported number straight from the raw evaluation output.

Nothing here recomputes or interprets: it reads reports/*.json and prints what
is in them, so any figure in the report can be checked against its source file.

    .venv/bin/python scripts/show_results.py
"""
from __future__ import annotations

import glob
import json
import os

ORDER = {"no_retrieval": 0, "simple_retrieval": 1, "full_no_decompose": 2, "full": 3}


def rows_from(pattern: str) -> list[tuple[dict, str]]:
    out = []
    for path in glob.glob(pattern):
        data = json.load(open(path))
        for row in data.get("generation_configs", {}).get("table", []):
            out.append((row, os.path.basename(path)))
    return sorted(out, key=lambda r: ORDER.get(r[0]["config"], 9))


def show(title: str, pattern: str) -> None:
    rows = rows_from(pattern)
    if not rows:
        print(f"\n{title}\n  (no files matching {pattern})")
        return
    print(f"\n{title}")
    print(f"  {'config':<20} {'acc':>6} {'resid':>7} {'quote':>7} {'cite':>6}   source file")
    for row, fname in rows:
        print(f"  {row['config']:<20} {row['verdict_acc']:>6} {row['resid_halluc']:>7} "
              f"{row['quote_fid']:>7} {row['cite_prec']:>6}   {fname}")


show("CURRENT local numbers  (v3: after the PARTIALLY_SUPPORTED fix)", "reports/*v3_*.json")
show("PREVIOUS local numbers (v2: what the draft report quotes)", "reports/*v2_*.json")
show("FIRST run (v1: superseded, kept for the record)", "reports/*final_*.json")
show("Frontier model (gpt-5.6-luna)", "reports/*oai_*.json")

for path in sorted(glob.glob("reports/*v3_retrieval*.json")) or sorted(glob.glob("reports/*abl*.json")):
    table = json.load(open(path)).get("retrieval", {}).get("table", [])
    if not table:
        continue
    print(f"\nRetrieval ablation   source file: {os.path.basename(path)}")
    print(f"  {'config':<26} {'R@8':>6} {'MRR':>6}")
    for r in table:
        print(f"  {r.get('config',''):<26} {r.get('R@8','')!s:>6} {r.get('MRR','')!s:>6}")
    break

print("\nGold set:", sum(1 for _ in open("eval/goldset.jsonl")), "claims")
