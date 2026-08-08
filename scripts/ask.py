#!/usr/bin/env python
"""Step 3: check a claim from the command line.

    python scripts/ask.py "F-1 students may work 20 hours per week on campus"
    python scripts/ask.py "OPT is 12 months" --json
    python scripts/ask.py "H-1B cap is 65000" --sources ina cfr
    python scripts/ask.py --retrieval-only "STEM OPT extension length"
    python scripts/ask.py "..." --provider ollama     # run fully local

Exit code is 0 for SUPPORTED / NOT_ADDRESSED, 2 for CONTRADICTED, so the tool
can be scripted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.config import load_config  # noqa: E402
from foga.factcheck import FactChecker  # noqa: E402
from foga.index import HybridIndex  # noqa: E402
from foga.llm import LLM  # noqa: E402
from foga.retrieve import Retriever  # noqa: E402

C = {"SUPPORTED": "\033[92m", "CONTRADICTED": "\033[91m",
     "PARTIALLY_SUPPORTED": "\033[93m", "NOT_ADDRESSED": "\033[90m"}
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("claim", nargs="+", help="the claim or question to check")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    ap.add_argument("--markdown", action="store_true", help="emit a markdown report")
    ap.add_argument("--sources", nargs="+",
                    choices=["ina", "cfr", "fam", "uscis_pm", "fedreg"],
                    help="restrict retrieval to these authorities")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="show what retrieval returns, skip generation (free)")
    ap.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder")
    ap.add_argument("--no-decompose", action="store_true", help="check the claim whole")
    ap.add_argument("--provider", choices=["openai", "ollama"], help="override llm.provider")
    ap.add_argument("--model", help="override the model id")
    ap.add_argument("-k", type=int, help="number of passages to retrieve")
    args = ap.parse_args()

    claim = " ".join(args.claim)
    cfg = load_config()
    if args.no_decompose:
        cfg["factcheck"]["decompose"] = False

    index = HybridIndex.load(Path(cfg.path("index")))
    retriever = Retriever(index, cfg, use_reranker=not args.no_rerank)

    # --- retrieval only --------------------------------------------------
    if args.retrieval_only:
        res = retriever.search(claim, final_k=args.k, sources=args.sources)
        print(f"{BOLD}query:{RESET} {res.query}")
        if res.expanded_query != res.query:
            print(f"{DIM}expanded:{RESET} {res.expanded_query}")
        if res.abstain:
            print(f"\n{C['NOT_ADDRESSED']}ABSTAIN{RESET}: {res.abstain_reason}")
        print(f"\n{len(res.results)} passages:\n")
        for r in res.results:
            print(f"  [{r.tag}] {BOLD}{r.chunk.citation}{RESET} — {r.chunk.title[:60]}")
            print(f"       {DIM}{r.chunk.source} | {r.found_by} | "
                  f"rerank={r.rerank_score:.2f}{RESET}" if r.rerank_score is not None
                  else f"       {DIM}{r.chunk.source} | {r.found_by}{RESET}")
            print(f"       {r.chunk.text[:180].replace(chr(10), ' ')}...")
            print(f"       {DIM}{r.chunk.url}{RESET}\n")
        return 0

    # --- full check ------------------------------------------------------
    llm = LLM(provider=args.provider, model=args.model)
    checker = FactChecker(index, cfg, llm=llm, retriever=retriever)
    result = checker.check(claim, sources=args.sources, verbose=not args.json)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.markdown:
        print(result.to_markdown())
    else:
        color = C.get(result.verdict, "")
        print(f"\n{color}{BOLD}{result.verdict}{RESET}  "
              f"(confidence: {result.confidence})\n")
        print(f"{BOLD}{result.summary}{RESET}\n")
        print(result.explanation + "\n")
        if result.caveats:
            print(f"{BOLD}Caveats{RESET}")
            for c in result.caveats:
                print(f"  - {c}")
            print()
        if result.currency_warnings:
            print(f"{BOLD}Recent regulatory activity on this topic{RESET}")
            for w in result.currency_warnings:
                print(f"  - {w['date']} {w['title'][:70]}")
                print(f"    {DIM}{w['url']}{RESET}")
            print()
        print(f"{BOLD}Sources{RESET}")
        seen = set()
        for s in result.subclaims:
            for src in s.sources:
                key = (src["citation"], src["url"])
                if key in seen:
                    continue
                seen.add(key)
                print(f"  [{src['tag']}] {src['citation']} ({src['source']})")
                print(f"      {DIM}{src['url']}{RESET}")
        print()
        for s in result.subclaims:
            g = s.grounding.get("grade", "n/a")
            print(f"  sub-claim {s.id}: {s.verdict} | grounding {g} "
                  f"| {s.grounding.get('summary', '')}")
        st = result.stats
        print(f"\n{DIM}{st['elapsed_s']}s | {st['llm_calls']} LLM calls | "
              f"{st['input_tokens']}+{st['output_tokens']} tokens | "
              f"~${st['est_cost_usd']:.4f} | {st['model']}{RESET}")

    return 2 if result.verdict == "CONTRADICTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
