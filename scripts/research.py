#!/usr/bin/env python
"""Agentic research mode — the model runs its own searches.

    python scripts/research.py "Can I start STEM OPT while my H-1B is pending?"
    python scripts/research.py "..." --steps 10 --json

Use this for questions the fixed workflow cannot answer in one retrieval pass:
anything that needs a follow-up search you could only know to run after reading
the first result. Compare it against `scripts/ask.py` on the same question —
the difference is the interesting result for the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.agent import ImmigrationAgent  # noqa: E402
from foga.config import load_config  # noqa: E402
from foga.index import HybridIndex  # noqa: E402
from foga.llm import LLM  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="+")
    ap.add_argument("--steps", type=int, default=8, help="max research steps")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--provider", choices=["openai", "ollama"])
    ap.add_argument("--model")
    args = ap.parse_args()

    question = " ".join(args.question)
    cfg = load_config()
    index = HybridIndex.load(Path(cfg.path("index")))
    llm = LLM(provider=args.provider, model=args.model)
    agent = ImmigrationAgent(index, cfg, llm=llm, max_steps=args.steps)

    if not args.json:
        print(f"{BOLD}Question:{RESET} {question}\n")
        print(f"{BOLD}Research trace{RESET}")

    result = agent.run(question, verbose=not args.json)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\n{BOLD}Answer{RESET}\n")
    print(result.answer)
    if result.citations:
        print(f"\n{BOLD}Sources consulted{RESET}")
        for c in result.citations:
            print(f"  {c['citation']} — {c['title'][:60]}")
            print(f"    {DIM}{c['url']}{RESET}")
    st = result.stats
    print(f"\n{DIM}{st['elapsed_s']}s | {st['tool_calls']} tool calls | "
          f"{st['llm_calls']} LLM calls | ~${st['est_cost_usd']:.4f} | "
          f"{st['model']}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
