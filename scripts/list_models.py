#!/usr/bin/env python
"""Print the models your OpenAI account can actually reach.

    python scripts/list_models.py
    python scripts/list_models.py --check gpt-5.6-luna

Model names move. If a default in config.yaml has been retired, this tells you
what to replace it with rather than leaving you to decode a 404.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.config import load_config, require_env  # noqa: E402

# Reference prices per 1M tokens (August 2026) for the models this project uses.
KNOWN_PRICES = {
    "gpt-5.6-sol": "$5.00 / $30.00",
    "gpt-5.6-terra": "$2.00 / $12.00",
    "gpt-5.6-luna": "$0.20 / $1.20",
    "gpt-5-mini": "$0.25 / $2.00",
    "gpt-5-nano": "$0.05 / $0.40",
    "text-embedding-3-small": "$0.02",
    "text-embedding-3-large": "$0.13",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", help="test that this model id actually responds")
    args = ap.parse_args()

    from openai import OpenAI

    cfg = load_config()
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    if args.check:
        print(f"calling {args.check} ...")
        r = client.chat.completions.create(
            model=args.check,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        )
        print(f"  -> {r.choices[0].message.content!r}")
        print(f"  usage: {r.usage.prompt_tokens} in / {r.usage.completion_tokens} out")
        return 0

    ids = sorted(m.id for m in client.models.list().data)
    chat = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4"))]
    embed = [i for i in ids if "embedding" in i]

    print(f"{len(ids)} models available to this key\n")
    print("CHAT / REASONING")
    for i in chat:
        price = KNOWN_PRICES.get(i, "")
        mark = "  <- configured" if i == cfg.get_path("llm.model") else ""
        print(f"  {i:34s} {price:18s}{mark}")
    print("\nEMBEDDINGS")
    for i in embed:
        price = KNOWN_PRICES.get(i, "")
        mark = "  <- configured" if i == cfg.get_path("retrieval.embed_model") else ""
        print(f"  {i:34s} {price:18s}{mark}")

    print("\nprices are input / output per 1M tokens, August 2026")
    print("configured in config.yaml:")
    print(f"  llm.model              {cfg.get_path('llm.model')}")
    print(f"  llm.judge_model        {cfg.get_path('llm.judge_model')}")
    print(f"  retrieval.embed_model  {cfg.get_path('retrieval.embed_model')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
