#!/usr/bin/env python
"""Step 2: chunk the corpus and build the hybrid retrieval index.

    python scripts/build_index.py                 # full build
    python scripts/build_index.py --dry-run       # chunk + report cost, no embedding
    python scripts/build_index.py --local-embed   # free local embeddings, no API

Embedding cost with text-embedding-3-small is about $0.02 per million tokens,
so a corpus of this size runs well under a dollar. `--dry-run` prints the exact
estimate before you spend anything.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.chunk import chunk_corpus  # noqa: E402
from foga.config import load_config  # noqa: E402
from foga.index import HybridIndex  # noqa: E402
from foga.llm import Embedder  # noqa: E402
from foga.schema import load_documents  # noqa: E402

PRICE_PER_MTOK = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="chunk and report, do not embed")
    ap.add_argument("--local-embed", action="store_true",
                    help="use a local sentence-transformers model instead of the API")
    args = ap.parse_args()

    cfg = load_config()
    processed = Path(cfg.path("processed"))
    docs_path = processed / "documents.jsonl"
    if not docs_path.exists():
        print(f"No corpus at {docs_path}. Run: python scripts/download.py")
        return 1

    t0 = time.time()
    docs = load_documents(docs_path)
    print(f"loaded {len(docs):,} documents")

    chunks = chunk_corpus(docs, cfg)
    print(f"chunked into {len(chunks):,} chunks in {time.time() - t0:.1f}s\n")

    # --- report ---------------------------------------------------------
    by_src = collections.Counter(c.source for c in chunks)
    print(f"{'source':12s} {'chunks':>8s} {'avg chars':>10s} {'distinct cites':>15s}")
    print("-" * 49)
    for src, n in sorted(by_src.items()):
        sub = [c for c in chunks if c.source == src]
        avg = sum(c.n_chars for c in sub) / max(1, len(sub))
        print(f"{src:12s} {n:8,d} {avg:10.0f} {len({c.citation for c in sub}):15,d}")
    total_chars = sum(c.n_chars for c in chunks)
    est_tokens = total_chars / 4
    print("-" * 49)
    print(f"{'TOTAL':12s} {len(chunks):8,d} {total_chars / 1e6:9.1f}M chars "
          f"~{est_tokens / 1e6:.1f}M tokens")

    model = cfg.get_path("retrieval.embed_model")
    if args.local_embed:
        print("\nembeddings: local sentence-transformers (free)")
    else:
        cost = est_tokens / 1e6 * PRICE_PER_MTOK.get(model, 0.02)
        print(f"\nembeddings: {model} -> estimated cost ${cost:.2f}")

    if args.dry_run:
        print("\n--dry-run: stopping before embedding.")
        return 0

    # --- build ----------------------------------------------------------
    embedder = Embedder(provider="local" if args.local_embed else None)
    print()
    index = HybridIndex.build(chunks, embedder)
    out = Path(cfg.path("index"))
    index.save(out)

    print(f"\nindex written to {out}")
    print(f"  dense.faiss   {(out / 'dense.faiss').stat().st_size / 1e6:.1f} MB")
    print(f"  bm25.pkl      {(out / 'bm25.pkl').stat().st_size / 1e6:.1f} MB")
    print(f"  chunks.jsonl  {(out / 'chunks.jsonl').stat().st_size / 1e6:.1f} MB")
    print(f"\ntotal {time.time() - t0:.0f}s")
    print("\nNext:  python scripts/ask.py \"F-1 students can work 20 hours per week off campus\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
