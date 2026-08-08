#!/usr/bin/env python
"""Step 1 of the pipeline: acquire the corpus from official government sources.

    python scripts/download.py                 # all sources, USCIS PM subset
    python scripts/download.py --full          # all sources, full USCIS PM (~3.6 h)
    python scripts/download.py --only ecfr usc # just those sources
    python scripts/download.py --stats         # what's on disk already

Every source here is an official bulk endpoint, an official API, or an
officially published URL manifest. Nothing is discovered by crawling links,
and every response is cached so a URL is requested at most once.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.config import load_config  # noqa: E402
from foga.schema import write_jsonl  # noqa: E402
from foga.sources import ecfr, fam, fedreg, irs, usc, uscis_pm  # noqa: E402
from foga.sources.http import PoliteSession  # noqa: E402

SOURCES = ["ecfr", "usc", "fam", "uscis_pm", "fedreg", "irs"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=SOURCES, help="restrict to these sources")
    ap.add_argument("--full", action="store_true",
                    help="fetch the complete USCIS Policy Manual, not the subset")
    ap.add_argument("--force", action="store_true", help="ignore the local cache")
    ap.add_argument("--stats", action="store_true", help="report on disk contents and exit")
    args = ap.parse_args()

    cfg = load_config()
    raw = Path(cfg.path("raw"))
    processed = Path(cfg.path("processed"))

    if args.stats:
        print(f"raw dir: {raw}")
        for p in sorted(raw.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(raw)!s:50s} {p.stat().st_size / 1e6:8.2f} MB")
        cache = raw / "_cache"
        if cache.exists():
            print(f"\ncached pages: {len(list(cache.glob('*')))}")
        for p in sorted(processed.glob("*.jsonl")):
            n = sum(1 for _ in open(p))
            print(f"  {p.name:40s} {n:6d} docs")
        return 0

    todo = args.only or SOURCES
    ua = cfg.get_path("sources.uscis_pm.user_agent")
    t0 = time.time()
    all_docs = []

    # --- 8 CFR ---------------------------------------------------------
    if "ecfr" in todo and cfg.get_path("sources.ecfr.enabled"):
        print("\n[1/5] 8 CFR via the eCFR API (single bulk XML request)")
        s = PoliteSession(ua, delay=1.0)
        docs = ecfr.parse(cfg, ecfr.download(cfg, s, force=args.force))
        write_jsonl(processed / "docs_cfr.jsonl", docs)
        all_docs += docs

    # --- INA / 8 USC ---------------------------------------------------
    if "usc" in todo and cfg.get_path("sources.usc.enabled"):
        print("\n[2/5] INA (8 U.S.C.) via the OLRC bulk release point")
        s = PoliteSession(ua, delay=1.0)
        docs = usc.parse(cfg, usc.download(cfg, s, force=args.force))
        write_jsonl(processed / "docs_ina.jsonl", docs)
        all_docs += docs

    # --- 9 FAM ---------------------------------------------------------
    if "fam" in todo and cfg.get_path("sources.fam.enabled"):
        print("\n[3/5] 9 FAM via the State Department TOC API")
        s = PoliteSession(ua, delay=1.5)
        docs = fam.parse(cfg, fam.download(cfg, s, force=args.force))
        write_jsonl(processed / "docs_fam.jsonl", docs)
        all_docs += docs

    # --- USCIS Policy Manual -------------------------------------------
    if "uscis_pm" in todo and cfg.get_path("sources.uscis_pm.enabled"):
        print("\n[4/5] USCIS Policy Manual via the official sitemap")
        delay = cfg.get_path("sources.uscis_pm.crawl_delay_seconds", 10)
        s = PoliteSession(ua, delay=delay)
        m = uscis_pm.download(cfg, s, force=args.force, full=args.full)
        docs = uscis_pm.parse(cfg, m)
        write_jsonl(processed / "docs_uscis_pm.jsonl", docs)
        all_docs += docs

    # --- Federal Register ----------------------------------------------
    if "fedreg" in todo and cfg.get_path("sources.fedreg.enabled"):
        print("\n[5/5] Federal Register via the public JSON API")
        s = PoliteSession(ua, delay=1.0)
        docs = fedreg.parse(cfg, fedreg.download(cfg, s, force=args.force))
        write_jsonl(processed / "docs_fedreg.jsonl", docs)
        all_docs += docs

    if "irs" in todo and cfg.get_path("sources.irs.enabled", True):
        print("\n[6/6] IRS publications from irs.gov/pub/irs-pdf (stable PDFs)")
        s = PoliteSession(ua, delay=1.0)
        docs = irs.parse(cfg, irs.download(cfg, s, force=args.force))
        write_jsonl(processed / "docs_irs.jsonl", docs)
        all_docs += docs

    # --- combined ------------------------------------------------------
    combined = processed / "documents.jsonl"
    merged: list = []
    for p in sorted(processed.glob("docs_*.jsonl")):
        merged += [__import__("json").loads(l) for l in open(p) if l.strip()]
    with open(combined, "w", encoding="utf-8") as fh:
        import json
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_src: dict[str, int] = {}
    chars = 0
    for r in merged:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        chars += len(r["text"])

    print(f"\n{'=' * 62}\nCORPUS SUMMARY  ({time.time() - t0:.0f}s)")
    for k, v in sorted(by_src.items()):
        print(f"  {k:10s} {v:6d} documents")
    print(f"  {'TOTAL':10s} {len(merged):6d} documents, {chars / 1e6:.1f}M characters")
    print(f"  -> {combined}")
    print("\nNext:  python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
