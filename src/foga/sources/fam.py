"""9 FAM (Foreign Affairs Manual, Volume 9 — Visas), from the State Department.

fam.state.gov renders its table of contents from a JSON endpoint that the site's
own front end calls:

    GET https://fam.state.gov/api/Tree/GetTreeByVolumeId?Id=09FAM

That gives us the complete, authoritative list of section URLs — no crawling
or link-discovery required. We then fetch each leaf page once and cache it.

9 FAM is the guidance consular officers actually apply at visa interviews, so
it is the source that answers "will they approve my F-1 renewal" style
questions that 8 CFR alone cannot.

Each page carries a change-transmittal stamp like "(CT:VISA-1902; 02-01-2024)",
which we extract as the effective date — a fact-checker needs to know how old
its guidance is.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..config import Config
from ..schema import Document
from .http import PoliteSession

CT_RE = re.compile(r"\(CT:[A-Z]+-\d+;\s*(\d{2}-\d{2}-\d{4})\)")


def _flatten_tree(nodes: list[dict], crumb: list[str], out: list[dict]) -> None:
    for n in nodes:
        label = re.sub(r"\s+", " ", (n.get("text") or "")).strip()
        url = n.get("url")
        if url and url.lower().endswith(".html"):
            out.append({"id": n.get("id"), "text": label, "url": url,
                        "breadcrumb": list(crumb)})
        kids = n.get("items") or []
        if kids:
            _flatten_tree(kids, crumb + [label], out)


def download(cfg: Config, session: PoliteSession, force: bool = False) -> Path:
    volume = cfg.get_path("sources.fam.volume", "09FAM")
    base = cfg.get_path("sources.fam.base")
    tree_url = cfg.get_path("sources.fam.tree_api").format(volume=volume)

    raw_txt = session.get(tree_url, suffix=".json", force=force)
    if not raw_txt:
        raise RuntimeError(f"FAM tree API failed: {tree_url}")
    tree = json.loads(raw_txt)

    leaves: list[dict] = []
    _flatten_tree(tree, [f"{volume[:2]} FAM"], leaves)
    # The tree repeats ids across folder levels; dedupe on the page URL.
    seen: set[str] = set()
    leaves = [x for x in leaves if not (x["url"] in seen or seen.add(x["url"]))]
    print(f"  FAM: {len(leaves)} sections listed in the official TOC")

    manifest = Path(cfg.path("raw")) / f"fam_{volume}_manifest.json"
    pages: list[dict] = []
    for i, leaf in enumerate(leaves, 1):
        html = session.get(base + leaf["url"], suffix=".html", force=force)
        if html:
            leaf["html_len"] = len(html)
            pages.append(leaf)
        if i % 25 == 0 or i == len(leaves):
            print(f"    fetched {i}/{len(leaves)} "
                  f"(cache hits {session.stats['hits']}, new {session.stats['fetched']})")
    manifest.write_text(json.dumps(pages, indent=1), encoding="utf-8")
    print(f"  FAM: {len(pages)} pages -> {manifest.name}")
    return manifest


def parse(cfg: Config, manifest: Path) -> list[Document]:
    volume = cfg.get_path("sources.fam.volume", "09FAM")
    base = cfg.get_path("sources.fam.base")
    ua = "FOGA-parse"
    session = PoliteSession(ua, delay=0)  # cache-only; nothing new is fetched
    as_of = dt.date.today().isoformat()
    pages = json.loads(manifest.read_text())

    docs: list[Document] = []
    for page in pages:
        html = session.get(base + page["url"], suffix=".html")
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one("div.WordSection1") or soup.body
        if body is None:
            continue

        # Drop the classification banners that top and tail every FAM page.
        for junk in body.select("p.HeaderFooterClassificationIndicator, .MsoHeader, .MsoFooter"):
            junk.decompose()

        text = body.get_text("\n", strip=True)
        text = re.sub(r"^UNCLASSIFIED \(U\)\s*", "", text)
        text = re.sub(r"\bUNCLASSIFIED \(U\)\b", "", text)
        if len(text) < 200:
            continue

        eff = None
        m = CT_RE.search(text)
        if m:
            mm, dd, yyyy = m.group(1).split("-")
            eff = f"{yyyy}-{mm}-{dd}"

        # "9 FAM 402.1 OVERVIEW OF NIV CLASSIFICATIONS" -> cite / title split
        raw_title = re.sub(r"\s+", " ", page["text"]).strip()
        sec_id = page["id"] or ""
        m2 = re.match(r"^([\d.\-()A-Z]+)\s+(.*)$", raw_title)
        num, heading = (m2.group(1), m2.group(2)) if m2 else (sec_id, raw_title)
        citation = f"9 FAM {num}"

        docs.append(
            Document(
                source="fam",
                doc_id=f"fam:{volume}-{sec_id}",
                citation=citation,
                title=heading.title() if heading.isupper() else heading,
                text=text,
                url=base + page["url"],
                breadcrumb=page.get("breadcrumb", []),
                as_of=as_of,
                effective_date=eff,
            )
        )
    print(f"  FAM: parsed {len(docs)} sections "
          f"({sum(1 for d in docs if d.effective_date)} with effective dates)")
    return docs
