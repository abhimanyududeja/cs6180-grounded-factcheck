"""USCIS Policy Manual.

USCIS publishes no bulk file for the Policy Manual, so this is the one source
that requires fetching page by page. We do it the legitimate way:

  * **Enumeration comes from the official sitemap**, `uscis.gov/sitemap.xml`,
    which USCIS publishes precisely so automated clients know the canonical URL
    set. We never follow links or guess URLs. (1,302 Policy Manual URLs today.)
  * **Rate is set by their robots.txt**, which specifies `Crawl-delay: 10` for
    all agents and does *not* disallow `/policy-manual/`. We honor that delay
    literally: 10 s between requests.
  * **Each URL is fetched exactly once** and cached to disk forever.

`subset_volumes` in config.yaml restricts the first run to the volumes that
matter for international students and employment-based applicants, so you can
build and test the whole system in ~25 minutes instead of ~3.6 hours. Run with
`--full` later to complete the corpus; already-cached pages are not re-fetched.

Policy Manual chapters carry numbered footnotes that cite the underlying INA
and 8 CFR provisions. We keep those footnotes: they are what lets the system
say "USCIS guidance says X, and it rests on 8 CFR 214.2(f)(9)".
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree

from ..config import Config
from ..schema import Document
from .http import PoliteSession

PM_URL_RE = re.compile(
    r"https://www\.uscis\.gov/policy-manual/volume-(\d+)(?:-part-([a-z]+))?(?:-chapter-(\d+))?$"
)


def _sitemap_urls(session: PoliteSession, sitemap: str) -> list[str]:
    """Read the sitemap index, then every child sitemap, keeping Policy Manual
    URLs. This is the officially published URL list."""
    idx = session.get(sitemap, suffix=".xml")
    if not idx:
        raise RuntimeError(f"Could not read {sitemap}")
    root = etree.fromstring(idx.encode())
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    children = [e.text for e in root.findall(".//s:sitemap/s:loc", ns)]

    urls: list[str] = []
    for child in children:
        body = session.get(child, suffix=".xml")
        if not body:
            continue
        sub = etree.fromstring(body.encode())
        for loc in sub.findall(".//s:url/s:loc", ns):
            if loc.text and "/policy-manual/" in loc.text and "/es/" not in loc.text:
                urls.append(loc.text.rstrip("/"))
    return sorted(set(urls))


def _select(urls: list[str], subset_volumes: list[int] | None) -> list[str]:
    """Keep chapter-level pages (they hold the text); optionally restrict to
    a subset of volumes."""
    keep: list[str] = []
    for u in urls:
        m = PM_URL_RE.match(u)
        if not m:
            continue
        vol = int(m.group(1))
        is_chapter = m.group(3) is not None
        if subset_volumes and vol not in subset_volumes:
            continue
        # Chapter pages carry the substance; part/volume pages are tables of
        # contents. Keep part pages too — some parts hold text directly.
        if is_chapter or m.group(2) is not None:
            keep.append(u)
    return keep


def download(
    cfg: Config, session: PoliteSession, force: bool = False, full: bool = False
) -> Path:
    sitemap = cfg.get_path("sources.uscis_pm.sitemap")
    subset = None if full else cfg.get_path("sources.uscis_pm.subset_volumes")

    all_urls = _sitemap_urls(session, sitemap)
    targets = _select(all_urls, subset)
    scope = "FULL Policy Manual" if full else f"volumes {subset}"
    est = len(targets) * session.delay / 60
    print(f"  USCIS PM: sitemap lists {len(all_urls)} policy-manual URLs; "
          f"fetching {len(targets)} ({scope})")
    print(f"  USCIS PM: honoring robots.txt Crawl-delay={session.delay}s "
          f"-> ~{est:.0f} min for uncached pages")

    manifest = Path(cfg.path("raw")) / (
        "uscis_pm_manifest_full.json" if full else "uscis_pm_manifest.json"
    )
    got: list[dict] = []
    for i, url in enumerate(targets, 1):
        html = session.get(url, suffix=".html", force=force)
        if html:
            got.append({"url": url, "html_len": len(html)})
        if i % 20 == 0 or i == len(targets):
            print(f"    {i}/{len(targets)} "
                  f"(cached {session.stats['hits']}, new {session.stats['fetched']}, "
                  f"err {session.stats['errors']})")
    manifest.write_text(json.dumps(got, indent=1), encoding="utf-8")
    print(f"  USCIS PM: {len(got)} pages -> {manifest.name}")
    return manifest


def _parse_page(html: str, url: str, as_of: str) -> Document | None:
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("h1")
    page_title = h1.get_text(" ", strip=True) if h1 else ""

    article = soup.select_one("article") or soup.select_one("#content") or soup.body
    if article is None:
        return None
    for junk in article.select(
        "nav, header, footer, script, style, .breadcrumb, .usa-sidenav, "
        ".book-navigation, .toc, .visually-hidden, .uscis-alert"
    ):
        junk.decompose()

    text = article.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) < 400:
        return None

    m = PM_URL_RE.match(url)
    if not m:
        return None
    vol, part, chap = m.group(1), (m.group(2) or "").upper(), m.group(3)
    citation = f"USCIS Policy Manual Vol. {vol}"
    if part:
        citation += f", Part {part}"
    if chap:
        citation += f", Ch. {chap}"

    # "Chapter 2 - Eligibility Requirements | USCIS" -> "Eligibility Requirements"
    heading = re.sub(r"\s*\|\s*USCIS\s*$", "", page_title)
    heading = re.sub(r"^(Chapter|Part)\s+[\w\d]+\s*[-–]\s*", "", heading).strip()

    # Pull the "Last Reviewed/Updated" stamp USCIS puts at the page foot.
    eff = None
    stamp = re.search(r"Last Reviewed/Updated:\s*(\d{2}/\d{2}/\d{4})", html)
    if stamp:
        mm, dd, yyyy = stamp.group(1).split("/")
        eff = f"{yyyy}-{mm}-{dd}"

    return Document(
        source="uscis_pm",
        doc_id=f"uscis_pm:{url.rsplit('/', 1)[-1]}",
        citation=citation,
        title=heading or page_title,
        text=text,
        url=url,
        breadcrumb=["USCIS Policy Manual", f"Volume {vol}"] + ([f"Part {part}"] if part else []),
        as_of=as_of,
        effective_date=eff,
    )


def parse(cfg: Config, manifest: Path) -> list[Document]:
    session = PoliteSession("FOGA-parse", delay=0)  # cache-only
    as_of = dt.date.today().isoformat()
    pages = json.loads(manifest.read_text())

    docs: list[Document] = []
    for page in pages:
        html = session.get(page["url"], suffix=".html")
        if not html:
            continue
        doc = _parse_page(html, page["url"], as_of)
        if doc:
            docs.append(doc)
    print(f"  USCIS PM: parsed {len(docs)} chapters")
    return docs
