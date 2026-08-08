"""The Immigration and Nationality Act, as codified at 8 U.S.C.

Source: the Office of the Law Revision Counsel bulk download. OLRC publishes
every title of the U.S. Code as a zipped USLM XML file at a "release point"
(the most recent public law incorporated), e.g.

    https://uscode.house.gov/download/releasepoints/us/pl/119/102/xml_usc08@119-102.zip

We discover the current release point from the official download page rather
than hardcoding it, so the corpus stays current when Congress passes new laws.

Each `<section>` becomes one Document, dual-cited as both "INA § 214" and
"8 U.S.C. § 1184" via `ina_crosswalk`.
"""

from __future__ import annotations

import datetime as dt
import re
import zipfile
from pathlib import Path

from lxml import etree

from ..config import Config
from ..schema import Document
from .http import PoliteSession
from .ina_crosswalk import ina_for_usc

USLM_NS = {"u": "http://xml.house.gov/schemas/uslm/1.0"}
DOWNLOAD_ROOT = "https://uscode.house.gov/download/"


def _find_release_point(session: PoliteSession, cfg: Config) -> str:
    """Scrape the one page OLRC provides for exactly this purpose: the download
    index. Returns the full URL of the current title-8 XML zip."""
    page = session.get(cfg.get_path("sources.usc.download_page"), suffix=".html")
    title = cfg.get_path("sources.usc.title", "08")
    if page:
        m = re.search(rf'href="(releasepoints/[^"]*xml_usc{title}@[^"]+\.zip)"', page)
        if m:
            return DOWNLOAD_ROOT + m.group(1)
    raise RuntimeError(
        "Could not find the current US Code release point on "
        f"{cfg.get_path('sources.usc.download_page')}"
    )


def download(cfg: Config, session: PoliteSession, force: bool = False) -> Path:
    url = _find_release_point(session, cfg)
    raw = Path(cfg.path("raw"))
    zpath = raw / Path(url).name
    print(f"  USC: GET {url}")
    got = session.get_bytes(url, zpath, force=force)
    if got is None:
        raise RuntimeError(f"US Code download failed: {url}")

    with zipfile.ZipFile(zpath) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
        target = raw / "usc_title8.xml"
        with zf.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    print(f"  USC: {target.stat().st_size / 1e6:.1f} MB -> {target.name}")
    return target


def _text_of(el) -> str:
    """USLM nests num/heading/content deeply. Render readable indented text."""
    out: list[str] = []

    def rec(node, depth: int) -> None:
        tag = etree.QName(node).localname if isinstance(node.tag, str) else ""
        if tag in ("note", "notes", "sourceCredit", "editorialNotes", "toc"):
            return  # historical annotations are noise for a fact-checker
        if tag in ("num", "heading", "chapeau", "content", "p", "continuation"):
            txt = " ".join(node.itertext())
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                out.append("  " * max(0, depth - 2) + txt)
            if tag in ("num", "heading"):
                return
        for child in node:
            rec(child, depth + 1)

    for child in el:
        rec(child, 0)
    # num + heading of a subsection belong on one line; stitch short lines up.
    merged: list[str] = []
    for line in out:
        s = line.strip()
        if merged and re.fullmatch(r"\(\w{1,4}\)", merged[-1].strip()):
            merged[-1] = merged[-1].rstrip() + " " + s
        else:
            merged.append(line)
    return "\n".join(merged)


def parse(cfg: Config, path: Path) -> list[Document]:
    as_of = dt.date.today().isoformat()
    tree = etree.parse(str(path))

    docs: list[Document] = []
    for sec in tree.iter("{http://xml.house.gov/schemas/uslm/1.0}section"):
        ident = sec.get("identifier", "")
        m = re.search(r"/t8/s([\w.\-]+)$", ident)
        if not m:
            continue
        usc_num = m.group(1)
        if sec.get("status") in ("repealed", "omitted", "transferred"):
            continue

        heading_el = sec.find("u:heading", USLM_NS)
        heading = " ".join(heading_el.itertext()).strip() if heading_el is not None else ""
        heading = re.sub(r"\s+", " ", heading)

        body = _text_of(sec)
        if len(body) < 80:
            continue

        ina = ina_for_usc(usc_num)
        # Dual citation. This is the whole point of the crosswalk: a user who
        # types "INA 214(b)" and a user who types "8 USC 1184(b)" must both hit.
        if ina:
            citation = f"INA § {ina} (8 U.S.C. § {usc_num})"
            alt = f"INA {ina}; INA section {ina}; 8 USC {usc_num}; 8 U.S.C. {usc_num}"
        else:
            citation = f"8 U.S.C. § {usc_num}"
            alt = f"8 USC {usc_num}"

        # Prepend the alias line so BM25 can match either citation style.
        text = f"Citation aliases: {alt}\n\n{body}"

        docs.append(
            Document(
                source="ina",
                doc_id=f"ina:8usc-{usc_num}",
                citation=citation,
                title=heading,
                text=text,
                url=f"https://www.law.cornell.edu/uscode/text/8/{usc_num}",
                breadcrumb=["Immigration and Nationality Act", "8 U.S.C."],
                as_of=as_of,
                extra={"usc_section": usc_num, "ina_section": ina},
            )
        )
    print(f"  USC: parsed {len(docs)} sections "
          f"({sum(1 for d in docs if d.extra.get('ina_section'))} with INA numbers)")
    return docs
