"""8 CFR — Aliens and Nationality, via the official eCFR API.

This is a genuine bulk endpoint: one unauthenticated GET returns the entire
title as ~5 MB of XML.

    GET https://www.ecfr.gov/api/versioner/v1/full/{YYYY-MM-DD}/title-8.xml

Structure (National Archives OFR schema):
    DIV1 TITLE > DIV3 CHAPTER > DIV4 SUBCHAP > DIV5 PART > DIV6 SUBPART > DIV8 SECTION

We emit one Document per DIV8 SECTION, carrying the full breadcrumb so a
retrieved chunk can say which part and subpart it came from.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from lxml import etree

from ..config import Config
from ..schema import Document
from .http import PoliteSession

_LEVELS = {
    "DIV1": "title", "DIV2": "subtitle", "DIV3": "chapter", "DIV4": "subchapter",
    "DIV5": "part", "DIV6": "subpart", "DIV7": "subject_group", "DIV8": "section",
}


def _clean_head(el) -> str:
    """Extract a heading, collapsing the wide spacing eCFR uses after §."""
    head = el.find("HEAD")
    if head is None:
        return ""
    txt = " ".join(head.itertext())
    return re.sub(r"\s+", " ", txt).strip().rstrip(".")


def _render_table(tbl) -> str:
    """Render a table as pipe-delimited rows. Left as raw itertext it becomes a
    field of blank lines that destroys chunk boundaries."""
    rows: list[str] = []
    for tr in tbl.iter("TR"):
        cells = [re.sub(r"\s+", " ", " ".join(td.itertext())).strip() for td in tr]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_section_contents(el) -> bool:
    """The 'Table N to § X—Section Contents' block at the head of a long section
    lists every subsection letter, (a) through (w). Feeding it to the chunker's
    numbering stack makes the stack believe the section has already reached (w)
    before the body's real (a) even starts, so every subsequent path comes out
    wrong. It carries no legal content — it is a table of contents — so drop it.
    """
    for cap in el.iter("CAPTION"):
        if "section contents" in " ".join(cap.itertext()).lower():
            return True
    return False


def _section_text(sec) -> str:
    """Flatten a SECTION's body into paragraphs, skipping the heading."""
    parts: list[str] = []
    for child in sec:
        if child.tag == "HEAD":
            continue
        if _is_section_contents(child):
            continue
        tables = list(child.iter("TABLE"))
        if tables:
            for tbl in tables:
                rendered = _render_table(tbl)
                if rendered:
                    parts.append(rendered)
            continue
        txt = " ".join(child.itertext())
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        if txt:
            # CITA / SOURCE / AUTH give the regulatory history — keep them,
            # they are how a user checks whether a rule is current.
            if child.tag in ("CITA", "SOURCE", "AUTH", "EDNOTE"):
                parts.append(f"[{child.tag.title()}] {txt}")
            else:
                parts.append(txt)
    return "\n\n".join(parts)


def download(cfg: Config, session: PoliteSession, force: bool = False) -> Path:
    api = cfg.get_path("sources.ecfr.api")
    title = cfg.get_path("sources.ecfr.title", 8)
    # eCFR requires a date on/before the latest issue date; "yesterday" is safe.
    date = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    url = f"{api}/full/{date}/title-{title}.xml"
    dest = Path(cfg.path("raw")) / f"ecfr_title{title}.xml"
    print(f"  eCFR: GET {url}")
    out = session.get_bytes(url, dest, force=force)
    if out is None:
        raise RuntimeError(f"eCFR download failed: {url}")
    print(f"  eCFR: {out.stat().st_size / 1e6:.1f} MB -> {out.name}")
    return out


def parse(cfg: Config, path: Path) -> list[Document]:
    title = cfg.get_path("sources.ecfr.title", 8)
    as_of = dt.date.today().isoformat()
    tree = etree.parse(str(path))
    root = tree.getroot()

    docs: list[Document] = []
    # Walk down carrying an ancestor breadcrumb.
    def walk(el, crumb: list[str]) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        level = _LEVELS.get(tag)
        if level == "section":
            head = _clean_head(el)
            num = el.get("N", "")
            body = _section_text(el)
            if not body or len(body) < 60:
                return
            # "§ 214.2 Special requirements..." -> citation "8 CFR 214.2"
            heading = re.sub(r"^§+\s*[\d.\-a-zA-Z]+\s*", "", head).strip() or head
            docs.append(
                Document(
                    source="cfr",
                    doc_id=f"cfr:{title}-{num}",
                    citation=f"{title} CFR § {num}",
                    title=heading,
                    text=body,
                    url=f"https://www.ecfr.gov/current/title-{title}/section-{num}",
                    breadcrumb=list(crumb),
                    as_of=as_of,
                )
            )
            return
        if level:
            crumb = crumb + [_clean_head(el) or f"{level} {el.get('N','')}"]
        for child in el:
            walk(child, crumb)

    walk(root, [])
    print(f"  eCFR: parsed {len(docs)} sections")
    return docs
