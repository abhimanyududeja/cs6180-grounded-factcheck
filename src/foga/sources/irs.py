"""IRS publications and form instructions — the tax half of the corpus.

Why tax sits alongside immigration: residence is the same question in both
bodies of law. Whether someone is a resident alien decides which immigration
rules apply *and* which tax return they file, and Publication 519 exists
precisely to explain the tax consequences of an immigration status. A claim like
"an F-1 student on OPT pays Social Security tax" cannot be settled from either
corpus alone.

IRS publications are served as stable PDFs from irs.gov/pub/irs-pdf/, so this
adapter needs no scraping and no API key. The paths have been stable for years,
and each is requested once and cached like every other source.

Extraction is the hard part rather than fetching. IRS publications are typeset
in narrow columns, so `pypdf` returns text broken mid-word at every line end and
littered with printer's marks. Left alone that text is embedded and retrieved as
if it were substantive guidance, so it is repaired here before chunking.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Config
from ..schema import Document, normalize_ws
from .http import PoliteSession

# filename stem -> (url, citation, human title)
PUBLICATIONS = {
    "irs_pub17": (
        "https://www.irs.gov/pub/irs-pdf/p17.pdf",
        "IRS Publication 17",
        "Your Federal Income Tax (For Individuals)",
    ),
    "irs_pub501": (
        "https://www.irs.gov/pub/irs-pdf/p501.pdf",
        "IRS Publication 501",
        "Dependents, Standard Deduction, and Filing Information",
    ),
    "irs_pub519": (
        "https://www.irs.gov/pub/irs-pdf/p519.pdf",
        "IRS Publication 519",
        "U.S. Tax Guide for Aliens",
    ),
    "irs_pub970": (
        "https://www.irs.gov/pub/irs-pdf/p970.pdf",
        "IRS Publication 970",
        "Tax Benefits for Education",
    ),
    "irs_form1040_instructions": (
        "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf",
        "IRS Form 1040 Instructions",
        "Instructions for Form 1040 and 1040-SR",
    ),
}

# Landing pages, so a citation can link somewhere a human can read.
_LANDING = {
    "IRS Publication 17": "https://www.irs.gov/forms-pubs/about-publication-17",
    "IRS Publication 501": "https://www.irs.gov/forms-pubs/about-publication-501",
    "IRS Publication 519": "https://www.irs.gov/forms-pubs/about-publication-519",
    "IRS Publication 970": "https://www.irs.gov/forms-pubs/about-publication-970",
    "IRS Form 1040 Instructions": "https://www.irs.gov/forms-pubs/about-form-1040",
}

# Page furniture that survives text extraction. Not content: embedded as-is it
# comes back as retrieved "evidence".
_NOISE = (
    re.compile(r"^the type and rule above prints on all proofs.*$", re.I | re.M),
    re.compile(r"^userid:.*$", re.I | re.M),
    re.compile(r"^draft as of.*$", re.I | re.M),
    re.compile(r"^page \d+ of \d+\s*$", re.I | re.M),
    re.compile(r"^\s*\d{1,4}\s*-\s*\d{1,2}-[a-z]{3}-\d{4}\s*$", re.I | re.M),
    re.compile(r"^\s*\d{1,4}\s*$", re.M),
)

# "perma-\nnently" -> "permanently". Narrow columns hyphenate constantly, and a
# split word matches no query.
_HYPHEN_BREAK = re.compile(r"([a-z])[ \t]*-[ \t]*\n[ \t]*([a-z])")
# A line ending in neither sentence punctuation nor a hyphen, followed by a
# lowercase letter, is a wrapped column line rather than a new paragraph.
_WRAPPED_LINE = re.compile(r"([^\n.:;?!\-])\n([a-z])")

# Publication headings: "Standard Deduction", "Who Must File", "Table 6."
_HEADING = re.compile(
    r"(?m)^(?:"
    r"(?:table \d+[a-z]?\.?.*)"
    r"|(?:chapter \d+\.?.*)"
    r"|(?:part \d+\.?.*)"
    r"|(?:[A-Z][A-Za-z',\- ]{3,60})"
    r")$"
)


def clean_pdf_text(text: str) -> str:
    """Strip page furniture and undo column wrapping."""
    for pattern in _NOISE:
        text = pattern.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    # Twice: rejoining one wrapped line can expose the next.
    for _ in range(2):
        text = _WRAPPED_LINE.sub(r"\1 \2", text)
    return text


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required to read IRS PDFs: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return clean_pdf_text("\n".join(p.extract_text() or "" for p in reader.pages))


def download(cfg: Config, session: PoliteSession, force: bool = False) -> Path:
    """Fetch the publications into data/raw/irs/. Cached, so re-runs are free."""
    dest = Path(cfg.get_path("paths.raw", "data/raw")) / "irs"
    dest.mkdir(parents=True, exist_ok=True)
    for stem, (url, _, _) in PUBLICATIONS.items():
        target = dest / f"{stem}.pdf"
        if target.exists() and not force:
            continue
        body = session.get_bytes(url) if hasattr(session, "get_bytes") else session.get(url).content
        target.write_bytes(body)
    return dest


def parse(cfg: Config, path: Path) -> list[Document]:
    """One Document per publication.

    Publications are split by heading at chunk time (STRATEGY["irs_pub"] ==
    "heading"), so the whole publication is kept together here and the chunker
    decides the boundaries.
    """
    docs: list[Document] = []
    for stem, (url, citation, title) in PUBLICATIONS.items():
        pdf = path / f"{stem}.pdf"
        if not pdf.exists():
            continue
        text = normalize_ws(_read_pdf(pdf))
        if not text.strip():
            print(f"[irs] WARNING: {pdf.name} produced no extractable text; skipping")
            continue
        docs.append(
            Document(
                source="irs_pub",
                doc_id=f"irs_pub:{stem}",
                citation=citation,
                title=title,
                text=text,
                url=_LANDING.get(citation, url),
                breadcrumb=["Internal Revenue Service", citation],
                extra={"pdf": url},
            )
        )
    return docs
