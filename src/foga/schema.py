"""Canonical document + chunk schema.

Everything downstream — retrieval, generation, citation verification, evaluation —
speaks in `Chunk` objects. Each source adapter's only job is to turn its native
format (eCFR XML, USLM XML, FAM HTML, USCIS HTML, Federal Register JSON) into
a list of `Document`s with this shape.

The non-negotiable invariant: **every Chunk carries a `citation` and a `url`
that a human can independently verify.** A chunk without both is dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

SourceId = Literal["ina", "cfr", "fam", "uscis_pm", "fedreg",
                   "irc", "irs_pub"]


def _slug(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def normalize_ws(text: str) -> str:
    """Collapse whitespace but preserve paragraph structure."""
    text = text.replace(" ", " ").replace("’", "'").replace("“", '"')
    text = text.replace("”", '"').replace("—", "—")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Document:
    """One logical unit of law before chunking — a CFR section, a USC section,
    a FAM subsection, a Policy Manual chapter, a Federal Register document."""

    source: SourceId
    doc_id: str                 # stable, source-scoped, e.g. "cfr:8-214.2"
    citation: str               # e.g. "8 CFR 214.2"
    title: str                  # human heading
    text: str
    url: str
    breadcrumb: list[str] = field(default_factory=list)
    as_of: str | None = None    # date the text was retrieved / is current to
    effective_date: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = normalize_ws(self.text)
        self.title = normalize_ws(self.title)


@dataclass
class Chunk:
    """A retrievable, citable span. `chunk_id` is deterministic so rebuilding the
    index does not invalidate a saved gold-set label."""

    chunk_id: str
    source: SourceId
    doc_id: str
    citation: str
    title: str
    text: str
    url: str
    breadcrumb: list[str]
    seq: int                    # position within the parent document
    n_chars: int
    authority_rank: int
    as_of: str | None = None
    effective_date: str | None = None

    @property
    def header(self) -> str:
        """The line shown above the chunk text in a prompt and in the UI."""
        crumb = " > ".join(self.breadcrumb[-2:]) if self.breadcrumb else ""
        return f"{self.citation} — {self.title}" + (f" ({crumb})" if crumb else "")

    def to_prompt_block(self, tag: str) -> str:
        """Render for the generator. `tag` is the short handle the model must
        cite, e.g. [S3]."""
        return (
            f"[{tag}] {self.citation} | {self.title}\n"
            f"source={self.source} url={self.url}\n"
            f"---\n{self.text}\n"
        )


def make_chunk(doc: Document, text: str, seq: int, authority_rank: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{seq:04d}:{_slug(text)}",
        source=doc.source,
        doc_id=doc.doc_id,
        citation=doc.citation,
        title=doc.title,
        text=text,
        url=doc.url,
        breadcrumb=doc.breadcrumb,
        seq=seq,
        n_chars=len(text),
        authority_rank=authority_rank,
        as_of=doc.as_of,
        effective_date=doc.effective_date,
    )


# --------------------------------------------------------------------------
# JSONL persistence — the corpus lives on disk as newline-delimited JSON so it
# stays greppable and diffable.
# --------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r,
                                ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_documents(path: Path) -> list[Document]:
    return [Document(**r) for r in read_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**r) for r in read_jsonl(path)]
