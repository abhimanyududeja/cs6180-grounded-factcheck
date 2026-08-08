"""Structure-aware chunking.

Naive fixed-window chunking is the single biggest failure mode for legal RAG,
and this corpus makes the reason concrete: **8 CFR § 214.2 is one section of
700,000 characters** covering every nonimmigrant category from A-1 diplomats to
Q cultural exchange. Split it on a 2,000-character window and you get ~350
chunks that all cite "8 CFR § 214.2" — so the system tells a student on an F-1
visa that its answer comes from a section that also contains the rules for
treaty investors and crewmen, with no way to tell which part it actually used.

Instead we run a stack machine over the statutory numbering — (a), (1), (i),
(A) — to recover each paragraph's position in the hierarchy. A chunk then cites
`8 CFR § 214.2(f)(9)(ii)` and links to the exact subsection. Retrieval improves
because the subsection path is prepended to the embedded text; verification
improves because a human can click through to precisely the provision quoted.

Three strategies, chosen per source:
  * `statutory`  — CFR and INA: stack machine over subsection markers.
  * `heading`    — FAM and USCIS PM: split at document headings.
  * `atomic`     — Federal Register: short abstracts, never split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Chunk, Document, make_chunk

# Roughly 4 characters per token for English legal prose. Used only to size
# chunks, so an approximation is fine and avoids a tokenizer dependency that
# would have to track model-specific encodings.
CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Statutory numbering
# ---------------------------------------------------------------------------

MARKER_RE = re.compile(r"^\s*\(([A-Za-z]{1,4}|\d{1,3})\)\s")

_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
          "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def _kinds(token: str) -> list[str]:
    """Which numbering sequences could this marker belong to? '(i)' is both the
    9th lowercase letter and the 1st lowercase roman numeral, so ambiguity is
    resolved later against what is already on the stack."""
    out: list[str] = []
    if token.isdigit():
        return ["num"]
    if token.islower():
        if len(token) == 1 and token.isalpha():
            out.append("alpha")
        if token in _ROMAN:
            out.append("roman")
    else:
        if len(token) == 1 and token.isalpha():
            out.append("ALPHA")
        if token.lower() in _ROMAN:
            out.append("ROMAN")
    return out


def _successor(kind: str, prev: str) -> str | None:
    """The marker that legitimately follows `prev` in sequence `kind`."""
    if kind == "num":
        return str(int(prev) + 1)
    if kind in ("alpha", "ALPHA"):
        base = prev.lower()
        if base == "z":
            return None
        nxt = chr(ord(base) + 1)
        return nxt if kind == "alpha" else nxt.upper()
    seq = _ROMAN
    low = prev.lower()
    if low in seq and seq.index(low) + 1 < len(seq):
        nxt = seq[seq.index(low) + 1]
        return nxt if kind == "roman" else nxt.upper()
    return None


def _first(kind: str) -> str:
    return {"num": "1", "alpha": "a", "ALPHA": "A", "roman": "i", "ROMAN": "I"}[kind]


@dataclass
class _Level:
    kind: str
    value: str


class NumberingStack:
    """Tracks the current subsection path across a document.

    Given the marker sequence (a) (1) (2) (i) (ii) (b), it yields paths
    (a), (a)(1), (a)(2), (a)(2)(i), (a)(2)(ii), (b) — which is exactly how a
    lawyer would cite each paragraph.
    """

    def __init__(self) -> None:
        self.stack: list[_Level] = []

    def push_marker(self, token: str) -> None:
        """Rule order matters, and getting it wrong is subtle.

        The hard case is '(i)', which is both the 9th letter and the 1st roman
        numeral. Inside 8 CFR 214.2 the H-1B rules live at (h), so when the
        parser is at (h)(4) and meets '(i)', it must read it as opening
        (h)(4)(i) — *not* as (h)'s successor (i), which would silently move
        every remaining H-1B paragraph under the media-representative
        subsection. Checking 'opens a deeper level' before 'continues a
        shallower level' is what prevents that.
        """
        kinds = _kinds(token)
        if not kinds:
            return
        top = self.stack[-1] if self.stack else None
        open_kinds = [k for k in kinds if _first(k) == token]

        # 1. Continues the innermost level: (a)(1) -> (a)(2).
        if top and top.kind in kinds and _successor(top.kind, top.value) == token:
            self.stack[-1] = _Level(top.kind, token)
            return

        # 2. Opens a new, deeper level: (a)(1) + '(i)' -> (a)(1)(i).
        #    A kind is never nested directly inside itself — (a)(a) and (1)(1)
        #    do not occur — so a repeat of the enclosing kind means the drafter
        #    restarted that level. (The same kind may reappear further down;
        #    (a)(1)(i)(A)(1) is normal, which is why only the immediately
        #    enclosing level is checked.)
        if open_kinds:
            if top and top.kind in open_kinds:
                self.stack[-1] = _Level(top.kind, token)
                return
            in_use = {lvl.kind for lvl in self.stack}
            k = next((k for k in open_kinds if k not in in_use), open_kinds[0])
            self.stack.append(_Level(k, token))
            return

        # 3. Continues a shallower level: (a)(1)(i) + '(2)' -> (a)(2).
        for depth in range(len(self.stack) - 2, -1, -1):
            lvl = self.stack[depth]
            if lvl.kind in kinds and _successor(lvl.kind, lvl.value) == token:
                self.stack = self.stack[:depth] + [_Level(lvl.kind, token)]
                return

        # 4. Out of sequence — common after tables, notes and [Reserved]
        #    paragraphs. Replace the deepest level of a compatible kind.
        for depth in range(len(self.stack) - 1, -1, -1):
            kind = self.stack[depth].kind
            if kind in kinds:
                self.stack = self.stack[:depth] + [_Level(kind, token)]
                return
        self.stack.append(_Level(kinds[0], token))

    @property
    def path(self) -> str:
        return "".join(f"({lvl.value})" for lvl in self.stack)

    def snapshot(self) -> list[str]:
        return [lvl.value for lvl in self.stack]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

@dataclass
class Block:
    text: str
    path: str          # e.g. "(f)(9)(ii)"
    path_parts: list[str]


# The CFR routinely opens a child subsection mid-paragraph, joined by an em
# dash: "(a) Foreign government officials—(1) General. The determination by...".
# Matching only at line starts would collapse the entire F-1 regime into a
# single "(f)" path, which is the difference between citing "8 CFR 214.2(f)"
# (60,000 characters) and "8 CFR 214.2(f)(9)(ii)" (the actual answer).
INLINE_MARKER_RE = re.compile(r"[—–]{1,2}\(([A-Za-z]{1,4}|\d{1,3})\)\s")


def _segment_line(line: str) -> list[tuple[str | None, str]]:
    """Split one line into (marker, text) segments at structural markers.

    Only line-initial markers and markers introduced by an em dash count.
    Bare parentheses mid-sentence are cross-references — "section 101(a)(15)(F)"
    or "(see paragraph (b)(2))" — and must not be treated as structure.
    """
    segments: list[tuple[str | None, str]] = []
    rest = line
    lead: str | None = None

    m = MARKER_RE.match(rest)
    if m:
        lead = m.group(1)
        rest = rest[m.end():]

    pos = 0
    while True:
        m2 = INLINE_MARKER_RE.search(rest, pos)
        if not m2:
            break
        segments.append((lead, rest[:m2.start()]))
        lead = m2.group(1)
        rest = rest[m2.end():]
        pos = 0
    segments.append((lead, rest))
    return segments


def _statutory_blocks(text: str) -> list[Block]:
    """Split into paragraphs and label each with its subsection path."""
    stack = NumberingStack()
    blocks: list[Block] = []
    buf: list[str] = []
    cur_path, cur_parts = "", []

    def flush() -> None:
        if buf and "".join(buf).strip():
            blocks.append(Block("\n".join(buf).strip(), cur_path, cur_parts))

    for line in text.split("\n"):
        if not line.strip():
            buf.append("")
            continue
        for marker, body in _segment_line(line):
            if marker is not None:
                flush()
                buf = []
                stack.push_marker(marker)
                cur_path, cur_parts = stack.path, stack.snapshot()
                buf.append(f"({marker}) {body}".rstrip())
            else:
                buf.append(body)

    flush()
    return blocks


def _heading_blocks(text: str) -> list[Block]:
    """FAM and Policy Manual pages use short title-case or ALL-CAPS lines and
    lettered/numbered headings ('A. Eligibility', '9 FAM 402.1-2') as section
    breaks. Group paragraphs under the nearest preceding heading."""
    # Deliberately conservative. An earlier version also treated any Title Case
    # line under 100 characters as a heading, which matched ordinary short
    # sentences and shattered 9 FAM into 7,500 fragments averaging 600
    # characters — too small to answer anything on their own.
    heading_re = re.compile(
        r"^\s*(?:"
        r"\d+\s?FAM\s[\d.\-()A-Z]+.*"          # 9 FAM 402.1-1(A) Statutory Authorities
        r"|[A-Z]\.\s+[A-Z]\S.{0,80}"           # A. Eligibility Requirements
        r"|\d{1,2}\.\s+[A-Z]\S.{0,80}"         # 1. Filing
        r"|Chapter\s+\d+\s*[-–].{0,80}"        # Chapter 2 - Eligibility
        r"|[A-Z][A-Z0-9 ,'\-/&()]{6,80}"       # ALL CAPS SECTION HEADING
        r")\s*$"
    )
    blocks: list[Block] = []
    buf: list[str] = []
    heading = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append(Block(body, heading, [heading] if heading else []))

    for line in text.split("\n"):
        s = line.strip()
        if s and heading_re.match(s) and len(s) < 100:
            flush()
            buf = [s]
            heading = s
        else:
            buf.append(line)
    flush()
    return blocks or [Block(text, "", [])]


def _split_oversized(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Last resort for a single paragraph bigger than the target: split on
    sentence boundaries with overlap so no sentence is orphaned."""
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text)
    out, cur = [], ""
    for sent in sentences:
        if cur and len(cur) + len(sent) > target_chars:
            out.append(cur.strip())
            tail = cur[-overlap_chars:] if overlap_chars else ""
            cur = tail + " " + sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRATEGY = {
    "cfr": "statutory", "ina": "statutory",
    "fam": "heading", "uscis_pm": "heading",
    "fedreg": "atomic",
    # IRC is drafted like the immigration statute, so the numbering stack applies.
    # IRS publications are prose explainers organised by heading, not by subsection.
    "irc": "statutory", "irs_pub": "heading",
}


def chunk_document(doc: Document, cfg, authority_rank: int) -> list[Chunk]:
    target = cfg.get_path("chunking.target_tokens", 500) * CHARS_PER_TOKEN
    overlap = cfg.get_path("chunking.overlap_tokens", 80) * CHARS_PER_TOKEN
    min_chars = cfg.get_path("chunking.min_tokens", 40) * CHARS_PER_TOKEN
    strategy = STRATEGY.get(doc.source, "heading")

    if strategy == "atomic":
        return [make_chunk(doc, doc.text, 0, authority_rank)] if len(doc.text) >= min_chars else []

    blocks = (_statutory_blocks if strategy == "statutory" else _heading_blocks)(doc.text)

    # Pack consecutive blocks into chunks, breaking whenever the top-level
    # subsection changes. That boundary is what keeps F-1 rules (f) from
    # bleeding into H-1B rules (h) inside one chunk.
    chunks: list[Chunk] = []
    buf: list[Block] = []
    seq = 0

    def emit(group: list[Block]) -> None:
        nonlocal seq
        if not group:
            return
        body = "\n\n".join(b.text for b in group).strip()
        if len(body) < min_chars:
            return
        # Cite the deepest path shared by every block in the chunk.
        paths = [b.path_parts for b in group if b.path_parts]
        common: list[str] = []
        if paths:
            for parts in zip(*paths):
                if len(set(parts)) == 1:
                    common.append(parts[0])
                else:
                    break
        pieces = [body] if len(body) <= target * 1.5 else _split_oversized(body, target, overlap)
        for piece in pieces:
            sub = _render_path(doc.source, common)
            c = make_chunk(doc, _decorate(doc, sub, piece), seq, authority_rank)
            if sub:
                c.citation = _cite_with_subsection(doc, sub)
                c.url = _subsection_url(doc, common)
                c.breadcrumb = doc.breadcrumb + [sub]
            chunks.append(c)
            seq += 1

    def top_of(b: Block) -> str:
        return b.path_parts[0] if b.path_parts else ""

    cur_top = None
    for b in blocks:
        oversized = sum(len(x.text) for x in buf) + len(b.text) > target
        # For statute and regulation text, a change of top-level subsection is a
        # hard boundary: F-1 rules at (f) must never share a chunk with H-1B
        # rules at (h), because the resulting chunk could not be honestly cited.
        # Headings in FAM and Policy Manual pages carry no such legal weight, so
        # there we pack purely by size — otherwise every heading would start a
        # new chunk and the corpus fragments into unretrievable slivers.
        crosses_boundary = strategy == "statutory" and top_of(b) != cur_top
        if buf and (crosses_boundary or oversized):
            emit(buf)
            buf = []
        if not buf:
            cur_top = top_of(b)
        buf.append(b)
    emit(buf)
    return chunks


def _cite_with_subsection(doc: Document, sub: str) -> str:
    """Attach a subsection path to a citation.

    Statute citations are dual-form — "INA § 214 (8 U.S.C. § 1184)" — so naive
    concatenation produces "INA § 214 (8 U.S.C. § 1184)(g)", which reads as if
    the subsection belonged to the parenthetical. The subsection has to go into
    both halves: "INA § 214(g) (8 U.S.C. § 1184(g))".
    """
    if doc.source not in ("cfr", "ina") or not sub:
        return doc.citation
    m = re.match(r"^(.*?)\s*\((8 U\.S\.C\..*?)\)$", doc.citation)
    if m:
        return f"{m.group(1)}{sub} ({m.group(2)}{sub})"
    return f"{doc.citation}{sub}"


def _render_path(source: str, parts: list[str]) -> str:
    if not parts:
        return ""
    if source in ("cfr", "ina"):
        return "".join(f"({p})" for p in parts)
    return parts[-1]


def _subsection_url(doc: Document, parts: list[str]) -> str:
    """eCFR supports deep links to a paragraph: .../section-214.2#p-214.2(f)(9)"""
    if doc.source == "cfr" and parts:
        num = doc.citation.split("§")[-1].strip()
        path = "".join(f"({p})" for p in parts)
        return f"{doc.url}#p-{num}{path}"
    return doc.url


def _decorate(doc: Document, subsection: str, body: str) -> str:
    """Prepend a compact context header to the embedded text.

    This matters more than it looks. The body of 8 CFR 214.2(f)(9) never says
    the words "student" or "F-1" — it says "the student" and relies on the
    reader knowing which subsection they are in. Without this header the chunk
    is nearly unretrievable for the query it should answer.
    """
    head = f"{doc.citation}{subsection} — {doc.title}"
    crumb = " > ".join(doc.breadcrumb[-2:]) if doc.breadcrumb else ""
    if crumb:
        head += f" [{crumb}]"
    return f"{head}\n\n{body}"


def chunk_corpus(docs: list[Document], cfg) -> list[Chunk]:
    ranks = cfg.get_path("project.authority_rank", {})
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, cfg, ranks.get(d.source, 5)))
    return out
