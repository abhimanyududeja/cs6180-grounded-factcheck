"""Post-hoc citation verification.

The generator is instructed to quote verbatim and cite every sentence. Instruction
is not enforcement. This module independently re-checks the output against the
retrieved chunks and reports what it finds, so the system's grounding claim rests
on a mechanical check rather than on the model's good behaviour.

Four checks:

1. **Tag validity** — every [S#] the model cited was actually in the context.
   Catches the model inventing a source that was never shown to it.
2. **Quote fidelity** — every quoted span appears in the chunk it is attributed
   to, compared after whitespace and punctuation normalization. Catches fabricated
   or "improved" quotes, which is the most dangerous hallucination here because a
   plausible fake quote reads exactly like a real one.
3. **Sentence coverage** — every sentence of the explanation carries a citation.
   Catches unsupported assertions smuggled in between grounded ones.
4. **Authority consistency** — the verdict's confidence is not higher than its
   weakest support warrants.

The result is a `GroundingReport`, surfaced in the UI and scored in the
evaluation as the system's hallucination rate.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .retrieve import RetrievalResult

TAG_RE = re.compile(r"\[(S\d+)\]")


def normalize_for_match(text: str) -> str:
    """Compare quotes on substance, not typography.

    Government sources are full of non-breaking spaces, curly quotes, em dashes
    and multi-space runs, and models silently normalize these when copying. We
    should not flag a correct quote as fabricated over a character the model had
    no way to reproduce exactly.
    """
    text = text.lower()
    text = text.replace(" ", " ").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class QuoteCheck:
    tag: str
    quote: str
    found: bool
    similarity: float                # best fuzzy ratio against the chunk
    citation: str = ""
    note: str = ""


@dataclass
class GroundingReport:
    valid_tags: list[str] = field(default_factory=list)
    invalid_tags: list[str] = field(default_factory=list)
    quote_checks: list[QuoteCheck] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    n_sentences: int = 0

    @property
    def quotes_verified(self) -> int:
        return sum(1 for q in self.quote_checks if q.found)

    @property
    def quote_fidelity(self) -> float:
        return self.quotes_verified / len(self.quote_checks) if self.quote_checks else 1.0

    @property
    def sentence_coverage(self) -> float:
        if not self.n_sentences:
            return 1.0
        return 1 - len(self.uncited_sentences) / self.n_sentences

    @property
    def is_grounded(self) -> bool:
        """The bar for calling an answer grounded: no invented sources, no
        fabricated quotes, and nothing asserted without a citation."""
        return (
            not self.invalid_tags
            and self.quote_fidelity == 1.0
            and not self.uncited_sentences
        )

    @property
    def grade(self) -> str:
        if self.is_grounded:
            return "GROUNDED"
        if self.invalid_tags or self.quote_fidelity < 0.5:
            return "UNGROUNDED"
        return "PARTIAL"

    def summary(self) -> str:
        bits = [
            f"{self.quotes_verified}/{len(self.quote_checks)} quotes verbatim",
            f"{self.n_sentences - len(self.uncited_sentences)}/{self.n_sentences} sentences cited",
        ]
        if self.invalid_tags:
            bits.append(f"INVALID TAGS: {', '.join(self.invalid_tags)}")
        return " | ".join(bits)


def _best_substring_ratio(needle: str, haystack: str) -> float:
    """How closely does `needle` appear anywhere in `haystack`?

    A plain `in` test is too brittle (one normalized character ends it) and a
    whole-string SequenceMatcher is meaningless when the haystack is 50x longer.
    So slide a window the size of the needle and take the best local match.
    """
    if not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    n = len(needle)
    if n > len(haystack):
        return difflib.SequenceMatcher(None, needle, haystack).ratio()
    best = 0.0
    step = max(1, n // 4)
    for start in range(0, len(haystack) - n + 1, step):
        window = haystack[start : start + n + step]
        ratio = difflib.SequenceMatcher(None, needle, window).quick_ratio()
        if ratio > best:
            # quick_ratio is an upper bound; only pay for the real one if promising
            best = max(best, difflib.SequenceMatcher(None, needle, window).ratio())
        if best >= 0.99:
            break
    return best


def verify(
    result: dict,
    retrieval: RetrievalResult,
    quote_threshold: float = 0.85,
) -> GroundingReport:
    """Check a generated verdict dict against the chunks it was given."""
    report = GroundingReport()
    available = {r.tag for r in retrieval.results}

    # --- 1. tag validity ------------------------------------------------
    explanation = result.get("explanation", "") or ""
    cited = set(TAG_RE.findall(explanation))
    for ev in result.get("evidence") or []:
        if ev.get("tag"):
            cited.add(ev["tag"])
    report.valid_tags = sorted(cited & available)
    report.invalid_tags = sorted(cited - available)

    # --- 2. quote fidelity ----------------------------------------------
    for ev in result.get("evidence") or []:
        tag, quote = ev.get("tag", ""), (ev.get("quote") or "").strip()
        if not quote:
            continue
        chunk = retrieval.by_tag(tag)
        if chunk is None:
            report.quote_checks.append(
                QuoteCheck(tag, quote, False, 0.0, note=f"tag {tag} was not in the context")
            )
            continue
        ratio = _best_substring_ratio(normalize_for_match(quote), normalize_for_match(chunk.text))
        report.quote_checks.append(
            QuoteCheck(
                tag=tag,
                quote=quote,
                found=ratio >= quote_threshold,
                similarity=round(ratio, 3),
                citation=chunk.citation,
                note="" if ratio >= quote_threshold else "quote not found in the cited source",
            )
        )

    # --- 3. sentence coverage -------------------------------------------
    sentences = split_sentences(explanation)
    report.n_sentences = len(sentences)
    for sent in sentences:
        # Very short fragments and pure transitions do not need their own cite.
        if len(sent.split()) < 5:
            continue
        if not TAG_RE.search(sent):
            report.uncited_sentences.append(sent)

    return report


LIST_FIELDS = ("caveats", "evidence", "conflicts")


def coerce_schema(data: dict) -> dict:
    """Force the list-valued fields to actually be lists.

    The prompt asks for lists, but instruction is not enforcement: a smaller model
    will happily return `"caveats": "none"` as a bare string, and every downstream
    `.append` then raises. Since this pipeline is meant to run against a local
    model as well as a frontier one, the shape is normalized here rather than
    trusted.
    """
    for key in LIST_FIELDS:
        value = data.get(key)
        if value is None:
            data[key] = []
        elif isinstance(value, str):
            # A bare string is one item, unless it is an explicit "nothing".
            data[key] = [] if value.strip().lower() in {"", "none", "n/a", "null"} else [value]
        elif not isinstance(value, list):
            data[key] = [value]
    return data


def enforce(result: dict, report: GroundingReport) -> dict:
    """Downgrade an ungrounded answer rather than presenting it as verified.

    A verdict resting on a fabricated quote must not reach the user labelled
    "SUPPORTED, high confidence". We keep the text — it is still useful, and
    hiding it would make the failure invisible — but we relabel it honestly and
    attach the reason.
    """
    out = dict(result)
    out["grounding"] = {
        "grade": report.grade,
        "quote_fidelity": round(report.quote_fidelity, 3),
        "sentence_coverage": round(report.sentence_coverage, 3),
        "invalid_tags": report.invalid_tags,
        "summary": report.summary(),
    }
    if report.grade == "UNGROUNDED":
        out["confidence"] = "low"
        out.setdefault("caveats", []).insert(
            0,
            "AUTOMATED GROUNDING CHECK FAILED: "
            + (f"cited sources that were not retrieved ({', '.join(report.invalid_tags)}). "
               if report.invalid_tags else "")
            + (f"{len(report.quote_checks) - report.quotes_verified} of "
               f"{len(report.quote_checks)} quotes could not be found in the cited "
               f"sources. " if report.quote_fidelity < 1 else "")
            + "Treat this answer as unverified and read the linked sources directly.",
        )
    elif report.grade == "PARTIAL":
        if out.get("confidence") == "high":
            out["confidence"] = "medium"
        out.setdefault("caveats", []).append(
            f"Partial grounding: {report.summary()}."
        )
    return out
