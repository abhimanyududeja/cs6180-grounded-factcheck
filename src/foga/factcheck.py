"""The fact-checking pipeline.

    claim
      -> decompose into atomic sub-claims
      -> retrieve independently for each
      -> generate a grounded verdict per sub-claim
      -> verify citations and quotes mechanically
      -> synthesize one overall assessment
      -> attach a currency warning from the Federal Register

Decomposition matters more than it might seem. Real questions bundle several
claims — "F-1 students get 12 months of OPT and STEM majors get 3 more years"
contains one true claim and one false one (the STEM extension is 24 months, not
36). Checked as a single unit, a model tends to return one blurred verdict.
Checked separately, the false half is isolated and named.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field

from .config import load_config
from .index import HybridIndex
from .llm import LLM, estimate_cost
from .prompts import (
    DECOMPOSE_SCHEMA,
    FACTCHECK_SCHEMA,
    NO_RETRIEVAL_USER,
    ABSTAIN_TEMPLATE,
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    FACTCHECK_SYSTEM,
    FACTCHECK_USER,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_USER,
)
from .retrieve import RetrievalResult, Retriever
from .verify import GroundingReport, coerce_schema, enforce, verify

VERDICTS = ("SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED", "NOT_ADDRESSED")


@dataclass
class SubClaimResult:
    id: int
    text: str
    verdict: str
    confidence: str
    explanation: str
    evidence: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    report: GroundingReport | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("retrieval", "report")}
        return d


@dataclass
class FactCheckResult:
    claim: str
    verdict: str
    confidence: str
    summary: str
    explanation: str
    caveats: list[str] = field(default_factory=list)
    subclaims: list[SubClaimResult] = field(default_factory=list)
    currency_warnings: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "explanation": self.explanation,
            "caveats": self.caveats,
            "subclaims": [s.to_dict() for s in self.subclaims],
            "currency_warnings": self.currency_warnings,
            "stats": self.stats,
        }

    def to_markdown(self) -> str:
        icon = {"SUPPORTED": "[SUPPORTED]", "CONTRADICTED": "[CONTRADICTED]",
                "PARTIALLY_SUPPORTED": "[PARTIALLY SUPPORTED]",
                "NOT_ADDRESSED": "[NOT ADDRESSED]"}.get(self.verdict, self.verdict)
        lines = [
            f"# {icon}  (confidence: {self.confidence})",
            "",
            f"**Claim:** {self.claim}",
            "",
            f"**Bottom line:** {self.summary}",
            "",
            self.explanation,
            "",
        ]
        if self.caveats:
            lines += ["## Caveats", ""] + [f"- {c}" for c in self.caveats] + [""]
        if self.currency_warnings:
            lines += ["## Recent regulatory activity", ""]
            lines += [f"- {w['date']} — [{w['title']}]({w['url']})"
                      for w in self.currency_warnings] + [""]
        lines += ["## Sub-claims", ""]
        for s in self.subclaims:
            lines.append(f"### {s.id}. {s.text}")
            lines.append(f"**{s.verdict}** ({s.confidence}) — "
                         f"grounding: {s.grounding.get('grade', 'n/a')}")
            lines.append("")
            lines.append(s.explanation)
            lines.append("")
            if s.sources:
                lines.append("| Tag | Authority | Source | Link |")
                lines.append("|---|---|---|---|")
                for src in s.sources:
                    lines.append(f"| {src['tag']} | {src['citation']} | "
                                 f"{src['source']} | {src['url']} |")
                lines.append("")
        return "\n".join(lines)


def _usable_subclaims(subs: list[dict], claim: str) -> list[dict]:
    """Reject a decomposition that shredded the claim into fragments.

    A smaller model will sometimes split on clause boundaries and return pieces
    like "An F-1 student may work" — grammatical, but not a checkable assertion.
    Retrieval on a fragment returns nothing on point and the system then abstains,
    which looks exactly like a retrieval failure and is not one.

    Rather than trust the split, require every piece to look like a claim and to
    be a real reduction of the original. If any piece fails, the whole
    decomposition is discarded and the claim is checked as one unit, which is
    always safe.
    """
    if not subs:
        return []
    for s in subs:
        text = (s.get("text") or "").strip()
        # Fragments are short and, unlike a real sub-claim, carry no assertion.
        if len(text.split()) < 5 or len(text) < 25:
            return []
        # A "sub"-claim longer than the original is a restatement, not a split.
        if len(text) > len(claim) + 40:
            return []
    return subs


class FactChecker:
    def __init__(self, index: HybridIndex, cfg=None, llm: LLM | None = None,
                 retriever: Retriever | None = None):
        self.cfg = cfg or load_config()
        self.index = index
        self.llm = llm or LLM()
        self.retriever = retriever or Retriever(index, self.cfg)
        self._fedreg = None

    # -- stage 1: decomposition ------------------------------------------

    def decompose(self, claim: str) -> list[dict]:
        if not self.cfg.get_path("factcheck.decompose", True):
            return [{"id": 1, "text": claim, "type": "assertion"}]
        max_sub = self.cfg.get_path("factcheck.max_subclaims", 6)
        try:
            resp = self.llm.chat(
                DECOMPOSE_SYSTEM,
                DECOMPOSE_USER.format(claim=claim, max_subclaims=max_sub),
                json_mode=True,
                schema=DECOMPOSE_SCHEMA,
            )
            subs = resp.json().get("subclaims") or []
            subs = [s for s in subs if s.get("text")][:max_sub]
            subs = _usable_subclaims(subs, claim)
            for i, s in enumerate(subs, 1):
                s["id"] = i
            return subs or [{"id": 1, "text": claim, "type": "assertion"}]
        except Exception as exc:
            # Decomposition is an optimization, not a requirement. If it fails,
            # check the claim whole rather than losing the whole request.
            print(f"  [warn] decomposition failed ({exc}); checking claim as one unit")
            return [{"id": 1, "text": claim, "type": "assertion"}]

    # -- stage 2+3: retrieve and judge one sub-claim ----------------------

    def _check_subclaim_no_retrieval(self, sub: dict) -> SubClaimResult:
        """The no-retrieval control the proposal asks for.

        Same model, same output schema, no evidence. Everything downstream that
        depends on retrieved chunks is therefore vacuous here: there is nothing to
        cite, so the mechanical verifier has nothing to check. That is the point.
        Grounding is recorded as UNGROUNDED whenever this baseline commits to a
        decisive verdict, because a decisive verdict with no evidence behind it is
        exactly the failure the system is built to prevent.
        """
        resp = self.llm.chat(
            FACTCHECK_SYSTEM,
            NO_RETRIEVAL_USER.format(claim=sub["text"]),
            json_mode=True,
            schema=FACTCHECK_SCHEMA,
        )
        try:
            data = resp.json()
        except ValueError:
            data = dict(ABSTAIN_TEMPLATE)
            data["caveats"] = list(data["caveats"]) + [
                "The model did not return parseable output for this sub-claim."
            ]
        if data.get("verdict") not in VERDICTS:
            data["verdict"] = "NOT_ADDRESSED"
        data = coerce_schema(data)

        decisive = data.get("verdict") in ("SUPPORTED", "CONTRADICTED")
        data["evidence"] = []
        data["grounding"] = {
            "grade": "UNGROUNDED" if decisive else "GROUNDED",
            "valid_tags": [],
            "quote_fidelity": 0.0 if decisive else None,
            "sentence_coverage": 0.0 if decisive else None,
            "issues": (["Decisive verdict asserted with no retrieved evidence."]
                       if decisive else []),
        }

        return SubClaimResult(
            id=sub["id"],
            text=sub["text"],
            verdict=data.get("verdict", "NOT_ADDRESSED"),
            confidence=data.get("confidence", "low"),
            explanation=data.get("explanation", ""),
            evidence=[],
            caveats=data.get("caveats") or [],
            conflicts=data.get("conflicts") or [],
            grounding=data["grounding"],
            sources=[],
        )

    def check_subclaim(self, sub: dict, sources: list[str] | None = None) -> SubClaimResult:
        if not self.cfg.get_path("retrieval.enabled", True):
            return self._check_subclaim_no_retrieval(sub)

        retrieval = self.retriever.search(sub["text"], sources=sources)

        if retrieval.abstain or not retrieval.results:
            data = dict(ABSTAIN_TEMPLATE)
            data["caveats"] = list(data["caveats"]) + [retrieval.abstain_reason]
            report = verify(data, retrieval)
        else:
            resp = self.llm.chat(
                FACTCHECK_SYSTEM,
                FACTCHECK_USER.format(claim=sub["text"], context=retrieval.context_block()),
                json_mode=True,
                schema=FACTCHECK_SCHEMA,
            )
            try:
                data = resp.json()
            except ValueError:
                data = dict(ABSTAIN_TEMPLATE)
                data["caveats"] = list(data["caveats"]) + [
                    "The model did not return parseable output for this sub-claim."
                ]
            if data.get("verdict") not in VERDICTS:
                data["verdict"] = "NOT_ADDRESSED"
            report = verify(data, retrieval)
            if self.cfg.get_path("factcheck.verify_citations", True):
                data = enforce(data, report)

        return SubClaimResult(
            id=sub["id"],
            text=sub["text"],
            verdict=data.get("verdict", "NOT_ADDRESSED"),
            confidence=data.get("confidence", "low"),
            explanation=data.get("explanation", ""),
            evidence=data.get("evidence") or [],
            caveats=data.get("caveats") or [],
            conflicts=data.get("conflicts") or [],
            grounding=data.get("grounding") or {},
            sources=[
                {
                    "tag": r.tag,
                    "citation": r.chunk.citation,
                    "title": r.chunk.title,
                    "source": r.chunk.source,
                    "url": r.chunk.url,
                    "found_by": r.found_by,
                    "rerank_score": r.rerank_score,
                    "text": r.chunk.text,
                }
                for r in retrieval.results
            ],
            retrieval=retrieval,
            report=report,
        )

    # -- stage 4: synthesis ----------------------------------------------

    def synthesize(self, claim: str, subs: list[SubClaimResult]) -> dict:
        if len(subs) == 1:
            s = subs[0]
            return {
                "verdict": s.verdict,
                "confidence": s.confidence,
                "summary": s.explanation.split(". ")[0].strip(". ") + ".",
                "explanation": s.explanation,
                "caveats": s.caveats,
            }
        payload = json.dumps(
            [{"id": s.id, "claim": s.text, "verdict": s.verdict,
              "confidence": s.confidence, "explanation": s.explanation,
              "caveats": s.caveats} for s in subs],
            indent=1,
        )
        try:
            resp = self.llm.chat(
                SYNTHESIZE_SYSTEM,
                SYNTHESIZE_USER.format(claim=claim, subverdicts=payload),
                json_mode=True,
            )
            return resp.json()
        except Exception:
            return {"verdict": _aggregate(subs), "confidence": "low",
                    "summary": "See sub-claim verdicts below.",
                    "explanation": " ".join(s.explanation for s in subs),
                    "caveats": [c for s in subs for c in s.caveats]}

    # -- stage 5: currency ------------------------------------------------

    def currency_check(self, claim: str, days: int = 365) -> list[dict]:
        """Surface recent Federal Register activity touching the same subject.

        The corpus is a snapshot. If a final rule published last month changed
        the very provision we just cited, the user needs to know before acting,
        even though our answer accurately reflects the text we indexed.
        """
        hits = self.retriever.search(claim, final_k=4, sources=["fedreg"])
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        out: list[dict] = []
        for r in hits.results:
            eff = r.chunk.effective_date or ""
            if eff and eff >= cutoff:
                out.append({
                    "date": eff,
                    "title": r.chunk.title,
                    "citation": r.chunk.citation,
                    "url": r.chunk.url,
                })
        return out[:3]

    # -- orchestration ----------------------------------------------------

    def check(
        self,
        claim: str,
        sources: list[str] | None = None,
        verbose: bool = False,
        currency: bool = True,
    ) -> FactCheckResult:
        t0 = time.time()
        subs_spec = self.decompose(claim)
        if verbose:
            print(f"  decomposed into {len(subs_spec)} sub-claim(s)")

        results: list[SubClaimResult] = []
        for spec in subs_spec:
            if verbose:
                print(f"  [{spec['id']}] {spec['text'][:80]}")
            res = self.check_subclaim(spec, sources=sources)
            if verbose:
                print(f"      -> {res.verdict} ({res.confidence}), "
                      f"grounding {res.grounding.get('grade', 'n/a')}")
            results.append(res)

        overall = self.synthesize(claim, results)
        warnings = self.currency_check(claim) if currency else []

        return FactCheckResult(
            claim=claim,
            verdict=overall.get("verdict") or _aggregate(results),
            confidence=overall.get("confidence", "low"),
            summary=overall.get("summary", ""),
            explanation=overall.get("explanation", ""),
            caveats=overall.get("caveats") or [],
            subclaims=results,
            currency_warnings=warnings,
            stats={
                "elapsed_s": round(time.time() - t0, 1),
                "llm_calls": self.llm.usage.calls,
                "input_tokens": self.llm.usage.input_tokens,
                "output_tokens": self.llm.usage.output_tokens,
                "est_cost_usd": round(estimate_cost(self.llm.usage, self.llm.model), 5),
                "model": self.llm.model,
                "n_subclaims": len(results),
                "grounded": sum(1 for r in results
                                if r.grounding.get("grade") == "GROUNDED"),
            },
        )


def _aggregate(subs: list[SubClaimResult]) -> str:
    """Deterministic fallback when the synthesis call is unavailable."""
    verdicts = [s.verdict for s in subs]
    if any(v == "CONTRADICTED" for v in verdicts):
        return "CONTRADICTED" if all(
            v in ("CONTRADICTED", "NOT_ADDRESSED") for v in verdicts
        ) else "PARTIALLY_SUPPORTED"
    if all(v == "SUPPORTED" for v in verdicts):
        return "SUPPORTED"
    if all(v == "NOT_ADDRESSED" for v in verdicts):
        return "NOT_ADDRESSED"
    return "PARTIALLY_SUPPORTED"
