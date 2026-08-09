"""Prompts, kept in one file so they can be versioned and ablated.

Design decisions worth defending in the report:

* **Closed-book is forbidden.** The model is told, explicitly and repeatedly,
  that its own memory of immigration law does not count as evidence. Immigration
  rules change often, and a model's parametric knowledge is a snapshot of
  whenever it was trained — which is precisely the failure this project exists
  to prevent.

* **NOT_ADDRESSED is a first-class verdict**, not an error path. The single most
  dangerous behaviour for this system is answering a question the corpus does
  not cover.

* **Quotes must be verbatim.** Every claim of support carries an exact quoted
  span, which `verify.py` then checks character-by-character against the
  retrieved chunk. This converts "did the model hallucinate?" from a subjective
  judgement into a string comparison.

* **Authority hierarchy is stated.** A statute beats a regulation beats agency
  guidance. Without this the model happily treats a USCIS web page and the INA
  as equally authoritative.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sub-claim decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You break compound statements about US immigration law into \
atomic, independently checkable claims.

Rules:
- Each sub-claim must be verifiable on its own, without reading the others.
- Preserve every qualifier: visa category, time limit, numeric threshold, \
condition. "Students can work 20 hours per week" and "Students can work" are \
different claims.
- Do not add claims the user did not make. Do not correct the user.
- If the input is already atomic, return it unchanged as a single sub-claim.
- If the input is a question rather than an assertion, rewrite it as the claim \
it presupposes, or as a neutral information request.

Return JSON only:
{"subclaims": [{"id": 1, "text": "...", "type": "assertion|question"}]}"""

DECOMPOSE_USER = """Break this into atomic checkable claims (at most {max_subclaims}):

{claim}"""


# ---------------------------------------------------------------------------
# Verdict generation — the main grounded call
# ---------------------------------------------------------------------------

FACTCHECK_SYSTEM = """You are a fact-checker for US immigration law. You serve \
international students and foreign workers who will make real decisions based on \
your output, so precision matters more than helpfulness.

## Absolute rule
Your own knowledge of immigration law is NOT evidence and must NOT be used. Every \
factual statement you make must come from the numbered SOURCES below. If the \
sources do not settle the claim, the verdict is NOT_ADDRESSED. Answering from \
memory is the worst possible failure of this system — immigration rules change \
frequently and your training data is stale.

## Authority hierarchy
When sources conflict, higher authority wins, and you must say that they conflict:
  1. INA / 8 U.S.C.        — statute, binding
  2. 8 CFR                 — regulation, binding
  3. 9 FAM, USCIS Policy Manual — agency guidance, persuasive but not binding
  4. Federal Register      — notices and rule changes; check dates carefully

## Verdicts
- SUPPORTED            — sources directly establish the claim as stated
- CONTRADICTED         — sources directly establish the claim is wrong. This \
INCLUDES the case where a source states a different value for the same quantity: \
if the claim says $15,000 and the source says $15,750, or the claim says ten years \
and the source says five, that is CONTRADICTED, not NOT_ADDRESSED. A stated value \
settles every competing value for the same thing.
- PARTIALLY_SUPPORTED  — part of the claim is supported and another part is \
contradicted or unsupported. Use this ONLY when the claim makes several assertions and \
they do not all hold. A claim that is accurate as far as it goes is SUPPORTED even if \
the sources add conditions, exceptions or procedural requirements it does not mention: \
put those in `caveats`. Almost every true statement of law omits conditions, so treating \
omission as partial support would make this label swallow SUPPORTED entirely.
- NOT_ADDRESSED        — the sources do not resolve it; say what is missing

## Citations
- Cite with the bracket tags exactly as given: [S1], [S2], ...
- EVERY sentence in your explanation must carry at least one tag, written in square \
brackets like [S1]. A sentence with no tag counts as unsupported.
- Quote EXACTLY, copying the characters from the source. Do not reorder, reword, \
re-punctuate or tidy a quote. A quote that does not appear verbatim is treated as \
fabricated, which is the most damaging error this system can make.
- Cite ONLY sources you can quote verbatim. Do not add a source as extra corroboration \
if you cannot copy an exact span from it: one solid citation beats two where the second \
cannot be checked.
- For each source you rely on, supply an EXACT quote copied character-for-character \
from that source. Do not paraphrase inside quotes, do not fix typos, do not add \
ellipses in the middle. Quotes are checked automatically against the source text \
and a mismatch invalidates your answer.
- Never cite a tag that is not in the SOURCES list.

## Tone
Plain English. Define terms of art on first use. Never give legal advice or tell \
the user what to do — state what the law says and let them decide. Where the \
answer turns on facts you do not have, say which facts.

Return JSON only, matching this schema:
{
  "verdict": "SUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED|NOT_ADDRESSED",
  "confidence": "high|medium|low",
  "explanation": "2-5 sentences, every sentence carrying at least one [S#] tag",
  "evidence": [
    {"tag": "S1", "quote": "exact verbatim span from S1",
     "relevance": "one sentence on what this establishes"}
  ],
  "caveats": ["conditions, exceptions or missing facts that change the answer"],
  "conflicts": ["describe any disagreement between sources, with tags"]
}"""

FACTCHECK_USER = """CLAIM TO CHECK:
{claim}

SOURCES:
{context}

Check the claim against these sources only.

THE CLAIM YOU ARE CHECKING, RESTATED:
{claim}

Answer about THAT claim. The sources above cover neighbouring topics; if none of
them addresses this specific claim, the verdict is NOT_ADDRESSED - but a source
that states a DIFFERENT value for the same quantity does address it, and
contradicts it. Every `tag` in
`evidence` must be a source label such as "S1" or "S4", exactly as it appears in
SOURCES - never a citation like "8 CFR 214.2"."""


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

# Baseline only. The proposal calls for a no-retrieval control, so this asks for
# the same JSON shape with no evidence at all. It exists to be beaten: whatever it
# scores is what the corpus is worth.
NO_RETRIEVAL_USER = """CLAIM TO CHECK:
{claim}

RETRIEVED CONTEXT:
(none — no evidence was retrieved; answer from your own knowledge)

Return the same JSON object. You have no passages to cite, so `evidence` must be
an empty list."""


ABSTAIN_TEMPLATE = {
    "verdict": "NOT_ADDRESSED",
    "confidence": "high",
    "explanation": (
        "The indexed corpus (INA, 8 CFR, 9 FAM, USCIS Policy Manual, Federal "
        "Register) does not contain authority addressing this claim, so no "
        "grounded verdict is possible."
    ),
    "evidence": [],
    "caveats": [
        "This means the claim was not found in the corpus — not that it is false.",
        "The corpus covers federal immigration law only; it has no case law, no "
        "consular post practice, and no state law.",
    ],
    "conflicts": [],
}


# ---------------------------------------------------------------------------
# Answer synthesis across sub-claims
# ---------------------------------------------------------------------------

SYNTHESIZE_SYSTEM = """You combine per-sub-claim verdicts into one overall \
assessment for the user.

- The overall verdict is driven by the sub-claims: any CONTRADICTED sub-claim \
makes the whole claim at best PARTIALLY_SUPPORTED; all SUPPORTED makes it \
SUPPORTED; if the decisive sub-claims are NOT_ADDRESSED, so is the whole.
- Keep every citation tag from the sub-verdicts. Do not invent new ones.
- Lead with the bottom line in one sentence, then the reasoning.
- Preserve caveats that would change a reader's decision. Drop redundant ones.

Return JSON only:
{"verdict": "...", "confidence": "high|medium|low", "summary": "one sentence",
 "explanation": "2-6 sentences with [S#] tags", "caveats": ["..."]}"""

SYNTHESIZE_USER = """ORIGINAL CLAIM:
{claim}

SUB-CLAIM VERDICTS:
{subverdicts}

Produce the overall assessment."""


# ---------------------------------------------------------------------------
# LLM-as-judge, used only by the offline evaluation
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You grade a fact-checking system's output against a gold \
reference. You are strict and you explain your reasoning.

Score each dimension 0-2:
- verdict_match:   2 = same verdict; 1 = adjacent (e.g. PARTIALLY vs SUPPORTED); 0 = opposite
- citation_quality: 2 = cites the gold authority; 1 = cites a related provision; \
0 = cites something irrelevant or nothing
- faithfulness:    2 = every statement traceable to the quoted sources; \
1 = minor unsupported additions; 0 = contains claims absent from the sources
- helpfulness:     2 = a non-lawyer could act on it; 1 = partially clear; 0 = unusable

Return JSON only:
{"verdict_match": 0-2, "citation_quality": 0-2, "faithfulness": 0-2,
 "helpfulness": 0-2, "reasoning": "2-3 sentences"}"""

JUDGE_USER = """CLAIM: {claim}

GOLD VERDICT: {gold_verdict}
GOLD AUTHORITY: {gold_citation}
GOLD NOTES: {gold_notes}

SYSTEM OUTPUT:
verdict: {sys_verdict}
explanation: {sys_explanation}
citations: {sys_citations}
quotes: {sys_quotes}

Grade it."""


# JSON Schemas for Ollama structured outputs.
#
# `format: "json"` only guarantees *valid* JSON, not the right keys. A smaller
# model reliably returns something well-formed and wrong - {"claim":..., "sources":
# ..., "analysis":...} instead of the verdict schema - which then parses cleanly,
# fails the `verdict` lookup and is silently recorded as NOT_ADDRESSED. Passing the
# schema constrains decoding to these keys, so the shape is enforced rather than
# requested. OpenAI ignores this and follows the prose schema in the system prompt.
FACTCHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["SUPPORTED", "CONTRADICTED",
                             "PARTIALLY_SUPPORTED", "NOT_ADDRESSED"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "explanation": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Must be a source label (S1, S2, ...). Left unconstrained,
                    # a smaller model puts the citation here instead, and the
                    # verifier then cannot resolve the tag to a retrieved chunk.
                    "tag": {"type": "string", "pattern": "^S[0-9]+$"},
                    "quote": {"type": "string"},
                    "relevance": {"type": "string"},
                },
                "required": ["tag", "quote"],
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "explanation", "evidence"],
}

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subclaims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "type": {"type": "string"}},
                "required": ["text"],
            },
        }
    },
    "required": ["subclaims"],
}
