"""Evaluation metrics.

Split deliberately into two families, because they answer different questions
and a project that reports only one of them is easy to fool:

**Retrieval metrics** (recall@k, MRR, nDCG) ask: did the pipeline put the
governing provision in front of the model at all? These need no LLM and no API
spend, so they can be run on every config change.

**Generation metrics** ask: given that context, did the model produce an honest,
grounded verdict? Verdict accuracy alone is not enough — a system can guess the
right verdict while citing the wrong statute, which is worse than useless to a
user who has to act on it. So we also report:

  * citation precision  — of the authorities cited, how many were right
  * quote fidelity      — fraction of quotes that actually appear in the source
  * hallucination rate  — fraction of answers failing the mechanical grounding check
  * abstention accuracy — did it say "I don't know" exactly when it should have

Abstention is scored in both directions. Refusing to answer a question the
corpus covers is a real failure, not a safe default.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


def normalize_citation(cit: str) -> str:
    """Reduce a citation to a comparable key.

    '8 CFR § 214.2(f)(10)(ii)(C)'          -> '8cfr214.2(f)(10)(ii)(c)'
    'INA § 214 (8 U.S.C. § 1184)'          -> 'ina214'
    'INA § 214(g) (8 U.S.C. § 1184(g))'    -> 'ina214(g)'

    The parallel U.S. Code cite has to be dropped, and it carries its own nested
    subsection parentheses — so it cannot be matched with a character class that
    stops at the first ')'.
    """
    c = cit.lower().replace("§", "").replace("u.s.c.", "usc").strip()
    c = re.sub(r"\s+", "", c)
    c = re.sub(r"\(8usc[\d.]+(?:\([a-z0-9]+\))*\)", "", c)   # drop the parallel USC cite
    return c


def citation_match(predicted: str, gold: str, level: str = "section") -> bool:
    """Does a predicted citation match the gold one?

    `level="section"` compares only the section number, so citing
    8 CFR 214.2(f)(10)(ii) counts as finding 8 CFR 214.2(f)(10). That is the
    right granularity for scoring: the system located the governing provision,
    and demanding an exact subsection match would penalize correct answers for
    quoting one clause deeper than the gold label happened to record.
    """
    p, g = normalize_citation(predicted), normalize_citation(gold)
    if not g:
        return False
    if level == "exact":
        return p == g
    p_sec = re.match(r"^([a-z]+[\d]*[\d.]*)", p)
    g_sec = re.match(r"^([a-z]+[\d]*[\d.]*)", g)
    if not (p_sec and g_sec):
        return g in p or p in g
    if p_sec.group(1) != g_sec.group(1):
        return False
    # Same section. Require the gold's subsection path to be a prefix of the
    # prediction's (or vice versa) so 214.2(f) and 214.2(h) do not both count.
    p_sub = "".join(re.findall(r"\(([a-z0-9]+)\)", p))
    g_sub = "".join(re.findall(r"\(([a-z0-9]+)\)", g))
    return p_sub.startswith(g_sub) or g_sub.startswith(p_sub)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_citations: list[str], gold_citation: str, k: int) -> float:
    return float(any(citation_match(c, gold_citation) for c in retrieved_citations[:k]))


def reciprocal_rank(retrieved_citations: list[str], gold_citation: str) -> float:
    for i, c in enumerate(retrieved_citations, 1):
        if citation_match(c, gold_citation):
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_citations: list[str], gold_citation: str, k: int) -> float:
    """Binary-relevance nDCG. With one relevant item the ideal DCG is 1, so
    this reduces to a rank-discounted hit — which is what we want: finding the
    provision at rank 1 should score higher than finding it at rank 8."""
    dcg = 0.0
    for i, c in enumerate(retrieved_citations[:k], 1):
        if citation_match(c, gold_citation):
            dcg += 1.0 / math.log2(i + 1)
            break
    return dcg


@dataclass
class RetrievalMetrics:
    n: int = 0
    recall_1: float = 0.0
    recall_3: float = 0.0
    recall_5: float = 0.0
    recall_8: float = 0.0
    mrr: float = 0.0
    ndcg_5: float = 0.0
    per_item: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {"n": self.n, "R@1": self.recall_1, "R@3": self.recall_3,
                "R@5": self.recall_5, "R@8": self.recall_8,
                "MRR": self.mrr, "nDCG@5": self.ndcg_5}


def score_retrieval(cases: list[dict]) -> RetrievalMetrics:
    """`cases`: [{"gold_citation": str, "retrieved": [citation, ...]}, ...].
    Cases with no gold citation (the abstention items) are excluded — there is
    no correct passage to retrieve."""
    scored = [c for c in cases if c.get("gold_citation")]
    m = RetrievalMetrics(n=len(scored))
    if not scored:
        return m
    for c in scored:
        got, gold = c["retrieved"], c["gold_citation"]
        row = {
            "id": c.get("id"),
            "gold": gold,
            "r@1": recall_at_k(got, gold, 1),
            "r@3": recall_at_k(got, gold, 3),
            "r@5": recall_at_k(got, gold, 5),
            "r@8": recall_at_k(got, gold, 8),
            "rr": reciprocal_rank(got, gold),
            "ndcg@5": ndcg_at_k(got, gold, 5),
            "top_citation": got[0] if got else None,
        }
        m.per_item.append(row)
    n = len(scored)
    m.recall_1 = sum(r["r@1"] for r in m.per_item) / n
    m.recall_3 = sum(r["r@3"] for r in m.per_item) / n
    m.recall_5 = sum(r["r@5"] for r in m.per_item) / n
    m.recall_8 = sum(r["r@8"] for r in m.per_item) / n
    m.mrr = sum(r["rr"] for r in m.per_item) / n
    m.ndcg_5 = sum(r["ndcg@5"] for r in m.per_item) / n
    return m


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

ADJACENT = {
    ("SUPPORTED", "PARTIALLY_SUPPORTED"), ("PARTIALLY_SUPPORTED", "SUPPORTED"),
    ("CONTRADICTED", "PARTIALLY_SUPPORTED"), ("PARTIALLY_SUPPORTED", "CONTRADICTED"),
}


def verdict_score(predicted: str, gold: str) -> float:
    """1.0 exact, 0.5 adjacent, 0.0 otherwise.

    Adjacency is partial credit, not a free pass: calling a false claim
    "partially supported" is a meaningfully smaller error than calling it
    "supported", but it is still wrong.
    """
    if predicted == gold:
        return 1.0
    return 0.5 if (predicted, gold) in ADJACENT else 0.0


@dataclass
class GenerationMetrics:
    n: int = 0
    verdict_accuracy: float = 0.0        # exact match
    verdict_score: float = 0.0           # with partial credit
    citation_precision: float = 0.0
    quote_fidelity: float = 0.0
    hallucination_rate: float = 0.0
    # Two rates, because one number cannot answer both questions. See
    # score_generation for why conflating them inverts the comparison.
    detected_hallucination_rate: float = 0.0
    residual_hallucination_rate: float = 0.0
    abstention_precision: float = 0.0    # of abstentions, how many were correct
    abstention_recall: float = 0.0       # of should-abstain items, how many did
    mean_latency_s: float = 0.0
    total_cost_usd: float = 0.0
    per_item: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "verdict_acc": round(self.verdict_accuracy, 3),
            "verdict_score": round(self.verdict_score, 3),
            "cite_prec": round(self.citation_precision, 3),
            "quote_fid": round(self.quote_fidelity, 3),
            "halluc_rate": round(self.hallucination_rate, 3),
            "resid_halluc": round(self.residual_hallucination_rate, 3),
            "detect_halluc": round(self.detected_hallucination_rate, 3),
            "abstain_P": round(self.abstention_precision, 3),
            "abstain_R": round(self.abstention_recall, 3),
            "latency_s": round(self.mean_latency_s, 1),
            "cost_usd": round(self.total_cost_usd, 4),
        }


def score_generation(cases: list[dict]) -> GenerationMetrics:
    """`cases`: [{"gold_verdict", "predicted_verdict", "gold_citation",
    "cited_citations", "quote_fidelity", "grounding_grade", "latency_s",
    "cost_usd"}]"""
    m = GenerationMetrics(n=len(cases))
    if not cases:
        return m

    exact = partial = 0.0
    cite_num = cite_den = 0
    quotes: list[float] = []
    halluc = 0
    abst_pred = abst_gold = abst_correct = 0

    for c in cases:
        gold_v, pred_v = c["gold_verdict"], c["predicted_verdict"]
        exact += float(gold_v == pred_v)
        partial += verdict_score(pred_v, gold_v)

        gold_c = c.get("gold_citation") or ""
        cited = c.get("cited_citations") or []
        if gold_c and cited:
            cite_den += 1
            cite_num += float(any(citation_match(x, gold_c) for x in cited))

        if c.get("quote_fidelity") is not None:
            quotes.append(c["quote_fidelity"])
        if c.get("grounding_grade") == "UNGROUNDED":
            halluc += 1

        should_abstain = gold_v == "NOT_ADDRESSED"
        did_abstain = pred_v == "NOT_ADDRESSED"
        abst_gold += int(should_abstain)
        abst_pred += int(did_abstain)
        abst_correct += int(should_abstain and did_abstain)

        m.per_item.append({
            "id": c.get("id"), "gold": gold_v, "pred": pred_v,
            "correct": gold_v == pred_v,
            "score": verdict_score(pred_v, gold_v),
            "cited_gold": bool(gold_c) and any(citation_match(x, gold_c) for x in cited),
            "grounding": c.get("grounding_grade"),
            "quote_fidelity": c.get("quote_fidelity"),
        })

    n = len(cases)
    m.verdict_accuracy = exact / n
    m.verdict_score = partial / n
    m.citation_precision = cite_num / cite_den if cite_den else 0.0
    m.quote_fidelity = sum(quotes) / len(quotes) if quotes else 1.0
    m.hallucination_rate = halluc / n

    # `hallucination_rate` above measures the *generator*: how often it produced
    # something ungrounded, whether or not the verifier then repaired it. A config
    # with a better verifier therefore scores WORSE on it, which inverts the
    # comparison if it is read as a quality ranking.
    #
    # `residual` measures what the user is actually shown: a verdict still
    # presented as SUPPORTED/CONTRADICTED that failed the mechanical check. A
    # caught-and-downgraded hallucination is no longer decisive, so it correctly
    # does not count here, while an uncaught one does. Lower is better, and this
    # is the faithfulness number to report.
    decisive_final = [c for c in cases
                      if c.get("predicted_verdict") in ("SUPPORTED", "CONTRADICTED")]
    m.residual_hallucination_rate = (
        sum(1 for c in decisive_final if c.get("grounding_grade") == "UNGROUNDED")
        / len(decisive_final) if decisive_final else 0.0
    )
    decisive_any = [c for c in cases
                    if c.get("predicted_verdict") in ("SUPPORTED", "CONTRADICTED")
                    or c.get("downgraded")]
    m.detected_hallucination_rate = (
        sum(1 for c in decisive_any if c.get("grounding_grade") == "UNGROUNDED")
        / len(decisive_any) if decisive_any else 0.0
    )
    m.abstention_precision = abst_correct / abst_pred if abst_pred else 0.0
    m.abstention_recall = abst_correct / abst_gold if abst_gold else 1.0
    m.mean_latency_s = sum(c.get("latency_s", 0) for c in cases) / n
    m.total_cost_usd = sum(c.get("cost_usd", 0) for c in cases)
    return m


def format_table(rows: list[dict], title: str = "") -> str:
    """Fixed-width table for the terminal and for pasting into the report."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    line = "  ".join(str(c).rjust(widths[c]) for c in cols)
    sep = "-" * len(line)
    out = ([title, "=" * len(title)] if title else []) + [line, sep]
    for r in rows:
        out.append("  ".join(f"{r.get(c, '')}".rjust(widths[c]) for c in cols))
    return "\n".join(out)
