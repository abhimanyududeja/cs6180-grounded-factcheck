"""Retrieval: hybrid candidate generation -> RRF fusion -> cross-encoder rerank.

Three stages, each fixing a failure the previous one cannot.

1. **Candidate generation.** Dense and BM25 searches run independently and each
   return ~40 candidates. Neither alone is sufficient (see `index.py`).

2. **Reciprocal Rank Fusion.** Combines the two ranked lists using only ranks,
   never raw scores. This matters because a FAISS cosine score (0-1) and a BM25
   score (unbounded, corpus-dependent) are not on comparable scales, so any
   weighted sum would need recalibration every time the corpus changes.

3. **Cross-encoder reranking.** RRF gives a good candidate pool but a mediocre
   ordering. A cross-encoder reads the query and each chunk *together* and
   scores actual relevance. It is ~40x slower per pair than a bi-encoder, which
   is exactly why it runs on 40 candidates rather than 9,721 chunks. This stage
   is what makes the top-8 context small enough to fit in a prompt while still
   containing the provision that answers the question.

The reranker's absolute score also drives **abstention**: if the best chunk
scores below a threshold, the corpus does not contain the answer, and a
fact-checker that guesses in that situation is worse than useless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .config import load_config
from .index import HybridIndex
from .schema import Chunk

# Rewriting a user's phrasing into statutory vocabulary before retrieval.
# These are not synonyms in general English — they are the specific places
# where everyday immigration vocabulary and the drafted text diverge, and each
# one was added after watching a real query fail.
QUERY_EXPANSIONS = {
    r"\bgreen card\b": "green card lawful permanent resident adjustment of status",
    r"\bwork permit\b": "work permit employment authorization document EAD Form I-765",
    r"\bopt\b": "OPT optional practical training",
    r"\bcpt\b": "CPT curricular practical training",
    r"\bh1b\b|\bh-1b\b": "H-1B specialty occupation nonimmigrant worker",
    r"\bf1\b|\bf-1\b": "F-1 academic student nonimmigrant",
    r"\bstem\b": "STEM science technology engineering mathematics extension",
    r"\bina\s+(\d+)": r"INA section \1 Immigration and Nationality Act",
    r"\btravel ban\b": "inadmissibility presidential proclamation entry restriction",
    r"\bgrace period\b": "grace period authorized period of stay departure",
    r"\bout of status\b": "failure to maintain status unlawful presence violation",
}


@dataclass
class Retrieved:
    """A chunk plus the full provenance of *why* it was retrieved. The UI and
    the evaluation both read these fields, so retrieval stays inspectable
    instead of being a black box."""

    chunk: Chunk
    rank: int
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    tag: str = ""                    # the handle the generator must cite, e.g. "S3"

    @property
    def found_by(self) -> str:
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense#{self.dense_rank + 1}")
        if self.sparse_rank is not None:
            parts.append(f"bm25#{self.sparse_rank + 1}")
        return "+".join(parts) or "?"


@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    results: list[Retrieved] = field(default_factory=list)
    abstain: bool = False
    abstain_reason: str = ""

    def context_block(self) -> str:
        return "\n\n".join(r.chunk.to_prompt_block(r.tag) for r in self.results)

    def by_tag(self, tag: str) -> Chunk | None:
        for r in self.results:
            if r.tag == tag:
                return r.chunk
        return None


def expand_query(query: str) -> str:
    """Append statutory vocabulary without discarding the user's own words —
    BM25 still needs the original tokens to match exact citations."""
    extra: list[str] = []
    low = query.lower()
    for pattern, expansion in QUERY_EXPANSIONS.items():
        if re.search(pattern, low):
            extra.append(re.sub(pattern, expansion, re.search(pattern, low).group(0))
                         if "\\1" in expansion else expansion)
    return f"{query} {' '.join(extra)}".strip() if extra else query


@lru_cache(maxsize=2)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


class Retriever:
    def __init__(self, index: HybridIndex, cfg=None, use_reranker: bool | None = None):
        self.index = index
        self.cfg = cfg or load_config()
        self.dense_k = self.cfg.get_path("retrieval.dense_k", 40)
        self.sparse_k = self.cfg.get_path("retrieval.sparse_k", 40)
        self.rrf_k = self.cfg.get_path("retrieval.rrf_k", 60)
        self.final_k = self.cfg.get_path("retrieval.final_k", 8)
        self.abstain_threshold = self.cfg.get_path("factcheck.abstain_threshold", -6.0)
        self.use_reranker = (
            self.cfg.get_path("retrieval.rerank", True) if use_reranker is None else use_reranker
        )
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from .llm import Embedder

            provider = self.index.meta.get("embed_provider")
            model = self.index.meta.get("embed_model")
            # Must match the model the index was built with, or the vectors are
            # meaningless. Read it off the index rather than the config.
            self._embedder = Embedder(provider=provider, model=model)
        return self._embedder

    def _authority_bonus(self, chunk) -> float:
        """Nudge binding authority above guidance that merely restates it.

        Without this the ranking systematically prefers the USCIS Policy Manual
        and 9 FAM, because they are written in the plain English a user's query
        is also written in, while 8 CFR says the same thing in drafted statutory
        language. That is a real lexical-match advantage and it has nothing to do
        with which source is authoritative.

        For a fact-checker the ordering matters: the Policy Manual is USCIS's
        interpretation and is persuasive, but 8 CFR is binding. Given two
        passages that both answer the question, the regulation is the better
        citation.

        The bonus is deliberately small. It breaks ties between comparably
        relevant passages; it cannot promote an irrelevant statute over a
        directly on-point Policy Manual chapter, which would be a worse failure
        than the one it fixes. Set `retrieval.authority_bonus: 0.0` to ablate it.
        """
        weight = self.cfg.get_path("retrieval.authority_bonus", 0.0)
        if not weight:
            return 0.0
        # authority_rank: 1 = statute, 2 = regulation, 3 = guidance, 4 = notices
        return weight * (4 - min(chunk.authority_rank, 4))

    def search(
        self,
        query: str,
        final_k: int | None = None,
        sources: list[str] | None = None,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """`mode` is one of hybrid | dense | sparse — the evaluation ablation
        uses it to quantify what each retrieval half contributes."""
        final_k = final_k or self.final_k
        expanded = expand_query(query)
        out = RetrievalResult(query=query, expanded_query=expanded)

        dense: list[tuple[int, float]] = []
        sparse: list[tuple[int, float]] = []
        if mode in ("hybrid", "dense"):
            qvec = self.embedder.encode([expanded])[0]
            dense = self.index.dense_search(qvec, self.dense_k)
        if mode in ("hybrid", "sparse"):
            sparse = self.index.sparse_search(expanded, self.sparse_k)

        # --- Reciprocal Rank Fusion ---------------------------------------
        fused: dict[int, dict] = {}
        for rank, (idx, score) in enumerate(dense):
            fused.setdefault(idx, {}).update(
                {"dense_rank": rank, "dense_score": score}
            )
        for rank, (idx, score) in enumerate(sparse):
            fused.setdefault(idx, {}).update(
                {"sparse_rank": rank, "sparse_score": score}
            )
        for idx, info in fused.items():
            info["rrf"] = sum(
                1.0 / (self.rrf_k + info[key] + 1)
                for key in ("dense_rank", "sparse_rank")
                if key in info
            )

        ordered = sorted(fused.items(), key=lambda kv: -kv[1]["rrf"])
        if sources:
            ordered = [(i, v) for i, v in ordered if self.index.get(i).source in sources]
        if not ordered:
            out.abstain = True
            out.abstain_reason = "No candidate passages matched the query."
            return out

        # --- Cross-encoder rerank -----------------------------------------
        pool = ordered[: max(self.dense_k, self.sparse_k)]
        if self.use_reranker and pool:
            model = _get_reranker(self.cfg.get_path("retrieval.rerank_model"))
            pairs = [(query, self.index.get(i).text[:2000]) for i, _ in pool]
            scores = model.predict(pairs, show_progress_bar=False)
            adjusted = [
                float(s) + self._authority_bonus(self.index.get(pool[j][0]))
                for j, s in enumerate(scores)
            ]
            order = sorted(range(len(pool)), key=lambda j: -adjusted[j])
            pool = [(pool[j][0], {**pool[j][1], "rerank": float(scores[j]),
                                  "rerank_adjusted": adjusted[j]})
                    for j in order]

        top = pool[:final_k]
        for rank, (idx, info) in enumerate(top):
            out.results.append(
                Retrieved(
                    chunk=self.index.get(idx),
                    rank=rank,
                    rrf_score=info.get("rrf", 0.0),
                    dense_rank=info.get("dense_rank"),
                    dense_score=info.get("dense_score"),
                    sparse_rank=info.get("sparse_rank"),
                    sparse_score=info.get("sparse_score"),
                    rerank_score=info.get("rerank"),
                    tag=f"S{rank + 1}",
                )
            )

        # --- Abstention ----------------------------------------------------
        # Refusing to answer is a feature here. A fact-checker that always
        # produces a verdict will confidently cite an irrelevant regulation
        # when the corpus simply has nothing on point.
        if self.use_reranker and out.results:
            best = out.results[0].rerank_score
            if best is not None and best < self.abstain_threshold:
                out.abstain = True
                out.abstain_reason = (
                    f"Best passage scored {best:.2f}, below the "
                    f"{self.abstain_threshold} relevance threshold. The corpus "
                    f"does not appear to contain authority on this claim."
                )
        return out
