"""Hybrid index: dense vectors (FAISS) + lexical (BM25).

Both halves are load-bearing for this corpus, for different reasons.

**Dense** handles the vocabulary gap. A student asks "can I work off campus?"
and the regulation says "employment other than on the school's premises."
No word overlaps; only embeddings connect them.

**BM25** handles the opposite failure. A user asks about "8 CFR 214.2(f)(10)"
or "Form I-765" or "INA 245(i)" — exact identifiers where embeddings are
notoriously weak, because "214.2(f)(10)" and "214.2(f)(11)" sit almost on top
of each other in vector space but mean entirely different things. Missing the
exact-identifier case is fatal for a citation-grounded fact-checker.

Fusing them with Reciprocal Rank Fusion needs no score calibration between the
two systems, which is why RRF is used here rather than a weighted sum.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .schema import Chunk

# Legal identifiers must survive tokenization intact: "214.2(f)(9)", "I-765",
# "8 U.S.C. 1101". A default word tokenizer shreds all three.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-–][a-z0-9]+)*(?:\([a-z0-9]+\))*")


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("§", " section ").replace("u.s.c.", "usc")
    toks = _TOKEN_RE.findall(text)
    out: list[str] = []
    for t in toks:
        out.append(t)
        # Index "214.2(f)(9)" also as "214.2" and "f" and "9" so a query for
        # the parent subsection still matches the child.
        if "(" in t:
            out.append(t.split("(")[0])
            out.extend(re.findall(r"\(([a-z0-9]+)\)", t))
        if "-" in t and len(t) < 12:          # I-765, H-1B
            out.append(t.replace("-", ""))
    return [t for t in out if t]


class HybridIndex:
    """Owns the chunks, the FAISS index and the BM25 model together, so they
    can never drift out of sync with each other."""

    def __init__(self, chunks: list[Chunk], faiss_index=None, bm25=None, meta: dict | None = None):
        self.chunks = chunks
        self.faiss = faiss_index
        self.bm25 = bm25
        self.meta = meta or {}
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    # -- build ------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, verbose: bool = True) -> "HybridIndex":
        import faiss
        from rank_bm25 import BM25Okapi

        texts = [c.text for c in chunks]
        if verbose:
            n_tok = sum(len(t) for t in texts) // 4
            print(f"  embedding {len(texts):,} chunks (~{n_tok / 1e6:.1f}M tokens)")
        vecs = embedder.encode(texts, show_progress=verbose)

        dims = vecs.shape[1]
        index = faiss.IndexFlatIP(dims)   # vectors are unit-normalized -> cosine
        index.add(vecs)

        if verbose:
            print(f"  building BM25 over {len(texts):,} chunks")
        bm25 = BM25Okapi([tokenize(t) for t in texts])

        meta = {
            "n_chunks": len(chunks),
            "dims": dims,
            "embed_model": getattr(embedder, "model", "unknown"),
            "embed_provider": getattr(embedder, "provider", "unknown"),
        }
        return cls(chunks, index, bm25, meta)

    # -- persist ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(directory / "dense.faiss"))
        with open(directory / "bm25.pkl", "wb") as fh:
            pickle.dump(self.bm25, fh)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        import faiss

        directory = Path(directory)
        if not (directory / "dense.faiss").exists():
            raise FileNotFoundError(
                f"No index at {directory}. Run: python scripts/build_index.py"
            )
        chunks = [
            Chunk(**json.loads(l))
            for l in open(directory / "chunks.jsonl", encoding="utf-8")
            if l.strip()
        ]
        index = faiss.read_index(str(directory / "dense.faiss"))
        with open(directory / "bm25.pkl", "rb") as fh:
            bm25 = pickle.load(fh)
        meta = json.loads((directory / "meta.json").read_text())
        return cls(chunks, index, bm25, meta)

    # -- search -----------------------------------------------------------

    def dense_search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.faiss.search(qvec.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def get(self, idx: int) -> Chunk:
        return self.chunks[idx]

    def __len__(self) -> int:
        return len(self.chunks)
