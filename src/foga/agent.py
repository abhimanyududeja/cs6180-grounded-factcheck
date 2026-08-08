"""Agentic mode: the model drives its own retrieval through tool calls.

The `factcheck.py` pipeline is a *fixed workflow* — decompose, retrieve once per
sub-claim, generate. That is the right default: it is predictable, cheap, and
easy to evaluate. But it has a real ceiling, which this module exists to break:

  A single retrieval pass answers only the question the user actually asked.
  "Can I start my STEM OPT while my H-1B petition is pending?" needs the STEM
  OPT rules, the cap-gap rules, and the H-1B change-of-status rules — three
  different searches, where you only know to run the second and third after
  reading the first.

So here the model gets tools and decides for itself:

  search_corpus       — retrieve on any query it composes
  lookup_citation     — fetch a specific provision by citation ("8 CFR 214.2(f)")
  check_recent_changes— query the Federal Register for activity on a topic
  compare_authorities — pull the same topic from statute vs regulation vs guidance

The report should compare the two modes head to head: the workflow is cheaper
and more predictable, the agent handles multi-hop questions the workflow cannot.
That trade-off is the interesting finding, not a defect in either one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import load_config
from .index import HybridIndex
from .llm import LLM, estimate_cost
from .retrieve import Retriever

AGENT_SYSTEM = """You are a research agent for US immigration law, working for \
international students and foreign workers.

You have tools that search an indexed corpus of the INA (8 U.S.C.), 8 CFR, 9 FAM, \
the USCIS Policy Manual, and recent Federal Register documents.

How to work:
1. Plan first. Multi-part questions need multiple searches — identify each part.
2. Search for one thing at a time, with the vocabulary the law uses, not the \
vocabulary the user used. Read what comes back before deciding the next search.
3. When a source references another provision, look that provision up rather than \
assuming what it says.
4. Check whether a rule you are relying on changed recently.
5. Stop when you can answer with citations, or when you can state precisely what \
the corpus does not cover.

Hard rules:
- Your own knowledge of immigration law is NOT evidence. Only tool results count.
- Cite every factual statement with the citation string returned by the tools \
(e.g. "8 CFR 214.2(f)(10)(ii)"), not with an invented reference.
- If searches come back empty or off-topic, say the corpus does not address it. \
Do not fill the gap from memory.
- Never give legal advice. Report what the law says.

Finish with a clear answer in plain English, with citations inline."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the immigration law corpus. Use the vocabulary the law "
                "uses. Returns passages with citations and links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": ["ina", "cfr", "fam", "uscis_pm", "fedreg"]},
                        "description": (
                            "Optional filter. ina=statute, cfr=regulation, "
                            "fam=State Dept consular guidance, "
                            "uscis_pm=USCIS Policy Manual, fedreg=recent rules."
                        ),
                    },
                    "k": {"type": "integer", "description": "How many passages (default 6)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_citation",
            "description": (
                "Retrieve a specific provision by its citation, e.g. "
                "'8 CFR 214.2(f)(10)', 'INA 245(i)', '9 FAM 402.5'. Use this when "
                "another source cross-references a provision you have not read."
            ),
            "parameters": {
                "type": "object",
                "properties": {"citation": {"type": "string"}},
                "required": ["citation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_recent_changes",
            "description": (
                "Search recent Federal Register rules and notices for changes "
                "affecting a topic. Use before relying on any rule."
            ),
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_authorities",
            "description": (
                "Retrieve what statute, regulation and agency guidance each say "
                "about one topic, to check whether they agree."
            ),
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result_summary: str
    n_results: int
    elapsed_s: float


@dataclass
class AgentResult:
    question: str
    answer: str
    trace: list[ToolCall] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "trace": [vars(t) for t in self.trace],
            "citations": self.citations,
            "stats": self.stats,
        }


class ImmigrationAgent:
    def __init__(self, index: HybridIndex, cfg=None, llm: LLM | None = None,
                 max_steps: int = 8):
        self.cfg = cfg or load_config()
        self.index = index
        self.llm = llm or LLM()
        self.retriever = Retriever(index, self.cfg)
        self.max_steps = max_steps
        self._seen: dict[str, dict] = {}   # citation -> source record

    # -- tool implementations --------------------------------------------

    def _record(self, results) -> list[dict]:
        out = []
        for r in results:
            rec = {
                "citation": r.chunk.citation,
                "title": r.chunk.title,
                "source": r.chunk.source,
                "url": r.chunk.url,
                "text": r.chunk.text[:1200],
            }
            self._seen[r.chunk.citation] = rec
            out.append(rec)
        return out

    def search_corpus(self, query: str, sources=None, k: int = 6) -> dict:
        res = self.retriever.search(query, final_k=k, sources=sources)
        if res.abstain:
            return {"n_results": 0, "note": res.abstain_reason, "passages": []}
        return {"n_results": len(res.results), "passages": self._record(res.results)}

    def lookup_citation(self, citation: str) -> dict:
        """Exact-ish citation lookup. Tries a literal match over chunk citations
        first, then falls back to lexical search — a user or model may write
        '8 CFR 214.2(f)' when the chunk is stored as '8 CFR § 214.2(f)(1)'."""
        norm = citation.lower().replace("§", "").replace(" ", "")
        exact = [c for c in self.index.chunks
                 if norm in c.citation.lower().replace("§", "").replace(" ", "")]
        if exact:
            exact.sort(key=lambda c: (len(c.citation), c.seq))
            recs = []
            for c in exact[:6]:
                rec = {"citation": c.citation, "title": c.title, "source": c.source,
                       "url": c.url, "text": c.text[:1200]}
                self._seen[c.citation] = rec
                recs.append(rec)
            return {"n_results": len(recs), "match": "exact", "passages": recs}
        res = self.retriever.search(citation, final_k=5)
        return {"n_results": len(res.results), "match": "lexical",
                "passages": self._record(res.results)}

    def check_recent_changes(self, topic: str) -> dict:
        res = self.retriever.search(topic, final_k=6, sources=["fedreg"])
        rows = [{"citation": r.chunk.citation, "title": r.chunk.title,
                 "effective_date": r.chunk.effective_date, "url": r.chunk.url,
                 "summary": r.chunk.text[:500]} for r in res.results]
        rows.sort(key=lambda x: x["effective_date"] or "", reverse=True)
        return {"n_results": len(rows), "documents": rows}

    def compare_authorities(self, topic: str) -> dict:
        out: dict = {}
        for label, srcs in (("statute", ["ina"]), ("regulation", ["cfr"]),
                            ("agency_guidance", ["fam", "uscis_pm"])):
            res = self.retriever.search(topic, final_k=3, sources=srcs)
            out[label] = self._record(res.results)
        return out

    # -- loop --------------------------------------------------------------

    def run(self, question: str, verbose: bool = False) -> AgentResult:
        t0 = time.time()
        dispatch = {
            "search_corpus": self.search_corpus,
            "lookup_citation": self.lookup_citation,
            "check_recent_changes": self.check_recent_changes,
            "compare_authorities": self.compare_authorities,
        }
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": question},
        ]
        trace: list[ToolCall] = []

        for step in range(self.max_steps):
            msg = self.llm.chat("", "", tools=TOOLS, messages=messages)
            if not isinstance(msg, dict):
                break
            messages.append({k: v for k, v in msg.items() if v is not None})
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = msg.get("content") or ""
                return self._finish(question, answer, trace, t0)

            for call in calls:
                fn = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if verbose:
                    print(f"  step {step + 1}: {fn}({json.dumps(arguments)[:100]})")
                ts = time.time()
                try:
                    result = dispatch[fn](**arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                n = result.get("n_results", len(result) if isinstance(result, dict) else 0)
                trace.append(ToolCall(
                    name=fn, arguments=arguments,
                    result_summary=json.dumps(result)[:300],
                    n_results=n, elapsed_s=round(time.time() - ts, 2),
                ))
                if verbose:
                    print(f"      -> {n} results in {trace[-1].elapsed_s}s")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result)[:12000],
                })

        # Ran out of steps: ask for the best answer available from what it has.
        messages.append({
            "role": "user",
            "content": ("You have reached the research step limit. Answer now using "
                        "only what your tool results established, and state plainly "
                        "what remains unresolved."),
        })
        final = self.llm.chat("", "", messages=messages)
        answer = final.text if hasattr(final, "text") else str(final)
        return self._finish(question, answer, trace, t0)

    def _finish(self, question: str, answer: str, trace: list[ToolCall],
                t0: float) -> AgentResult:
        # Report only the sources the answer actually names.
        cited = [rec for cit, rec in self._seen.items()
                 if cit.lower().replace("§", "").replace(" ", "")[:18]
                 in answer.lower().replace("§", "").replace(" ", "")]
        return AgentResult(
            question=question,
            answer=answer,
            trace=trace,
            citations=cited or list(self._seen.values())[:8],
            stats={
                "elapsed_s": round(time.time() - t0, 1),
                "tool_calls": len(trace),
                "llm_calls": self.llm.usage.calls,
                "input_tokens": self.llm.usage.input_tokens,
                "output_tokens": self.llm.usage.output_tokens,
                "est_cost_usd": round(estimate_cost(self.llm.usage, self.llm.model), 5),
                "model": self.llm.model,
            },
        )
