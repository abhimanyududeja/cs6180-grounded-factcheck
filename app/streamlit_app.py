"""FOGA demo UI.

    streamlit run app/streamlit_app.py

Built for the demo, so the design goal is not "look impressive" — it is
**make the grounding visible**. Anyone can show a chatbot answering an
immigration question. The interesting part of this project is that every
sentence traces to a provision, the quotes are machine-verified against the
source text, and the system refuses when the corpus does not cover the question.
So the UI puts retrieval provenance, the grounding report and the raw source
text on screen next to the answer, rather than hiding them behind the prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foga.agent import ImmigrationAgent  # noqa: E402
from foga.config import ROOT, load_config  # noqa: E402
from foga.factcheck import FactChecker  # noqa: E402
from foga.index import HybridIndex  # noqa: E402
from foga.llm import LLM  # noqa: E402
from foga.retrieve import Retriever  # noqa: E402

st.set_page_config(page_title="FOGA — Immigration Law Fact-Checker",
                   page_icon="⚖️", layout="wide")

VERDICT_STYLE = {
    "SUPPORTED": ("#0b6b3a", "#e6f4ec", "SUPPORTED"),
    "CONTRADICTED": ("#8b1a1a", "#fbeaea", "CONTRADICTED"),
    "PARTIALLY_SUPPORTED": ("#8a6100", "#fdf4e3", "PARTIALLY SUPPORTED"),
    "NOT_ADDRESSED": ("#444a52", "#eef0f2", "NOT ADDRESSED — not in corpus"),
}
SOURCE_LABEL = {
    "ina": "INA / 8 U.S.C. (statute)",
    "cfr": "8 CFR (regulation)",
    "fam": "9 FAM (State Dept guidance)",
    "uscis_pm": "USCIS Policy Manual (agency guidance)",
    "fedreg": "Federal Register (rules & notices)",
}

EXAMPLES = [
    "F-1 students can work 20 hours per week on campus while school is in session",
    "The STEM OPT extension is 36 months long",
    "F-1 students get 12 months of OPT and STEM graduates can extend it by 36 months",
    "The H-1B cap is 85,000 visas per year",
    "You must hold a green card for ten years before applying for naturalization",
    "The USCIS Nebraska Service Center processes Form I-765 in 45 days",
]


@st.cache_resource(show_spinner="Loading index…")
def get_index():
    cfg = load_config()
    return HybridIndex.load(Path(cfg.path("index")))


@st.cache_resource(show_spinner="Loading reranker…")
def get_retriever(_index, rerank: bool):
    return Retriever(_index, load_config(), use_reranker=rerank)


def verdict_banner(verdict: str, confidence: str) -> None:
    fg, bg, label = VERDICT_STYLE.get(verdict, ("#333", "#eee", verdict))
    st.markdown(
        f"""<div style="background:{bg};border-left:6px solid {fg};
        padding:14px 18px;border-radius:6px;margin-bottom:8px;">
        <span style="color:{fg};font-size:1.35rem;font-weight:700;">{label}</span>
        <span style="color:{fg};opacity:.75;margin-left:14px;">
        confidence: {confidence}</span></div>""",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], key_prefix: str) -> None:
    for s in sources:
        score = s.get("rerank_score")
        score_txt = f" · relevance {score:.2f}" if score is not None else ""
        with st.expander(
            f"[{s['tag']}]  {s['citation']} — {s['title'][:70]}"
            f"  ({SOURCE_LABEL.get(s['source'], s['source'])}{score_txt})"
        ):
            st.caption(f"retrieved by: {s.get('found_by', '?')}")
            st.markdown(f"[Open the official source]({s['url']})")
            st.text(s["text"][:3000])


def render_grounding(sub) -> None:
    """The grounding report is the point of the project, so it gets real estate."""
    g = sub.grounding or {}
    grade = g.get("grade", "n/a")
    colour = {"GROUNDED": "green", "PARTIAL": "orange", "UNGROUNDED": "red"}.get(grade, "gray")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Grounding check:** :{colour}[{grade}]")
    c2.metric("Quotes verified verbatim", f"{g.get('quote_fidelity', 0) * 100:.0f}%")
    c3.metric("Sentences with a citation", f"{g.get('sentence_coverage', 0) * 100:.0f}%")
    if g.get("invalid_tags"):
        st.error(f"Model cited sources that were never retrieved: {g['invalid_tags']}")

    if sub.report and sub.report.quote_checks:
        st.caption("Each quote below was checked character-by-character against the "
                   "cited source, after normalizing whitespace and punctuation.")
        for q in sub.report.quote_checks:
            icon = "✅" if q.found else "❌"
            st.markdown(f"{icon} **{q.tag}** ({q.citation}) — similarity {q.similarity:.2f}")
            st.caption(f"> {q.quote[:400]}")
            if q.note:
                st.warning(q.note)


# ---------------------------------------------------------------------------

st.title("⚖️ FOGA — Grounded Fact-Checker for US Immigration Law")
st.caption(
    "Every answer is grounded in the INA (8 U.S.C.), 8 CFR, 9 FAM, the USCIS "
    "Policy Manual and the Federal Register. Quotes are verified against the "
    "source text automatically. **This is not legal advice.**"
)

try:
    index = get_index()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    cfg = load_config()
    mode = st.radio(
        "Mode", ["Fact-check (workflow)", "Research agent (tool use)", "Retrieval only"],
        help=("Workflow: decompose → retrieve → verdict, one pass. Predictable and "
              "cheap.\n\nAgent: the model runs its own searches and follows "
              "cross-references. Better on multi-hop questions, slower and pricier."),
    )
    provider = st.selectbox("LLM provider", ["openai", "ollama"],
                            index=0 if cfg.get_path("llm.provider") == "openai" else 1)
    model = st.text_input(
        "Model",
        value=(cfg.get_path("llm.model") if provider == "openai"
               else cfg.get_path("llm.ollama_model")),
    )
    st.divider()
    rerank = st.checkbox("Cross-encoder reranking", value=True,
                         help="Turn off to see how much reranking contributes.")
    decompose = st.checkbox(
        "Decompose compound claims", value=False,
        help="Off by default: on a local 8B model decomposition cost 12.3 points of "
             "accuracy (0.575 -> 0.452) in our evaluation, because weak sub-claims "
             "cascade into a blanket NOT_ADDRESSED. Turn it on to see that effect.",
    )
    k = st.slider("Passages retrieved", 3, 15, cfg.get_path("retrieval.final_k", 8))
    picked = st.multiselect(
        "Restrict to authorities", list(SOURCE_LABEL),
        format_func=lambda s: SOURCE_LABEL[s],
        help="Leave empty to search everything.",
    )
    st.divider()
    st.caption(f"**Index:** {len(index):,} chunks")
    st.caption(f"**Embeddings:** `{index.meta.get('embed_model')}`")
    counts: dict[str, int] = {}
    for c in index.chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        st.caption(f"  {SOURCE_LABEL.get(s, s)}: {n:,}")

example = st.selectbox("Try an example", [""] + EXAMPLES)
claim = st.text_area(
    "Claim or question",
    value=example,
    height=90,
    placeholder="e.g. F-1 students may work 20 hours per week on campus",
)
go = st.button("Check", type="primary", disabled=not claim.strip())

if go and claim.strip():
    cfg = load_config()
    cfg["factcheck"]["decompose"] = decompose
    cfg["retrieval"]["final_k"] = k
    retriever = get_retriever(index, rerank)
    retriever.final_k = k

    # ---- retrieval only ------------------------------------------------
    if mode == "Retrieval only":
        with st.spinner("Searching…"):
            res = retriever.search(claim, final_k=k, sources=picked or None)
        if res.expanded_query != res.query:
            st.caption(f"Expanded query: _{res.expanded_query}_")
        if res.abstain:
            st.warning(res.abstain_reason)
        st.subheader(f"{len(res.results)} passages")
        render_sources(
            [{"tag": r.tag, "citation": r.chunk.citation, "title": r.chunk.title,
              "source": r.chunk.source, "url": r.chunk.url, "text": r.chunk.text,
              "found_by": r.found_by, "rerank_score": r.rerank_score}
             for r in res.results],
            "ro",
        )

    # ---- agent ----------------------------------------------------------
    elif mode == "Research agent (tool use)":
        llm = LLM(provider=provider, model=model)
        agent = ImmigrationAgent(index, cfg, llm=llm)
        with st.spinner("Researching…"):
            result = agent.run(claim)
        st.markdown(result.answer)
        st.divider()
        st.subheader(f"Research trace — {len(result.trace)} tool calls")
        for i, t in enumerate(result.trace, 1):
            with st.expander(f"{i}. `{t.name}` → {t.n_results} results ({t.elapsed_s}s)"):
                st.json(t.arguments)
                st.caption(t.result_summary)
        if result.citations:
            st.subheader("Sources consulted")
            for c in result.citations:
                st.markdown(f"- **{c['citation']}** — [{c['title'][:70]}]({c['url']})")
        st.caption(
            f"{result.stats['elapsed_s']}s · {result.stats['tool_calls']} tool calls · "
            f"{result.stats['llm_calls']} LLM calls · "
            f"~${result.stats['est_cost_usd']:.4f} · {result.stats['model']}"
        )

    # ---- fact-check workflow --------------------------------------------
    else:
        llm = LLM(provider=provider, model=model)
        checker = FactChecker(index, cfg, llm=llm, retriever=retriever)
        with st.spinner("Checking against statute, regulation and guidance…"):
            result = checker.check(claim, sources=picked or None)

        verdict_banner(result.verdict, result.confidence)
        if result.summary:
            st.markdown(f"### {result.summary}")
        st.markdown(result.explanation)

        if result.caveats:
            st.warning("**Caveats**\n\n" + "\n".join(f"- {c}" for c in result.caveats))

        if result.currency_warnings:
            st.info(
                "**Recent regulatory activity on this topic** — the answer above "
                "reflects the indexed text; check whether these changed it.\n\n"
                + "\n".join(f"- {w['date']} — [{w['title']}]({w['url']})"
                            for w in result.currency_warnings)
            )

        st.divider()
        tabs = st.tabs([f"Sub-claim {s.id}" for s in result.subclaims] + ["Raw JSON"])
        for tab, sub in zip(tabs, result.subclaims):
            with tab:
                st.markdown(f"**{sub.text}**")
                verdict_banner(sub.verdict, sub.confidence)
                st.markdown(sub.explanation)
                if sub.conflicts:
                    st.error("**Sources disagree**\n\n"
                             + "\n".join(f"- {c}" for c in sub.conflicts))
                st.divider()
                render_grounding(sub)
                st.divider()
                st.markdown("#### Retrieved authorities")
                render_sources(sub.sources, f"sub{sub.id}")
        with tabs[-1]:
            st.json(result.to_dict())

        st.caption(
            f"{result.stats['elapsed_s']}s · {result.stats['llm_calls']} LLM calls · "
            f"{result.stats['input_tokens']}+{result.stats['output_tokens']} tokens · "
            f"~${result.stats['est_cost_usd']:.4f} · {result.stats['model']} · "
            f"{result.stats['grounded']}/{result.stats['n_subclaims']} sub-claims "
            f"fully grounded"
        )

st.divider()
st.caption(
    "FOGA is a course project for educational use. It is not legal advice, it is "
    "not a substitute for an immigration attorney or a DSO, and its corpus is a "
    "point-in-time snapshot of public federal sources. Always verify against the "
    "linked official source before acting."
)
