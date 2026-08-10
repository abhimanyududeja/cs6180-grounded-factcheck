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

import json
import os
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

st.set_page_config(page_title="Grounded Fact-Checker: Immigration and Tax",
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
    "irs_pub": "IRS publications (tax)",
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




EVAL_ORDER = {"no_retrieval": 0, "simple_retrieval": 1, "full_no_decompose": 2, "full": 3}
EVAL_LABEL = {
    "no_retrieval": "no retrieval",
    "simple_retrieval": "simple retrieval",
    "full_no_decompose": "full, no decomposition",
    "full": "full",
}


def _eval_rows(pattern: str) -> list[dict]:
    import glob
    rows = []
    for path in glob.glob(str(ROOT / pattern)):
        data = json.loads(Path(path).read_text())
        for row in data.get("generation_configs", {}).get("table", []):
            rows.append(row)
    return sorted(rows, key=lambda r: EVAL_ORDER.get(r["config"], 9))


def render_evaluation() -> None:
    """Show the measured results, read from reports/ rather than hardcoded.

    The demo and the report quote the same figures because both come from these
    files. If a run is re-done, this view changes with it.
    """
    st.subheader("Verdict evaluation", anchor=False)
    rows = _eval_rows("reports/*v3_*.json")
    if not rows:
        st.info("No evaluation output found. Run scripts/evaluate.py first.")
        return
    st.caption(f"{sum(1 for _ in open(ROOT / 'eval/goldset.jsonl'))} labeled claims · "
               "qwen3:8b · no API cost")
    st.dataframe(
        [{"config": EVAL_LABEL.get(r["config"], r["config"]),
          "accuracy": r["verdict_acc"],
          "citation prec.": r["cite_prec"],
          "quote fidelity": r["quote_fid"],
          "residual halluc.": r["resid_halluc"],
          "latency (s)": r["latency_s"]} for r in rows],
        hide_index=True, use_container_width=True,
    )
    best = max(rows, key=lambda r: r["verdict_acc"])
    base = next((r for r in rows if r["config"] == "no_retrieval"), None)
    if base:
        st.markdown(
            f"**Grounding beats memory.** Accuracy goes from {base['verdict_acc']} without "
            f"retrieval to {best['verdict_acc']} with the best configuration. The baseline's "
            f"residual hallucination rate of {base['resid_halluc']} means every decisive "
            "verdict it produced was ungrounded."
        )

    dec = {r["config"]: r for r in rows}
    if "full" in dec and "full_no_decompose" in dec:
        gap = round(dec["full_no_decompose"]["verdict_acc"] - dec["full"]["verdict_acc"], 3)
        st.markdown(
            f"**Decomposition costs {gap * 100:.1f} points** and is off by default. Sub-claims "
            "come back NOT_ADDRESSED individually and the synthesis rule averages them "
            "instead of noticing the contradiction."
        )

    st.divider()
    st.subheader("Retrieval ablation", anchor=False)
    import glob
    abl = sorted(glob.glob(str(ROOT / "reports/*v3_retrieval*.json")))
    if abl:
        table = json.loads(Path(abl[-1]).read_text()).get("retrieval", {}).get("table", [])
        if table:
            st.caption("39 provision-level gold items. A hit means the exact gold provision "
                       "at that rank. No LLM involved.")
            st.dataframe(
                [{"config": r.get("config"), "R@1": r.get("R@1"), "R@5": r.get("R@5"),
                  "R@8": r.get("R@8"), "MRR": r.get("MRR")} for r in table],
                hide_index=True, use_container_width=True,
            )

    oai = _eval_rows("reports/*oai_*.json")
    if oai:
        st.divider()
        st.subheader("Same pipeline, frontier model", anchor=False)
        st.caption(
            "A one-off measurement, not a mode you can run here: the demo is local only. "
            "Same corpus, same retrieval, same 73 claims, with only the generator swapped, "
            "to test whether the accuracy gap is the small model or the pipeline."
        )
        local = dec.get("full_no_decompose")
        f = oai[0]
        if local:
            st.dataframe([
                {"metric": "accuracy", "qwen3:8b": local["verdict_acc"], "gpt-5.6-luna": f["verdict_acc"]},
                {"metric": "citation precision", "qwen3:8b": local["cite_prec"], "gpt-5.6-luna": f["cite_prec"]},
                {"metric": "quote fidelity", "qwen3:8b": local["quote_fid"], "gpt-5.6-luna": f["quote_fid"]},
                {"metric": "residual hallucination", "qwen3:8b": local["resid_halluc"], "gpt-5.6-luna": f["resid_halluc"]},
                {"metric": "latency (s)", "qwen3:8b": local["latency_s"], "gpt-5.6-luna": f["latency_s"]},
            ], hide_index=True, use_container_width=True)
            st.markdown(
                "**Citation precision is identical.** It depends on retrieving the right "
                "provision, not on the model reading it, so the whole accuracy gap is "
                "generation. The retrieval work stands regardless of model."
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


st.markdown(
    """
    <style>
    h1 > a[href^="#"], h2 > a[href^="#"], h3 > a[href^="#"],
    h4 > a[href^="#"], h5 > a[href^="#"], h6 > a[href^="#"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Grounded Fact-Checker for U.S. Immigration and Tax Law")
st.caption(
    "Every answer is grounded in the INA (8 U.S.C.), 8 CFR, 9 FAM, the USCIS "
    "Policy Manual and the Federal Register on the immigration side, and IRS "
    "Publications 17, 501, 519 and 970 plus the Form 1040 instructions on the tax "
    "side. Quotes are verified against the source text automatically. "
    "**This is not legal or tax advice.**"
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
        "Mode", ["Fact-check (workflow)", "Research agent (tool use)", "Retrieval only",
                 "Evaluation results"],
        index=0,
        help=("Workflow: decompose → retrieve → verdict, one pass. Predictable and "
              "cheap.\n\nAgent: the model runs its own searches and follows "
              "cross-references. Better on multi-hop questions, and slower."),
    )
    # The demo is local-only: no API key, no per-query cost, nothing leaving the
    # machine. The OpenAI driver is still in llm.py and reachable from the
    # evaluation harness with --provider openai, which is what produced the
    # frontier comparison in the report; it is simply not offered here.
    # Local by default. A hosted deployment sets FOGA_LLM_PROVIDER=openai because
    # an 8B model cannot be served from a small container; retrieval and
    # verification are unchanged either way. There is deliberately no UI control
    # for this: a dropdown would let a demo query bill someone's API account.
    provider = os.environ.get("FOGA_LLM_PROVIDER", "ollama")
    if provider == "openai":
        model = st.text_input(
            "Model", value=os.environ.get("FOGA_LLM_MODEL", cfg.get_path("llm.model")))
        st.caption(
            "Hosted build: generation runs on the OpenAI API. Retrieval, reranking and "
            "verification are unchanged. The reported numbers come from the local "
            "qwen3:8b build; see REPORT.md section 7.1."
        )
    else:
        model = st.text_input("Model", value=cfg.get_path("llm.ollama_model"))
        st.caption("Runs locally through Ollama. No API key, no per-query cost.")
    st.divider()
    rerank = st.checkbox("Cross-encoder reranking", value=True,
                         help="Turn off to see how much reranking contributes.")
    decompose = st.checkbox(
        "Decompose compound claims", value=False,
        # Renaming the key discards any value a browser session cached from before
        # this default flipped, which otherwise silently keeps the worse setting.
        key="decompose_off_by_default",
        help="Off by default: on a local 8B model decomposition cost 11.0 points of "
             "accuracy (0.562 -> 0.452) in our evaluation. It splits a claim into "
             "pieces, each piece comes back NOT_ADDRESSED, and the synthesis rule "
             "averages them into PARTIALLY_SUPPORTED instead of noticing the "
             "contradiction. Turn it on to see that effect.",
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

if mode == "Evaluation results":
    render_evaluation()
    st.stop()

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

        # The workflow path runs every answer through verify.py, which checks each
        # quote against the passage it is attributed to. The agent path does not:
        # it writes prose with inline citations and nothing re-checks them. That is
        # the one guarantee this project rests on, so the difference is stated on
        # screen rather than left for the reader to infer from a missing badge.
        st.warning(
            "**Retrieval-backed, but not quote-verified.** The agent must search "
            "before it answers and cites only what its searches returned, but unlike "
            "Fact-check it does not run the mechanical quote check, returns no verdict "
            "label, and is not covered by any number in the report. Open the linked "
            "provisions yourself. For a verified verdict, switch Mode to "
            "**Fact-check (workflow)**.",
            icon="⚠️",
        )
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
            # Not a markdown heading: Streamlit anchors every heading, and the CSS
            # that hides those does not reach one rendered inside st.markdown. An
            # anchor beside a verdict summary reads like a link to the source.
            st.subheader(result.summary, anchor=False)
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
