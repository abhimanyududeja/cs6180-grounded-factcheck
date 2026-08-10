# Grounded RAG Fact-Checker for U.S. Immigration and Tax Law

Give it a claim. It retrieves from a fixed corpus of statutes, regulations and agency
guidance, returns **SUPPORTED**, **CONTRADICTED**, **PARTIALLY_SUPPORTED** or
**NOT_ADDRESSED**, and shows the exact provisions and quotes it relied on. A separate
pass then checks *mechanically* whether those quotes actually appear in the passages
cited, by string comparison rather than by asking a model.

Runs entirely on your machine. No API key, no per-query cost, nothing leaves the laptop.

CS 6180 (Foundations for Generative AI) final project.
Abhimanyu Dudeja · Kashish Rahulbhai Khatri · Prasanna Adarsh Kolli

> **Not legal or tax advice.** A course project over public documents. Read the linked
> source before acting on anything it says.

---

## Why this problem

Ask a general-purpose model how long the STEM OPT extension runs. Sometimes it says 17
months (the pre-2016 rule), sometimes 24 (correct), sometimes 36 (never true). It cannot
tell you which, and it cannot cite a regulation. For a student deciding whether to accept
a job offer, that is not a small error.

Immigration and tax law suit grounded RAG unusually well: the sources are public, stable
and **written to be cited**. `8 CFR 214.2(f)(9)(i)` either says what you claim or it does
not, which makes hallucination *mechanically detectable* rather than a matter of opinion.
The two domains also overlap, since residence is simultaneously an immigration question
and a tax question.

## What it does

```
claim
  |- 1. decompose   split into atomic sub-claims   (off by default; see Results)
  |- 2. retrieve    dense + BM25 -> RRF -> cross-encoder rerank -> authority prior
  |- 3. gate        nothing scores well? -> abstain
  |- 4. generate    verdict + explanation + tagged quotes
  |- 5. verify      do the quotes appear in the cited passages? -> downgrade if not
```

Three modes in the demo:

| Mode | What it is |
|---|---|
| **Fact-check** | The evaluated pipeline. Verdict, citations, grounding grade. |
| **Research agent** | Tool-calling agent that runs its own searches. Retrieval-backed but *not* quote-verified, and not covered by any reported number. |
| **Retrieval only** | No LLM. The ranked passages and their scores. |

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
brew install ollama && ollama serve      # in its own terminal
ollama pull qwen3:8b                     # ~5 GB

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then build the corpus and index:

```bash
.venv/bin/python scripts/download.py     # ~45 min, mostly the USCIS crawl delay
.venv/bin/python scripts/build_index.py  # local embeddings, free
```

`data/` is not in the repo: it is roughly 160 MB of source documents, cached pages and a
FAISS index, all reproducible from those two commands. The evaluation output the report
cites *is* committed, under `reports/`, so the numbers can be checked without rebuilding
anything:

```bash
.venv/bin/python scripts/show_results.py
```

That prints every reported figure next to the file it came from. The USCIS Policy Manual fetch is
the slow part, because USCIS publishes no bulk archive and their `robots.txt` sets a
10-second crawl delay that the fetcher honours. Every URL is requested at most once ever
and cached, so re-running costs nothing.

Run it:

```bash
make app                                             # demo on :8501
.venv/bin/python scripts/ask.py "your claim here"    # CLI
.venv/bin/python -m pytest tests/ -q                 # 9 tests
```

## Corpus

2,229 documents, **14,726 chunks**, all from official bulk endpoints or published
sitemaps.

| Source | Domain | How it is obtained |
|---|---|---|
| 8 CFR | immigration | eCFR API, one request for the full title |
| INA / 8 U.S.C. | immigration | Office of the Law Revision Counsel bulk release ZIP |
| 9 FAM | immigration | The site's own JSON tree endpoint |
| USCIS Policy Manual | immigration | Official sitemap, at the `robots.txt` crawl delay |
| Federal Register | immigration | Public JSON API |
| IRS Pubs 17, 501, 519, 970 | tax | Stable PDFs from irs.gov |
| IRS Form 1040 instructions | tax | Same |

## Two design decisions worth the space

**Structure-aware chunking.** `8 CFR 214.2` is a single section of roughly 700,000
characters covering every nonimmigrant category from A-1 diplomats to Q cultural
visitors. A fixed-window chunker turns it into hundreds of chunks that all cite "8 CFR
214.2", so a verdict cannot say *which part* it used. Instead a stack machine walks the
statutory numbering, `(a)` then `(1)` then `(i)` then `(A)`, and recovers each
paragraph's position in the hierarchy. Chunks then cite `8 CFR 214.2(f)(9)(ii)` and deep
link to that paragraph.

**Verification is mechanical, not promised.** The prompt tells the model to quote
verbatim and tag every sentence. Instruction is not enforcement. `src/foga/verify.py`
independently re-checks that every tag was in the retrieved context, that every quoted
span appears in the passage it is attributed to, and that every sentence carries a
citation. An answer that fails is relabelled and shown *with the failure attached*
rather than quietly dropped.

## Results

73 hand-written claims (56 immigration, 17 tax), `qwen3:8b` throughout, zero API cost.
Full numbers and caveats in [RESULTS.md](RESULTS.md); the write-up is
[REPORT.md](REPORT.md).

| config | accuracy | residual hallucination |
|---|---|---|
| no retrieval | 0.205 | 1.000 |
| simple retrieval | 0.452 | 0.000 |
| **full, no decomposition** | **0.562** | 0.314 |
| full | 0.452 | 0.292 |

Retrieval ablation over the 39 provision-level gold items:

| config | R@8 | MRR |
|---|---|---|
| dense only | 0.538 | 0.300 |
| BM25 only | 0.385 | 0.184 |
| hybrid (RRF) | 0.641 | 0.270 |
| hybrid + rerank | 0.667 | 0.312 |
| **+ authority prior** | **0.769** | **0.375** |

Three things these numbers say, including the ones that cut against the design:

1. **Grounding beats memory**, roughly three to one.
2. **Decomposition costs 11 points** on an 8B model and is off by default. It splits a
   claim into pieces, each piece comes back NOT_ADDRESSED, and the synthesis rule
   averages them instead of noticing the contradiction.
3. **The most faithful configuration is not the most accurate one.** `simple_retrieval`
   ships zero ungrounded verdicts but is 11 points less accurate. Retrieving harder
   surfaces passages that are close but not on point, and the model quotes them.

## Limitations

- **73 claims.** One flipped verdict moves accuracy 1.4 points; gaps under about 4 points
  are not real.
- **The gold set is hand-written by the team**, not reviewed by an attorney or a tax
  professional. Every number rests on those labels.
- **The prompt was revised after seeing failures on these same claims**, so these are
  development numbers, not held-out.
- **The abstention gate never fires** at its configured threshold. Documented rather than
  quietly left in.
- **The research-agent mode is not quote-verified** and is excluded from every number.
- **No case law**, no state tax law. The corpus is a snapshot as of the download date.

## Layout

```
config.yaml               every tunable setting; all scripts read this
src/foga/
  chunk.py                structure-aware chunking + the numbering stack machine
  index.py                FAISS + BM25 with a legal-identifier-safe tokenizer
  retrieve.py             hybrid -> RRF -> rerank -> authority prior -> abstain
  factcheck.py            the workflow pipeline
  verify.py               mechanical citation and quote verification
  agent.py                tool-calling research agent
  llm.py                  Ollama and OpenAI drivers, cost accounting
  sources/                one adapter per authority
scripts/                  download, build_index, ask, research, evaluate
eval/goldset.jsonl        73 labeled claims
app/streamlit_app.py      demo UI
```

## Evaluation

```bash
.venv/bin/python scripts/evaluate.py --retrieval                      # free, no LLM
.venv/bin/python scripts/evaluate.py --configs full_no_decompose      # one config
.venv/bin/python scripts/evaluate.py --configs no_retrieval simple_retrieval full_no_decompose full
```

`--provider openai` runs the same harness against a frontier model, which is how the
comparison in REPORT.md §7.1 was produced. The demo itself is local-only.
