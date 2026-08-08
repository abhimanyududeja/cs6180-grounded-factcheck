# FOGA — A Grounded RAG Fact-Checker for US Immigration Law

A retrieval-augmented fact-checking system for international students and foreign
workers. You give it a claim about US immigration law; it returns a verdict, an
explanation where **every sentence carries a citation**, links to the exact
provisions relied on, and a machine-verified check that the quotes it used are
genuinely in those provisions.

It is built to refuse. If the corpus does not settle the question, it says
`NOT_ADDRESSED` instead of guessing — because the failure mode that matters here
is a confident wrong answer that someone acts on.

> **Not legal advice.** This is a course project. It is not a substitute for an
> immigration attorney or your school's DSO. Always read the linked official
> source before acting.

---

## Why this problem

Ask a general-purpose chatbot "how long is the STEM OPT extension?" and it will
answer from memory. Sometimes memory says 17 months (the pre-2016 rule), sometimes
24 (correct), sometimes 36 (never true). The model cannot tell you which, cannot
cite a regulation, and cannot tell you whether the rule changed last month. For a
student deciding whether to accept a job offer, that is not a small error.

Immigration law is an unusually good fit for grounded RAG:

- **The corpus is public, authoritative and machine-readable.** Statute,
  regulation and agency guidance are all published by the government.
- **Answers have exact addresses.** "8 CFR § 214.2(f)(10)(ii)(C)" either says what
  you claim or it doesn't. That makes hallucination *mechanically detectable*
  rather than a matter of opinion.
- **The stakes are real and the users are underserved.** International students
  routinely rely on forum posts and word of mouth.
- **It changes.** A snapshot corpus needs to know it might be stale, which forces
  the design to handle currency explicitly.

---

## What it does

```
claim
  ├── decompose into atomic sub-claims        ← compound claims hide half-truths
  ├── for each sub-claim:
  │     ├── hybrid retrieval (dense + BM25 → RRF)
  │     ├── cross-encoder rerank              ← or abstain if nothing scores well
  │     ├── grounded verdict generation
  │     └── mechanical citation verification  ← quotes checked against the source
  ├── synthesize one overall verdict
  └── Federal Register currency check         ← "this rule changed last month"
```

Two operating modes:

| | **Workflow** (`ask.py`) | **Agent** (`research.py`) |
|---|---|---|
| Retrieval | one pass per sub-claim | model issues its own searches |
| Tools | — | `search_corpus`, `lookup_citation`, `check_recent_changes`, `compare_authorities` |
| Good at | single-hop claims, batch evaluation | multi-hop questions, following cross-references |
| Cost | low, predictable | 3-6× higher, variable |

---

## The corpus, and how it is obtained without scraping

Four of the five sources are genuine bulk endpoints or official APIs. The fifth
is fetched from an officially published URL manifest at the rate the site's own
`robots.txt` specifies.

| Source | What it is | How we get it | Bulk? |
|---|---|---|---|
| **8 CFR** | Binding immigration regulations | eCFR API — `GET /api/versioner/v1/full/{date}/title-8.xml` returns all 5.4 MB in one unauthenticated request | ✅ one request |
| **INA / 8 U.S.C.** | The statute itself | Office of the Law Revision Counsel bulk release-point ZIP (USLM XML). The current release point is discovered from the official download index, so the corpus tracks new public laws | ✅ one file |
| **9 FAM** | State Dept guidance consular officers actually apply | `GET fam.state.gov/api/Tree/GetTreeByVolumeId?Id=09FAM` — the JSON endpoint the site's own front end uses — gives the complete authoritative section list; we then fetch each of the ~150 pages once | ✅ official index |
| **Federal Register** | Recent rules and notices, for currency checks | `api.federalregister.gov` JSON API, no key | ✅ API |
| **USCIS Policy Manual** | USCIS's own interpretation | **No bulk file exists.** See below | ⚠️ page-by-page |

### On the USCIS Policy Manual specifically

USCIS publishes no downloadable Policy Manual archive, so this one source has to
be fetched page by page. We do that the legitimate way, and the distinction from
"scraping" is worth being precise about:

1. **URLs come from USCIS's own sitemap** (`uscis.gov/sitemap.xml`), which lists
   1,302 Policy Manual pages. Sitemaps exist precisely so automated clients know
   the canonical URL set. We never follow links, guess URLs, or crawl.
2. **The rate comes from their `robots.txt`**, which sets `Crawl-delay: 10` for
   all agents and does *not* disallow `/policy-manual/`. We wait exactly 10
   seconds between requests. Their own published policy is the rate limit.
3. **We identify ourselves** with a descriptive User-Agent and a contact email.
4. **Each URL is requested at most once, ever.** Responses cache to
   `data/raw/_cache/`, so re-running the pipeline, rebuilding the index or
   restarting after a crash sends zero new requests.
5. **The content is public domain.** US government works carry no copyright
   (17 U.S.C. § 105).

The default run fetches Volumes 2, 6 and 7 (Nonimmigrants, Employment-Based
Immigrants, Adjustment of Status) — 271 pages, ~45 minutes — which is the
material international students and workers actually need. `--full` fetches all
1,302 (~3.6 hours); already-cached pages are skipped.

If you would rather not fetch it at all, `config.yaml` has
`sources.uscis_pm.enabled: false`, and the system runs on statute, regulation and
FAM alone.

---

## Setup

Requires Python 3.12 (3.13+ has patchy wheels for `faiss` and `torch`).

```bash
make setup                      # venv + dependencies
cp .env.example .env            # then add your OpenAI key
make download                   # ~45 min, mostly waiting on the 10 s crawl delay
make index                      # chunk + embed + build (~$0.10)
make test                       # chunker regression tests
```

Then:

```bash
make ask Q="F-1 students can work 20 hours per week on campus"
make app                        # Streamlit demo
make eval-retrieval             # free — retrieval metrics + ablation
make eval                       # full evaluation (~$0.30)
```

### Getting the model

You are using the **OpenAI API**, so there is no model to download — you need a
key:

1. Sign in at <https://platform.openai.com/api-keys>, create a key, and add a few
   dollars of credit under Billing. The whole project (index build + full
   evaluation + a live demo) runs comfortably under **$5**.
2. Put it in `.env` as `OPENAI_API_KEY=sk-...`.

Defaults in `config.yaml`, with current pricing per 1M tokens:

| Role | Model | Input / Output | Why |
|---|---|---|---|
| Generation | `gpt-5.6-luna` | $0.20 / $1.20 | Cheap enough to run the eval repeatedly |
| Evaluation judge | `gpt-5.6-terra` | $2.00 / $12.00 | A stronger model should grade, not the model under test |
| Embeddings | `text-embedding-3-small` | $0.02 | 1536-d; the full corpus embeds for about seven cents |

`python scripts/list_models.py` prints what your account can actually reach, in
case the defaults have moved on.

**Optional local fallback.** An 8 GB machine can run a ~4B model, which is worth
having for the report as a small-vs-large ablation and as demo insurance if the
classroom wifi fails:

```bash
brew install ollama && ollama serve      # in one terminal
ollama pull qwen3:4b                     # ~2.6 GB
python scripts/ask.py "..." --provider ollama
python scripts/build_index.py --local-embed   # free embeddings too
```

---

## Design decisions worth defending

### 1. Structure-aware chunking (the biggest single win)

**8 CFR § 214.2 is one section of 700,000 characters.** It covers every
nonimmigrant category — A-1 diplomats through Q cultural visitors. A fixed
2,000-character window turns it into ~350 chunks that *all* cite "8 CFR § 214.2",
so the system tells an F-1 student its answer came from a section that also
contains the rules for crewmen and treaty investors, with no way to say which
part it used.

Instead, [`chunk.py`](src/foga/chunk.py) runs a stack machine over the statutory
numbering — `(a)`, `(1)`, `(i)`, `(A)` — to recover each paragraph's position in
the hierarchy. Chunks then cite `8 CFR § 214.2(f)(9)(ii)` and deep-link to that
exact paragraph. One section becomes **204 distinct citable provisions**.

Two bugs found while building this, both now regression-tested in
[`tests/test_chunk.py`](tests/test_chunk.py):

- The section-contents table at the top of § 214.2 lists `(a)` through `(w)`.
  Fed to the numbering stack, it convinced the parser the section had already
  reached `(w)` before the body's real `(a)` began, so every path came out wrong.
- `(i)` is both the 9th letter and the 1st roman numeral. Reading it as `(h)`'s
  successor rather than as opening `(h)(4)(i)` silently relocated **166 H-visa
  chunks** into the media-representative subsection.

### 2. Hybrid retrieval, because each half fails differently

Dense embeddings handle the vocabulary gap: a student asks "can I work off
campus?" and the regulation says "employment other than on the school's
premises" — zero word overlap. BM25 handles exact identifiers: `214.2(f)(10)`,
`Form I-765`, `INA 245(i)`. Embeddings are notoriously weak here because
`214.2(f)(10)` and `214.2(f)(11)` sit almost on top of each other in vector space
and mean entirely different things.

Fused with Reciprocal Rank Fusion, which uses only ranks — a FAISS cosine score
and an unbounded BM25 score are not on comparable scales, so any weighted sum
would need recalibration whenever the corpus changes.

### 3. The INA ↔ 8 U.S.C. crosswalk

Everyone — USCIS, the FAM, every immigration lawyer — cites the statute as
"INA 214(b)". The published text is numbered "8 U.S.C. 1184(b)". A user typing
the first will never lexically match the second.
[`ina_crosswalk.py`](src/foga/sources/ina_crosswalk.py) maps the principal
provisions of INA Titles I–III, and every statute chunk carries both citations.
This single table is responsible for a large share of retrieval quality on
statute questions.

### 4. Verification is mechanical, not promised

The prompt tells the model to quote verbatim and cite every sentence. Instruction
is not enforcement. [`verify.py`](src/foga/verify.py) independently re-checks the
output against the retrieved chunks:

- every `[S#]` tag was actually in the context (catches invented sources)
- every quoted span appears in the chunk it is attributed to, compared after
  whitespace/punctuation normalization (catches fabricated quotes — the most
  dangerous failure, because a plausible fake quote reads exactly like a real one)
- every sentence carries a citation (catches unsupported assertions smuggled
  between grounded ones)

An answer that fails is **relabelled and shown with the failure attached**, not
silently dropped — hiding it would make the failure invisible. This turns "did it
hallucinate?" into a string comparison, and gives the evaluation a hallucination
rate that does not depend on an LLM judge's opinion.

### 5. Authority-aware ranking

The first evaluation run exposed something the design had declared but never
enforced. The USCIS Policy Manual and 9 FAM **systematically outranked the
regulations they paraphrase** — not because they are better answers, but because
they are written in the same plain English as the user's query, while 8 CFR says
the same thing in drafted statutory language. That is a lexical-match advantage
with nothing to do with which source is binding.

`Retriever._authority_bonus` applies a small post-rerank prior by authority rank
(statute > regulation > guidance > notices). It is deliberately small — it breaks
ties between comparably relevant passages and cannot promote an irrelevant
statute over an on-point Policy Manual chapter. `retrieval.authority_bonus: 0.0`
ablates it, and the ablation table below reports both.

### 6. Abstention is a feature

If the best reranked passage scores below threshold, the system refuses. Three
gold-set items are deliberately out-of-corpus (a USCIS processing time, a BIA
holding, a question about Canadian law) and the evaluation scores abstention in
**both** directions — refusing a question the corpus *does* cover is also a
failure, not a safe default.

---

## Measured results

Corpus as built: **2,224 documents → 13,126 chunks** (8 CFR 4,745 · USCIS PM
3,405 · 9 FAM 2,566 · INA 1,809 · Federal Register 601), 19.2M characters.

Retrieval ablation over the 39 gold items that name a specific provision
(`make eval-retrieval`, free). Scored strictly: a hit means **the exact gold
provision** appeared at that rank.

| Config | R@1 | R@3 | R@5 | R@8 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense only | 0.154 | 0.410 | 0.462 | 0.538 | 0.300 | 0.333 |
| BM25 only | 0.103 | 0.256 | 0.282 | 0.410 | 0.200 | 0.207 |
| hybrid (RRF) | 0.103 | 0.436 | 0.538 | 0.641 | 0.269 | 0.325 |
| hybrid + rerank | 0.128 | 0.436 | 0.564 | 0.667 | 0.312 | 0.363 |
| **hybrid + rerank + authority** | **0.154** | **0.564** | **0.641** | **0.769** | **0.375** | **0.428** |

Each stage earns its place: RRF adds +10 points of R@8 over the better single
retriever, reranking adds ~3 more, and the authority prior adds ~10.

**Read these numbers honestly**, and say so in the report:

- They were produced with the **free local `bge-small` embeddings**, not
  `text-embedding-3-small`. Rebuild with `make index` and re-run to get the
  numbers for your actual configuration.
- The metric is a **lower bound on usefulness**. Of the 9 items where the gold
  provision missed the top 8, 8 were cases where the Policy Manual or 9 FAM
  chapter that *correctly answers the question* crowded out the binding provision
  the gold label happens to name. Only one (`visa-vs-status`) was a genuine
  topical miss. A metric that credited "retrieved something that answers it"
  would read much higher — and would be much less informative.
- R@1 of 0.154 is the honest headline: the system usually needs several passages
  in context, which is exactly why `final_k` is 8 and not 1.

---

## Repository layout

```
config.yaml               all tunable settings; every script reads this
src/foga/
  config.py               config loading
  schema.py               Document / Chunk — the one shared vocabulary
  llm.py                  OpenAI + Ollama drivers, embeddings, cost accounting
  sources/                one adapter per authority
    http.py               polite cached HTTP (+ the fam.state.gov TLS fix)
    ecfr.py  usc.py  fam.py  uscis_pm.py  fedreg.py
    ina_crosswalk.py      INA ↔ 8 U.S.C. section mapping
  chunk.py                structure-aware chunking + the numbering stack machine
  index.py                FAISS + BM25, legal-identifier-safe tokenizer
  retrieve.py             hybrid → RRF → cross-encoder rerank → abstain
  prompts.py              all prompts, versioned in one place
  factcheck.py            the workflow pipeline
  verify.py               mechanical citation + quote verification
  agent.py                tool-calling research agent
  evaluation/metrics.py   retrieval + generation metrics
scripts/
  download.py  build_index.py  ask.py  research.py  evaluate.py
  verify_goldset.py  list_models.py
app/streamlit_app.py      demo UI
eval/goldset.jsonl        42 hand-written claims with gold verdicts and citations
tests/test_chunk.py       regression tests for the numbering machine
```

---

## Mapping to the course syllabus

| Week | Topic | Where it shows up |
|---|---|---|
| 6 | Prompt engineering | [`prompts.py`](src/foga/prompts.py) — decomposition, structured JSON schemas, role prompting, explicit authority hierarchy, forced abstention |
| 7 | **RAG** | The whole system: chunking, embeddings, vector + lexical search, RRF, reranking, grounding, and a documented catalogue of failure modes |
| 8 | Agentic systems | [`agent.py`](src/foga/agent.py) — four tools, a multi-step research loop, compared head-to-head against the fixed workflow |
| 12 | Evaluation | [`evaluate.py`](scripts/evaluate.py) — retrieval ablation (dense / BM25 / RRF / rerank), verdict accuracy, citation precision, quote fidelity, hallucination rate, abstention precision & recall |
| 13 | Inference & scaling | Cost and latency tracked per call; small-vs-large model ablation via `--provider ollama` |

---

## Limitations

Stated plainly, because a fact-checker that oversells itself is the thing it is
supposed to prevent.

- **No case law.** No BIA, AAO or federal court decisions. Much of immigration law
  lives in precedent this corpus cannot see.
- **A snapshot.** The Federal Register currency check flags recent activity, but
  the indexed text is as of the download date.
- **Federal only.** No consular post practice, no state law, no USCIS operational
  data (processing times, fees as actually charged).
- **The gold set is hand-written** by the project team, not by an attorney. Run
  `make verify-gold` to confirm each gold citation exists in the corpus; the legal
  correctness of each label still needs human review before the numbers are quoted.
- **Retrieval ceiling.** If the governing provision is not retrieved, no amount of
  prompting recovers it. The ablation table quantifies how often that happens.
- **Not legal advice.** Really.
