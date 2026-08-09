---
title: "Grounded RAG Fact-Checking for U.S. Immigration and Tax Law"
subtitle: "CS 6180 Foundations for Generative AI, Final Project Report"
author: "Abhimanyu Dudeja · Kashish Rahulbhai Khatri · Prasanna Adarsh Kolli"
date: "August 2026"
---

# 1. Problem

Language models answer legal and tax questions fluently and without sources, which is
the combination that makes them unsafe in those domains. Ask a general-purpose model how
long the STEM OPT extension runs and it will answer from memory: sometimes 17 months (the
pre-2016 rule), sometimes 24 (correct), sometimes 36 (never true). It cannot tell you
which, and it cannot cite a regulation. For a student deciding whether to accept a job
offer, that is not a small error.

We built a fact-checker that cannot answer from memory. It takes a claim, retrieves
passages from a fixed corpus of U.S. statutes, regulations and agency guidance, and
returns SUPPORTED, CONTRADICTED, PARTIALLY_SUPPORTED or NOT_ADDRESSED together with the
provisions and exact quotes it relied on. A verification pass then re-checks
mechanically, by string comparison rather than by asking a model, whether the quotes it
used actually appear in the passages it cited.

The measurable question is not whether the answers sound right. It is whether grounding
measurably beats parametric memory, and which parts of the pipeline earn their cost.

## Scope note

The project proposal named sports rules and statistics as the leading candidate domain
while stating that the exact domain was still open. We settled it as U.S. immigration and
tax law.

These two domains have a property sports statistics lack: the primary sources are public,
stable and written to be cited. A verdict can point at 8 CFR 214.2(f)(9)(i) or IRS
Publication 501, and that provision either says what the claim says or it does not, which
makes hallucination mechanically detectable rather than a matter of opinion. The domains
also overlap. Residence is simultaneously an immigration question and a tax question, and
Publication 519 exists precisely to explain the tax consequences of an immigration
status, so a single corpus supports cross-domain claims.

# 2. System design

```
claim
  |- 1. decompose   split into atomic sub-claims (ablated; see section 7)
  |- 2. retrieve    dense + BM25 -> RRF -> cross-encoder rerank -> authority prior
  |- 3. gate        nothing scores well? -> abstain, no LLM call
  |- 4. generate    verdict + explanation + tagged quotes
  |- 5. verify      do the quotes appear in the cited passages? -> downgrade if not
```

Everything runs locally. No API key is required and nothing leaves the machine.

| Layer | Choice | Rationale |
|---|---|---|
| LLM | `qwen3:8b` via Ollama | Runs on a laptop; no key, no per-run cost |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim | Local, fast, no embedding API bill |
| Vector store | FAISS, inner product on unit vectors | Exact cosine search at this corpus size |
| Lexical | BM25 with a legal-identifier-safe tokenizer | Statutory text is full of exact identifiers |
| Fusion | Reciprocal rank fusion, k=60 | Rank-based, so no score calibration needed |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores the fused pool |
| Verification | `difflib` span matching | String comparison, not an LLM's opinion |

Four design decisions carry most of the system's behaviour.

**Structure-aware chunking.** 8 CFR 214.2 is a single section of roughly 700,000
characters covering every nonimmigrant category from A-1 diplomats to Q cultural
visitors. A fixed-window chunker turns it into hundreds of chunks that all cite "8 CFR
214.2", so a verdict cannot say which part of the section it used. Instead a stack
machine walks the statutory numbering, (a) then (1) then (i) then (A), and recovers each
paragraph's position in the hierarchy. Chunks then cite 8 CFR 214.2(f)(9)(ii) and deep
link to that paragraph. Two ambiguities in that machine are regression-tested: the
section contents table at the top of 214.2 lists (a) through (w) and convinces a naive
parser the section has already reached (w) before the body begins, and (i) is both the
ninth letter and the first roman numeral.

**Hybrid retrieval.** Dense embeddings handle the vocabulary gap, where a user asks about
working off campus and the regulation says "employment other than on the school's
premises". BM25 handles exact identifiers such as 214.2(f)(10) and Form I-765, which
embeddings handle badly because 214.2(f)(10) and 214.2(f)(11) sit almost on top of each
other in vector space and mean different things.

**Authority-aware ranking.** The USCIS Policy Manual and 9 FAM systematically outranked
the regulations they paraphrase, because they are written in the same plain English as
the query while the regulation says the same thing in drafted statutory language. That is
a lexical-match advantage with nothing to do with which source is binding. A small
post-rerank prior by authority rank (statute, then regulation, then guidance, then
notices) corrects it. Section 5 measures what it is worth.

**Mechanical verification.** The prompt tells the model to quote verbatim and tag every
sentence. Instruction is not enforcement. The verifier independently re-checks that every
tag was in the retrieved context, that every quoted span appears in the passage it is
attributed to after whitespace and punctuation normalization, and that every sentence
carries a citation. An answer that fails is relabelled and shown with the failure
attached rather than silently dropped. This turns "did it hallucinate" into a string
comparison rather than a judgment call.

# 3. Corpus

Five immigration sources and five tax sources, 2,229 documents, **14,726 chunks**.

| Source | Domain | How it is obtained |
|---|---|---|
| 8 CFR | immigration | eCFR API, one unauthenticated request for the full title |
| INA / 8 U.S.C. | immigration | Office of the Law Revision Counsel bulk release-point ZIP |
| 9 FAM | immigration | The site's own JSON tree endpoint, then one fetch per section |
| USCIS Policy Manual | immigration | URLs from the official sitemap, at the `robots.txt` crawl delay |
| Federal Register | immigration | Public JSON API, for currency checks |
| IRS Pub 17, 501, 519, 970 | tax | Stable PDFs from irs.gov/pub/irs-pdf |
| IRS Form 1040 instructions | tax | Same |

Only the USCIS Policy Manual requires page-by-page fetching, because USCIS publishes no
bulk archive. URLs come from their sitemap rather than from crawling, the rate is the one
their `robots.txt` specifies, each URL is requested at most once ever and cached, and the
client identifies itself with a contact address.

Extraction, not fetching, was the hard part on the tax side. IRS publications are typeset
in narrow columns, so text extraction returns words broken mid-hyphen at every line end
along with printer's marks and running footers. Left alone that text is embedded and
retrieved as though it were substantive guidance. The loader strips page furniture,
repairs hyphenated line breaks and rejoins wrapped column lines before chunking. IRS
publications are chunked by heading rather than by the statutory stack machine, since
they are prose explainers rather than drafted law.

# 4. Evaluation method

**73 claims**: 56 immigration, 17 tax. 44 SUPPORTED, 18 CONTRADICTED, 10 NOT_ADDRESSED,
1 PARTIALLY_SUPPORTED. Ten are deliberately unanswerable from the corpus, including a
USCIS processing time, a question about Canadian tax law and a claim about a future tax
year, so that abstention is scored in both directions. Refusing a question the corpus does
cover is a failure too, not a safe default.

Four configurations, each an ablation of the same pipeline rather than a separate
implementation, so a difference between rows is attributable to the setting that changed:

- **`no_retrieval`**: the model answers from memory. No evidence, no verification.
- **`simple_retrieval`**: dense top-k only. No fusion, rerank, authority prior or
  decomposition.
- **`full_no_decompose`**: hybrid, rerank, authority prior and verification, but the
  claim is checked as one unit.
- **`full`**: the above plus claim decomposition into atomic sub-claims.

The first two are the baselines the project proposal calls for.

## Two metric decisions

**Gold citations are tagged by granularity.** The immigration labels name an exact
provision (8 CFR 214.2(f)(9)(i)); the tax labels name a document (IRS Publication 501).
A document-level label is satisfied by any chunk of a 200,000-character publication, so
mixing the two would inflate recall and make the retrieval numbers incomparable. The
retrieval ablation therefore scores only the 39 provision-level items. All 73 count in
the verdict evaluation, where citation granularity is irrelevant.

**Hallucination is two numbers, not one.** A single rate conflates what the generator
produced with what the user is shown.

- **`residual`**: verdicts still presented as SUPPORTED or CONTRADICTED that failed the
  mechanical check. This is the faithfulness number, and lower is better.
- **`detected`**: everything the verifier caught, repaired or not. A configuration with a
  better verifier scores higher here, so it is not a quality ranking.

A metric that was never measured reports `n/a`, never `0.0`. Reporting an unmeasured
baseline as perfectly faithful would invert the comparison the project exists to make.

# 5. Results: retrieval

Scored strictly over the 39 provision-level gold items on the merged 14,726-chunk index.
A hit means the exact gold provision appeared at that rank. No LLM is involved, so this
is free and fast to re-run.

| config | R@1 | R@3 | R@5 | R@8 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense only | 0.154 | 0.410 | 0.462 | 0.538 | 0.300 | 0.333 |
| BM25 only | 0.077 | 0.256 | 0.282 | 0.385 | 0.184 | 0.198 |
| hybrid (RRF) | 0.103 | 0.436 | 0.564 | 0.641 | 0.270 | 0.335 |
| hybrid + rerank | 0.128 | 0.436 | 0.564 | 0.667 | 0.312 | 0.363 |
| **hybrid + rerank + authority** | **0.154** | **0.564** | **0.641** | **0.769** | **0.375** | **0.428** |

Every stage earns its place. Fusion adds 10 points of R@8 over the better single
retriever, reranking adds 3 more, and the authority prior adds another 10 and is the
single largest contributor after fusion.

Adding 1,600 tax chunks to a 13,126-chunk immigration index left these numbers
essentially unchanged, which is the desired outcome: the tax corpus does not crowd out
immigration retrieval.

Read R@1 of 0.154 as the honest headline. The system usually needs several passages in
context, which is why the pipeline retrieves 8 and not 1. The metric is also a lower
bound on usefulness: in most of the cases where the gold provision missed the top 8, the
Policy Manual or FAM chapter that correctly answers the question crowded out the binding
provision the gold label happens to name.

# 6. Results: verdicts

73 claims, four configurations, `qwen3:8b` throughout. Total cost: zero.

| config | accuracy | citation prec. | quote fidelity | residual halluc. | detected | latency |
|---|---|---|---|---|---|---|
| `no_retrieval` | 0.205 | 0.000 | 0.000 | 1.000 | 1.000 | 7.3 s |
| `simple_retrieval` | 0.452 | 0.508 | **1.000** | **0.000** | 0.000 | 38.8 s |
| **`full_no_decompose`** | **0.562** | 0.603 | 0.811 | 0.314 | 0.314 | 49.7 s |
| `full` | 0.452 | **0.635** | 0.804 | 0.292 | 0.292 | 121.5 s* |

\* `full`'s latency is not comparable to the other rows: the retrieval ablation was
run on the same machine while it was in progress. Its per-claim cost is roughly
double `full_no_decompose`, which is the figure to quote.


By domain, for the best configuration: immigration 32/56, tax 10/17. The tax corpus
performs on par with the immigration corpus it was merged into, which is the evidence
that the merge worked rather than simply not breaking anything.

**Grounding beats memory decisively.** Accuracy goes from 0.205 to 0.562, three times
better, and residual hallucination falls from 1.000 to 0.314. The no-retrieval baseline's
residual rate of 1.000 is worth stating plainly: every decisive verdict it produced was
ungrounded, which is exactly what answering from memory means. This is the project's
central claim and it holds.

**The retrieval stack earns its cost.** `full_no_decompose` beats `simple_retrieval` by
11.0 points, 0.562 against 0.452, and does so while being *faster* (49.7 s against 38.8 s)
because the abstention gate skips the LLM call when nothing scores well. Hybrid fusion,
reranking and the authority prior are each justified by section 5 and together they move
the end-to-end number as well.

**There is a real tension between accuracy and faithfulness.** `simple_retrieval` has a
perfect residual hallucination rate of 0.000 and a perfect quote fidelity of 1.000, but
it is 10 points less accurate. It is more trustworthy and less useful. The higher
residual rate of the stronger configurations is not noise to be explained away: retrieving
more aggressively surfaces passages that are topically close but not quite on point, and
the model quotes them. Which trade a deployment should take depends on whether a wrong
answer or a missing answer is worse, and for immigration advice that is not obvious.

# 7. The decomposition finding

Claim decomposition, splitting a compound claim into atomic sub-claims before checking
each, is the one component that does not survive contact with an 8B model.

| | accuracy | change |
|---|---|---|
| `full_no_decompose` | 0.562 | |
| `full` (decomposition on) | 0.452 | **-0.110** |

Enabling it costs 11.0 points and roughly doubles the LLM calls per claim. At n=73 one
flipped verdict is worth 1.4 points, so a 12-point gap is roughly nine claims and well
outside noise.

The mechanism is visible in the per-item output. The synthesis rule is that any
CONTRADICTED sub-claim makes the whole claim contradicted, and if the decisive sub-claims
are NOT_ADDRESSED then so is the whole. That is sound when the decomposition is good. On
a smaller model it often is not: the model splits on clause boundaries and produces
grammatical fragments that are not checkable assertions, retrieval on a fragment returns
nothing on point, and the sub-claim comes back NOT_ADDRESSED. One weak fragment then
propagates to the whole claim. The failure looks exactly like a retrieval failure and is
not one.

A guard now rejects a decomposition whose pieces are too short to be assertions and falls
back to checking the claim whole, which removed the worst of it. The residual 12-point
gap is decomposition working as designed on splits that pass the guard, and it is still
a net loss at this model size.

The honest reading is that decomposition is a capability-dependent optimization. It
presumes the model can both split a claim well and resolve each piece independently, and
an 8B model does neither reliably. On a frontier model it may well pay for itself, which
is a comparison this project has the harness to run but did not, because running it costs
money and the local path was the design constraint.

## 7.1 The same configuration on a frontier model

`full_no_decompose`, unchanged, run against `gpt-5.6-luna` over the same 73 claims:

| | `qwen3:8b` (local) | `gpt-5.6-luna` |
|---|---|---|
| accuracy | 0.562 | **0.753** |
| citation precision | 0.603 | 0.603 |
| quote fidelity | 0.811 | **0.991** |
| residual hallucination | 0.314 | **0.000** |
| latency per claim | 49.7 s | **5.1 s** |
| cost | none | $4.53 |

Three things stand out. Accuracy rises 19 points. Quote fidelity reaches 0.991 and the
residual hallucination rate reaches zero, meaning every decisive verdict the frontier
model shipped was backed by a quote that actually appears in the passage it cited. And it
is ten times faster, because a hosted model on dedicated hardware beats an 8B model
sharing a laptop with the retrieval stack.

Citation precision is identical at 0.603, which is the useful detail. That metric depends
on retrieving the right provision, not on the model reading it, so it isolates the two
halves: **retrieval quality is unchanged by the model, and the entire 19-point gap comes
from generation.** The retrieval work in section 5 stands on its own.

What this does not settle is whether decomposition pays for itself on a stronger model.
That comparison needs `full` run against the same model, and it was not run: the first
frontier configuration cost $4.53 rather than the $0.30 the original estimate assumed, and
decomposition roughly doubles the calls per claim. The honest position is that section 7
measures decomposition on an 8B model only, and the frontier case remains open.

# 8. Porting a frontier-model pipeline to a local model

The retrieval engine was originally written and tuned against a frontier model. Running
it on `qwen3:8b` surfaced seven places where it silently assumed one. All seven are worth
recording because of the shape they share.

| # | Assumption | Symptom |
|---|---|---|
| 1 | Reasoning mode is off | 315 s per claim, and reasoning text bled into quoted spans so the verifier rejected correct answers |
| 2 | Fields the schema calls lists are lists | Hard crash when the model returned `"caveats": "none"` |
| 3 | Decomposition returns checkable claims | Claims shredded into fragments; universal abstention |
| 4 | `format: "json"` yields the right keys | Model returned `claim`/`sources`/`analysis`; verdict silently became NOT_ADDRESSED |
| 5 | The claim survives a long context | The claim sat above 18,000 characters of passages and the model answered a different one |
| 6 | Evidence tags are source labels | Model emitted citations, so the verifier could not resolve the tag |
| 7 | Contradiction is distinguished from absence | A source stating $15,750 did not register as contradicting a claimed $15,000 |

Six of the seven fail *quietly*. There is no exception and no warning: the pipeline
produces a well-formed NOT_ADDRESSED that is indistinguishable from honest abstention.
Before these were fixed, all three retrieval configurations scored 0.100 on a 10-claim
sample, and the natural conclusion would have been that an 8B model cannot do this task.
That conclusion would have been wrong.

That is worth stating in a report about a fact-checker, because it is the same failure
the system is built to prevent, one level up. A verdict of "I cannot verify this" is
trustworthy only if it is reached for the right reason, and nothing in the metrics
distinguished a principled abstention from a JSON key mismatch.

The fixes for (4) and (7) are the two most transferable. Ollama supports constrained
decoding against a JSON Schema, which enforces the output shape rather than requesting
it. And a verdict prompt needs to say explicitly that a source stating a different value
for the same quantity contradicts the claim, because "the sources do not mention $15,000"
is a defensible-sounding reading of a passage that says $15,750.

# 9. Limitations

Stated plainly, because a fact-checker that oversells itself is the thing it exists to
prevent.

- **73 claims is a small sample.** One flipped verdict moves accuracy by 1.4 points.
  Gaps under about 4 points should not be read as real. The 11.0-point and 11.0-point gaps
  in sections 6 and 7 are outside that, the differences among citation precision figures
  largely are not.
- **The gold set is hand-written by the team**, not by an attorney or a tax
  professional. Each gold citation was checked to exist in the corpus and each label was
  checked against a retrieved passage, but legal correctness has not been reviewed by
  anyone qualified to certify it.
- **The tax labels are document-level**, not provision-level, so they contribute nothing
  to the retrieval metrics and are excluded from them.
- **No case law.** No BIA, AAO or federal court decisions, and much of immigration law
  lives in precedent this corpus cannot see. No state tax law.
- **A snapshot.** The Federal Register check flags recent activity, but the indexed text
  is current only as of the download date.
- **The prompt was revised after seeing failures on these same claims.** Fixes (5), (6)
  and (7) in section 8 were developed against the gold set they are evaluated on, so the
  verdict numbers are development numbers rather than held-out ones.
- **The frontier comparison is partial.** Section 7.1 runs `full_no_decompose` against
  `gpt-5.6-luna`, but not `full`, so whether decomposition pays for itself on a stronger
  model is still open. The harness supports it with a single flag; the cost is the
  constraint, not the code.
- **Not legal or tax advice.** A course artifact over public documents.

# 10. Work split

Per the proposal: **Kashish** on retrieval, covering the corpus, chunking, embeddings,
the vector store and the retrieval comparison in section 5. **Abhimanyu** on generation,
covering prompting, verdict logic, the verification pass and the answer-side metrics in
sections 6 and 8. **Prasanna** on integration and evaluation, covering the labeled gold
set, the end-to-end pipeline, the demo interface and the evaluation harness. Report, demo
and write-up split equally.

# 11. Conclusion

Grounding works, and the measurement says so: 0.205 to 0.562 accuracy, and a residual
hallucination rate that falls from every-decisive-verdict-ungrounded to roughly one in
three. The retrieval stack is justified component by component in section 5 and end to
end in section 6.

Two results cut against the design. Decomposition costs 11 points on an 8B model despite
being the most sophisticated stage in the pipeline, and the most faithful configuration
is not the most accurate one. Both are reported here as findings rather than smoothed
over, on the same principle the system itself runs on: an answer is only worth as much as
the evidence behind it, and saying so is the point.
