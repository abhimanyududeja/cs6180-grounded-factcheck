---
title: "Grounded RAG Fact-Checking"
subtitle: "U.S. Immigration and Tax Law, CS 6180 Final Project"
author: "Abhimanyu Dudeja · Kashish Rahulbhai Khatri · Prasanna Adarsh Kolli"
date: "August 2026"
---

## The problem

Ask a chatbot how long the STEM OPT extension runs.

Sometimes 17 months (the pre-2016 rule). Sometimes 24 (correct). Sometimes 36 (never true).

- It cannot tell you which.
- It cannot cite a regulation.
- For someone deciding whether to accept a job offer, that is not a small error.

## What we built

A fact-checker that **cannot answer from memory**.

```
claim -> retrieve -> grounded verdict -> mechanical verification -> verdict + provisions
```

Returns SUPPORTED, CONTRADICTED, PARTIALLY_SUPPORTED or NOT_ADDRESSED, with the exact
provisions and quotes it used.

Runs entirely locally. No API key, no per-run cost.

## Why immigration and tax

Both have a property most domains lack: **answers have exact addresses.**

8 CFR 214.2(f)(9)(i) either says what you claim or it does not. That makes hallucination
*mechanically detectable* rather than a matter of opinion.

They also overlap: residence is both an immigration question and a tax question.

## Corpus: 14,726 chunks

| Source | Domain |
|---|---|
| 8 CFR, INA / 8 U.S.C. | immigration |
| 9 FAM, USCIS Policy Manual | immigration |
| Federal Register | immigration |
| IRS Pub 17, 501, 519, 970, Form 1040 | tax |

Obtained through bulk APIs and official sitemaps, not scraping. Each URL fetched once,
ever, and cached.

## The chunking problem

**8 CFR 214.2 is one section of 700,000 characters.**

It covers every visa category from A-1 diplomats to Q cultural visitors.

A fixed-window chunker produces hundreds of chunks all citing "8 CFR 214.2". The system
cannot say *which part* it used.

We run a stack machine over the statutory numbering, (a) then (1) then (i), and recover
the full path: **8 CFR 214.2(f)(9)(ii)**.

## Verification is mechanical, not promised

The prompt says quote verbatim. **Instruction is not enforcement.**

A separate pass checks by string comparison:

- every tag was actually in the retrieved context
- every quoted span appears in the passage it is attributed to
- every sentence carries a citation

Failures are relabelled and **shown**, not dropped. Hiding them would make them invisible.

## Retrieval ablation

39 provision-level gold items. A hit means the *exact* gold provision at that rank.

| config | R@8 | MRR |
|---|---|---|
| dense only | 0.538 | 0.300 |
| BM25 only | 0.385 | 0.184 |
| hybrid (RRF) | 0.641 | 0.270 |
| hybrid + rerank | 0.667 | 0.312 |
| **+ authority prior** | **0.769** | **0.375** |

Every stage earns its place. Fusion +10 points, rerank +3, authority prior +10.

## The authority prior

The Policy Manual and 9 FAM **systematically outranked the regulations they paraphrase.**

Not because they are better answers. Because they are written in the same plain English
as the query, while the regulation says the same thing in drafted statutory language.

A lexical-match advantage with nothing to do with which source is binding.

Small post-rerank prior: statute > regulation > guidance > notices.

## Verdict results: 73 claims

| config | accuracy | residual halluc. | latency |
|---|---|---|---|
| no retrieval | 0.205 | 1.000 | 7.3 s |
| simple retrieval | 0.452 | **0.000** | 38.8 s |
| **full, no decomposition** | **0.562** | 0.314 | 49.7 s |
| full | 0.452 | 0.292 | 121.5 s\* |

\* `full` ran while the retrieval ablation shared the machine, so its latency is not
comparable. Its real per-claim cost is roughly double the row above.

**Grounding beats memory 3x.** Residual hallucination of 1.000 for the baseline means
every decisive verdict it gave was ungrounded. That is what answering from memory means.

## The finding: decomposition hurts

| | accuracy |
|---|---|
| full, no decomposition | **0.562** |
| full, decomposition on | 0.452 |

**-11.0 points, and nearly double the latency.**

Mechanism: an 8B model splits on clause boundaries into fragments that are not checkable
assertions. Retrieval on a fragment returns nothing. One weak fragment propagates to the
whole claim through the synthesis rule.

It looks exactly like a retrieval failure. It is not one.

## The tension we did not smooth over

The **most faithful** configuration is not the **most accurate** one.

- `simple_retrieval`: residual hallucination 0.000, quote fidelity 1.000, accuracy 0.452
- `full_no_decompose`: accuracy 0.562, residual hallucination 0.314

More trustworthy, less useful. Retrieving harder surfaces passages that are topically
close but not on point, and the model quotes them.

Which trade to take depends on whether a wrong answer or a missing answer is worse.

## Local vs frontier model

Same configuration, same 73 claims, only the model changes.

| | `qwen3:8b` | `gpt-5.6-luna` |
|---|---|---|
| accuracy | 0.562 | **0.753** |
| citation precision | 0.603 | 0.603 |
| quote fidelity | 0.811 | **0.991** |
| residual hallucination | 0.314 | **0.000** |
| latency | 49.7 s | **5.1 s** |

**Citation precision is identical.** That isolates the halves: retrieval is unchanged by
the model, so the entire 19-point gap is generation.

The frontier model shipped **zero** ungrounded decisive verdicts.

## Seven silent assumptions

Porting the pipeline to a local model surfaced seven places it assumed a frontier model.

| Assumption | Symptom |
|---|---|
| Reasoning mode off | 315 s/claim; reasoning bled into quotes |
| Schema lists are lists | Crash on `"caveats": "none"` |
| Decomposition returns claims | Fragments; universal abstention |
| `format: "json"` gives right keys | Wrong keys, silent NOT_ADDRESSED |
| Claim survives long context | Answered a different claim |
| Tags are source labels | Verifier could not resolve them |
| Contradiction distinguished from absence | $15,750 did not contradict $15,000 |

## Why that matters here

**Six of the seven fail silently.**

No exception. No warning. A well-formed NOT_ADDRESSED that is indistinguishable from
honest abstention.

Before fixing them, every configuration scored 0.100. The natural conclusion would have
been "an 8B model cannot do this."

That conclusion would have been wrong. **Nothing in the metrics distinguished a
principled abstention from a JSON key mismatch.**

## Limitations

- **73 claims.** One verdict is 1.4 points. Gaps under ~4 points are not real.
- **Gold set is hand-written by us**, not by an attorney. Labels checked against
  retrieved passages, not certified.
- **Prompt was revised after seeing failures on these claims**, so these are
  development numbers, not held-out.
- **No case law**, no state tax law, snapshot corpus.
- **The research-agent mode is retrieval-backed but not quote-verified**, returns no
  verdict, and is not in any number here.
- **Not legal or tax advice.**

## Conclusion

Grounding works, and the measurement says so: **0.205 to 0.562**, residual hallucination
from 1.000 to 0.314.

Two results cut against the design, and we report them as findings:

- Decomposition costs 11 points despite being the most sophisticated stage
- The most faithful configuration is not the most accurate one

An answer is only worth as much as the evidence behind it. That applies to ours too.
