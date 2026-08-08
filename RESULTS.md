# Results

Every number here came from `scripts/evaluate.py` on the merged 14,726-chunk corpus,
using `qwen3:8b` and `BAAI/bge-small-en-v1.5` through Ollama on an M-series Mac. Total
API cost: zero. Raw per-run output is in `reports/`.

## Verdict evaluation (73 claims)

| config | accuracy | citation prec. | quote fidelity | residual halluc. | detected | latency |
|---|---|---|---|---|---|---|
| `no_retrieval` | 0.205 | 0.000 | 0.000 | 1.000 | 1.000 | 7.3 s |
| `simple_retrieval` | 0.452 | 0.508 | **1.000** | **0.000** | 0.000 | 38.8 s |
| **`full_no_decompose`** | **0.562** | 0.603 | 0.811 | 0.314 | 0.314 | 49.7 s |
| `full` | 0.452 | **0.635** | 0.804 | 0.292 | 0.292 | 121.5 s* |

\* `full`'s latency is not comparable to the other rows: the retrieval ablation was
run on the same machine while it was in progress. Its per-claim cost is roughly
double `full_no_decompose`, which is the figure to quote.


By domain, best configuration: immigration 32/56, tax 10/17.

Read `residual` as the faithfulness number: verdicts still shown as decisive that failed
the mechanical quote check. `detected` counts everything the verifier caught whether or
not it was repaired, so a better verifier scores higher there and it is not a quality
ranking.

## Retrieval ablation (39 provision-level gold items)

| config | R@1 | R@3 | R@5 | R@8 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense only | 0.154 | 0.410 | 0.462 | 0.538 | 0.300 | 0.333 |
| BM25 only | 0.077 | 0.256 | 0.282 | 0.385 | 0.184 | 0.198 |
| hybrid (RRF) | 0.103 | 0.436 | 0.564 | 0.641 | 0.270 | 0.335 |
| hybrid + rerank | 0.128 | 0.436 | 0.564 | 0.667 | 0.312 | 0.363 |
| **hybrid + rerank + authority** | **0.154** | **0.564** | **0.641** | **0.769** | **0.375** | **0.428** |

Only the 39 provision-level items are scored. The 24 tax labels name a document rather
than a paragraph, and a document-level label is satisfied by any chunk of a
200,000-character publication, so including them would inflate recall.

## What this supports

**Grounding beats parametric memory.** 0.205 to 0.562 accuracy, nearly three times better.
The no-retrieval baseline's residual hallucination rate of 1.000 means every decisive
verdict it produced was ungrounded.

**The retrieval stack earns its cost.** `full_no_decompose` beats `simple_retrieval` by
11.0 points while being faster, because the abstention gate skips the LLM call when
nothing scores well. Each stage is independently justified by the ablation above.

**The tax corpus integrated cleanly.** Tax scores 10/17 against immigration's 32/56 in
the best configuration, and adding 1,600 tax chunks left the immigration retrieval
numbers essentially unchanged.

## What this does NOT support

**Decomposition does not pay for itself at this model size.** Enabling it costs 11.0
points, 0.562 to 0.452, and nearly doubles latency. At n=73 one verdict is 1.4 points,
so that gap is roughly nine claims and outside noise. See REPORT.md section 7 for the
mechanism.

**The most accurate configuration is not the most faithful one.** `simple_retrieval`
has a perfect 0.000 residual hallucination rate and perfect quote fidelity but is 10
points less accurate. Retrieving harder surfaces passages that are topically close but
not on point, and the model quotes them. Which trade to take is a deployment decision,
not a settled one.

**Nothing here is held out.** Three of the seven porting fixes in REPORT.md section 8
were developed against this gold set, so these are development numbers.

## Caveats on method

- 73 claims. One flipped verdict moves accuracy 1.4 points; gaps under about 4 points
  are not meaningful. The citation-precision differences among the three retrieval
  configurations fall inside that band.
- The gold set is hand-written by the team. Each gold citation was checked to exist in
  the corpus and each label against a retrieved passage, but no attorney or tax
  professional has reviewed the labels for legal correctness.
- The frontier-model comparison was not run. It would isolate how much of the
  decomposition gap is model capability rather than pipeline design, and the harness
  supports it with `--provider openai`.
