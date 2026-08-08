# Final report outline

A scaffold for the Week 14 PDF. Each section names the command that produces the
evidence for it, so nothing has to be reconstructed from memory the night before.

Target: 8–12 pages. Figures and tables carry the argument; prose explains what
they mean and why the decision was made.

---

## 1. Problem and motivation (~1 page)

- Who this is for: international students and foreign workers, who currently rely
  on forum posts, word of mouth, and chatbots answering from stale memory.
- Concrete failure to lead with: ask a general chatbot the STEM OPT extension
  length and it may say 17 months (the pre-2016 rule), 24 (correct), or 36
  (never true) — with no citation and no way for the reader to tell which.
- Why immigration law suits grounded RAG: public authoritative corpus, answers
  have exact addresses, hallucination is mechanically detectable, stakes are real.
- Scope: federal immigration law only. State the limits here, not in a footnote.

## 2. System overview (~1 page)

- Architecture figure: sources → chunking → hybrid index → retrieval →
  generation → verification → synthesis.
- The two modes (fixed workflow vs. tool-using agent) and when each is right.
- Table of the five sources with document and chunk counts.
  → `make stats` and `python scripts/build_index.py --dry-run`

## 3. Data acquisition (~1.5 pages)

This section is more interesting than it sounds and is worth real space, because
"there is no dataset" was the starting condition.

- Table of the five sources and the access method for each.
- The distinction between bulk endpoints (eCFR, OLRC, Federal Register, the FAM
  tree API) and the one source with no bulk file (USCIS Policy Manual).
- The ethics and mechanics of the Policy Manual fetch: sitemap-driven
  enumeration, `Crawl-delay: 10` honored from their own robots.txt, identifying
  User-Agent, fetch-once caching, public-domain content.
- Worth a paragraph: `fam.state.gov` serves an incomplete TLS chain. The fix was
  to route verification through the OS trust store so the missing intermediate is
  fetched via AIA — *not* to disable verification. Small detail, but it is exactly
  the kind of shortcut that quietly undermines a project about trustworthiness.

## 4. Chunking (~1.5 pages) — **the strongest technical section**

- Lead with the number: 8 CFR § 214.2 is one section of 700,000 characters
  spanning every nonimmigrant category.
- Show what fixed-window chunking produces (~350 chunks all citing "8 CFR
  § 214.2") versus the stack machine (204 distinct citable provisions with deep
  links).
- Walk through the numbering stack machine with a worked example.
- Both bugs, honestly: the section-contents table poisoning the stack, and the
  `(i)` roman-vs-alpha ambiguity relocating 166 H-visa chunks. Show the
  regression tests. Bugs found and fixed with tests read as competence.
  → `make test`
- **Suggested figure:** citation-count distribution before vs. after.

## 5. Retrieval (~1.5 pages)

- Why hybrid: the vocabulary gap (dense) vs. exact identifiers (sparse), with a
  real example of each failing alone.
- Why RRF rather than a weighted score sum.
- The custom tokenizer that keeps `214.2(f)(9)`, `I-765` and `H-1B` intact.
- The INA ↔ 8 U.S.C. crosswalk and why it matters so much.
- Cross-encoder reranking, and the abstention threshold that falls out of it.
- **Ablation table** — the centerpiece.
  → `make eval-retrieval`

| Config | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| dense only | | | | | |
| BM25 only | | | | | |
| hybrid (RRF) | | | | | |
| hybrid + rerank | | | | | |

Discuss *which* queries each configuration wins on, not just the averages. The
per-item detail is in the eval JSON.

## 6. Generation and grounding (~1.5 pages)

- Prompt design: forbidding closed-book answers, the authority hierarchy,
  `NOT_ADDRESSED` as a first-class verdict, verbatim-quote requirement.
- Sub-claim decomposition, with the compound gold-set item as the worked example
  (12 months OPT ✅ + 36-month STEM extension ❌ → one verdict blurs them, two
  verdicts isolate the false half).
- The verification layer: what it checks, what it caught, and the enforcement
  behaviour when a check fails.
- **Suggested figure:** a screenshot of the UI grounding panel with a failed
  quote check, if you can produce one.

## 7. Evaluation (~2 pages)

- Gold set: 42 claims, composition by category and difficulty, and how it was
  built. Be explicit that it is hand-written and not attorney-reviewed.
  → `make verify-gold` confirms every gold citation resolves in the corpus
- Metrics and why each one is there — especially why verdict accuracy alone is
  insufficient (a system can guess the right verdict while citing the wrong
  statute, which is worse than useless to someone who has to act on it).
- Results table.
  → `make eval`
- **Ablations to report:**
  - retrieval configuration (§5)
  - decomposition on/off → `--no-decompose`
  - model size → `--provider ollama` vs OpenAI, same retrieval context
  - workflow vs. agent on the multi-hop questions
- **Error analysis** — the most valuable part. Take 5–8 misses and say what went
  wrong: retrieval miss, generation error, or a bad gold label. Some will be bad
  gold labels; say so.
- Cost and latency per query.

## 8. Limitations and future work (~0.5 page)

- No case law; snapshot corpus; federal only; hand-written gold set; retrieval
  ceiling.
- Future: BIA/AAO decisions from EOIR, scheduled re-indexing, per-user context
  (visa type, dates) to make answers situation-specific, calibrated confidence.

## 9. Contributions and reproduction (~0.5 page)

- Who did what.
- Exact reproduction commands and the runtime/cost of each.

---

## Demo script (Week 14, ~5 minutes)

Rehearse this. Have the app already running and the index already warm — do not
spend demo time loading the reranker.

1. **A true claim** — "F-1 students can work 20 hours per week on campus."
   Show the verdict, then open the sources panel to show it cites
   8 CFR § 214.2(f)(9)(i) and deep-links to that paragraph. *(~40 s)*
2. **A false claim** — "The STEM OPT extension is 36 months."
   CONTRADICTED, with the regulation saying 24. *(~30 s)*
3. **A compound claim** — the OPT + 36-month STEM one. Show decomposition
   isolating the true half from the false half. This is the money shot. *(~60 s)*
4. **An out-of-corpus claim** — "The Nebraska Service Center processes I-765 in
   45 days." Show it refuse rather than invent a number. *(~30 s)*
5. **The grounding panel** — show quotes being verified character-by-character
   against source text, ideally including one that fails. *(~45 s)*
6. **Agent mode** — "Can I start STEM OPT while my H-1B petition is pending?"
   Show the multi-step trace: three searches the workflow could not have planned
   in advance. *(~60 s)*
7. **The ablation table** — one slide, the retrieval numbers. *(~30 s)*

Have a fallback: pre-recorded output or `--provider ollama` in case the network
fails.
