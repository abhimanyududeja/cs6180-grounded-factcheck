# Running this locally

Everything needed to run the demo is in this folder, including the prebuilt search
index, so there is no corpus to download and nothing to rebuild.

You need to install two things: Ollama with the model, and the Python dependencies.
Budget about 20 minutes, most of it downloading.

## 1. Ollama and the model

```bash
brew install ollama
ollama serve
```

Leave that running in its own terminal window. In a second terminal:

```bash
ollama pull qwen3:8b
```

That is roughly 5 GB. It is the generator; nothing works without it.

## 2. Python

Requires Python 3.12. Newer versions have patchy wheels for faiss and torch.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Roughly 3 GB, mostly PyTorch, which the cross-encoder reranker needs.

## 3. Run it

```bash
.venv/bin/streamlit run app/streamlit_app.py --server.port 8502
```

Opens at http://localhost:8502

## Check it works

Set Mode to **Fact-check**, leave "Decompose compound claims" **unchecked**, and try:

```
An F-1 student may work up to 20 hours per week on campus while school is in session.
```

Expect **SUPPORTED**, citing 8 CFR 214.2(f)(9)(i).

**The first query takes a minute or two.** Ollama loads the model and
sentence-transformers downloads the cross-encoder on first use. After that a check runs
in roughly 20 to 40 seconds. Run one before presenting so the model is warm.

## The four modes

| Mode | What it does |
|---|---|
| **Fact-check** | The evaluated pipeline. Verdict, citations, grounding grade. |
| **Research agent** | Tool-calling agent. Retrieval-backed but not quote-verified. |
| **Retrieval only** | No LLM. The ranked passages and their scores. Instant. |
| **Evaluation results** | The measured results, read from `reports/`. No LLM. |

## What is not in this folder

- `.venv` — you create it in step 2
- `data/raw` and `data/processed` — the source documents and intermediate files. Only
  needed to rebuild the index, which you do not need to do. `data/index` is included.
- `.env` — only needed for `--provider openai` in the evaluation harness. The demo is
  local only and does not use it.

## If something breaks

**"Ollama model not found"** — `ollama pull qwen3:8b`, and check `ollama serve` is still
running in the other terminal.

**Port already in use** — something else is on 8502. Use `--server.port 8503`.

**A query hangs for minutes** — check the `ollama serve` terminal for errors. First run
after boot is always slow.

**Verdicts look wrong** — check that "Decompose compound claims" is unchecked. With it on,
accuracy drops about 9 points and compound claims come back PARTIALLY_SUPPORTED. That is
a measured finding, not a bug, but it is not the configuration to demo.
