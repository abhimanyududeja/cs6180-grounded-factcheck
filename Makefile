PY := .venv/bin/python
STREAMLIT := .venv/bin/streamlit

# Python used to create the venv. 3.12 is preferred (3.13+ has patchy faiss/torch
# wheels). Override on any machine: make setup PYTHON=/path/to/python3.12
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3 2>/dev/null)

.PHONY: help setup download download-full index index-local test ask app eval eval-retrieval clean stats verify-gold

help:
	@echo "FOGA — Grounded RAG fact-checker for US immigration law"
	@echo ""
	@echo "  make setup            create the venv and install dependencies"
	@echo "  make download         fetch the corpus (USCIS Policy Manual subset, ~45 min)"
	@echo "  make download-full    fetch everything incl. the full Policy Manual (~3.6 h)"
	@echo "  make index            chunk + embed + build the hybrid index (~\$$0.10)"
	@echo "  make index-local      same, with free local embeddings (no API key)"
	@echo "  make test             run the chunker regression tests"
	@echo "  make verify-gold      check that every gold citation resolves in the corpus"
	@echo "  make eval-retrieval   retrieval metrics + ablation (free, no API)"
	@echo "  make eval             full evaluation incl. generation (~\$$0.30)"
	@echo "  make app              launch the Streamlit demo"
	@echo "  make stats            what is on disk"
	@echo ""
	@echo "  make ask Q=\"F-1 students can work 20 hours per week\""

setup:
	@test -n "$(PYTHON)" || (echo "No python3 found. Install Python 3.12."; exit 1)
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo ""
	@echo "Next: put your OpenAI key in .env, set FOGA_CONTACT_EMAIL, then: make download"

download:
	$(PY) scripts/download.py

download-full:
	$(PY) scripts/download.py --full

index:
	$(PY) scripts/build_index.py

index-local:
	$(PY) scripts/build_index.py --local-embed

test:
	$(PY) tests/test_chunk.py

verify-gold:
	$(PY) scripts/verify_goldset.py

ask:
	@test -n "$(Q)" || (echo 'usage: make ask Q="your claim"'; exit 1)
	$(PY) scripts/ask.py "$(Q)"

app:
	$(STREAMLIT) run app/streamlit_app.py

eval-retrieval:
	$(PY) scripts/evaluate.py --retrieval

eval:
	$(PY) scripts/evaluate.py --all

stats:
	$(PY) scripts/download.py --stats

clean:
	rm -rf data/index data/processed reports/*.json reports/*.md
	@echo "kept data/raw (the cache) so nothing is re-downloaded"
