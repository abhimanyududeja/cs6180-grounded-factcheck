# Hosted build of the demo.
#
# This is NOT the system the report measures. The report's numbers come from
# qwen3:8b running locally through Ollama; an 8B model needs roughly 6 GB of RAM
# and cannot be served from a small container, so the hosted build calls the
# OpenAI API for generation instead.
#
# Everything else is held fixed: same corpus, same index, same retrieval,
# reranking and authority prior, same verification. Only the generator changes.
# See REPORT.md section 7.1 for what that swap is worth (0.534 to 0.753 accuracy).

FROM python:3.12-slim

# faiss and sentence-transformers need a toolchain at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch explicitly. The default wheel pulls CUDA
# libraries, which add gigabytes that a CPU-only host will never use.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the cross-encoder into the image. Downloading it on first request makes
# the first user wait a minute and fails outright if the host has no egress.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Render provides $PORT. Bind 0.0.0.0 or the health check cannot reach it.
CMD streamlit run app/streamlit_app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
