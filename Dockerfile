# ---- Build stage: install dependencies ----
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download embedding model during build to avoid cold-start on first deploy
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh')" 2>/dev/null || true

# ---- Runtime stage: minimal image ----
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create writable data directories (mounted as volume in docker-compose)
RUN mkdir -p data/lancedb_data data/images data/memory paper_cache

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1

# Non-root user for security
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
