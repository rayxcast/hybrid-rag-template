############################
# Builder stage
############################
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1
ENV HF_HOME=/tmp/hf

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir uv

# Copy dependency files first → great caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

############################
# Final runtime stage
############################
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/.cache

# [Existing ENV variables...]
ENV PATH="/app/.venv/bin:$PATH"

# 1. Copy the virtual environment
COPY --from=builder /app/.venv /app/.venv

# 2. Copy application code
COPY app/ app/

# Create non-root user (very good practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--log-level", "info"]