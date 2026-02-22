# # Use official slim Python 3.12 image (small, secure, fast)
# FROM python:3.12-slim

# # Set working directory
# WORKDIR /app

# # RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr libtesseract-dev
# RUN apk add --no-cache poppler-utils tesseract-ocr tesseract-ocr-dev

# # Install uv (super-fast package manager) — we use it for deps
# RUN pip install --no-cache-dir uv

# # FORCE CPU-ONLY TORCH (Saves ~5GB)
# RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

# # Copy requirements first for caching
# COPY requirements.txt .
# # Install dependencies with uv (creates virtualenv-like isolation inside container)
# RUN uv pip install --system --no-cache -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Expose FastAPI port
# EXPOSE 8000

# # Run with uvicorn (production: --workers based on CPU, no reload)
# CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN uv pip install --system --no-cache --no-deps torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]