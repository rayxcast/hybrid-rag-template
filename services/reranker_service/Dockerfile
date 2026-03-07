# ────────────────────────────────────────────────
# Builder stage: install deps using uv
FROM ghcr.io/astral-sh/uv:python3.11-alpine AS builder

# Add build deps if needed (usually not for your stack)
# RUN apk add --no-cache build-base

WORKDIR /app

# Optimize: copy only lockfile first → great cache hit if deps unchanged
COPY uv.lock pyproject.toml ./

# Install runtime deps only (no dev/tools)
# --system = install globally in image (no venv)
# --no-install-project = skip installing the project itself yet
RUN uv sync --frozen --no-install-project --no-dev

# ────────────────────────────────────────────────
# Final runtime stage: tiny base + copied artifacts
FROM python:3.11-alpine

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your actual code (changes often → last layer)
COPY . .

# Ensure PATH has binaries (uv installs to /usr/local/bin)
ENV PATH="/usr/local/bin:$PATH"

# Run (use gunicorn or uvicorn directly in prod)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]