# Engineering Context

This file captures durable project facts for future maintainers and reviewers.

## Project Goal

Hybrid RAG Template is an open-source, production-oriented starting point for teams
adding retrieval-augmented generation to an existing application. It is not intended
to be a complete SaaS product or a turnkey internet-facing deployment.

## Core Architecture

- FastAPI app exposes ingest, query, status, and demo UI routes.
- Qdrant stores dense and sparse document vectors.
- Redis Stack stores semantic answer cache entries and active index revision state.
- A separate FastAPI reranker service runs FastEmbed cross-encoder reranking.
- LiteLLM and LlamaIndex integrations provide provider flexibility for LLMs and embeddings.
- Docker Compose is the recommended local runtime; OrbStack is recommended on macOS.

## Production-Template Defaults

- Only the API port is published by default in Docker Compose.
- Qdrant, Redis, and the reranker are internal services by default.
- App and reranker containers run as non-root users.
- Upload ingestion validates filename, extension, non-empty content, and size.
- Container path ingestion is disabled unless `ALLOW_PATH_INGEST=true`.
- Optional API-key auth protects `/ingest/` and `/query/` when enabled.
- Semantic cache entries are scoped to collection, embedding config, and index revision.

## Cache Correctness

Redis semantic cache entries are not reusable across document/index states. The active
revision is stored at:

```text
index_revision:{COLLECTION_NAME}
```

Successful ingestion bumps this revision after indexing completes. Failed ingestion does
not bump it. TTL controls cache growth, while revision scope controls correctness.

## Quality Gates

The CI gate runs:

- `uv run python -m pytest`
- focused Ruff checks on hardened production-template surfaces
- `docker compose config --no-interpolate`
- `docker compose build app reranker_service`

Full-repo Ruff is intentionally not the gate yet. Older modules still have style debt
that should be normalized in a dedicated cleanup slice rather than mixed into feature work.

## Known Tradeoffs

- API-key auth is a template guard, not a full identity system.
- Upload extension checks are not malware scanning or MIME verification.
- Multi-tenant document authorization is not implemented.
- Rate limiting should be added at a proxy, gateway, or middleware layer.
- Live Qdrant/Redis integration tests are not yet part of CI.
