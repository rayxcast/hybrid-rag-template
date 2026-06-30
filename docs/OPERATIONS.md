# Operations

This guide covers local operation and production-template concerns for adopters.

## Startup

Create local configuration:

```bash
cp .env.example .env
```

Start the stack:

```bash
docker compose up --build
```

Open:

- Demo UI: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Liveness: http://localhost:8000/healthz
- Readiness: http://localhost:8000/readyz
- Status: http://localhost:8000/status/

First startup may fetch model files before the API accepts traffic.

## Shutdown and Reset

Stop containers:

```bash
docker compose down
```

Remove Qdrant and Redis volumes:

```bash
docker compose down -v
```

Use volume removal when switching embedding providers, changing embedding dimensions,
or intentionally resetting local state.

## Healthchecks

- App healthcheck calls `GET /readyz`.
- Qdrant healthcheck calls `/readyz`.
- Redis healthcheck calls `redis-cli ping`.
- Reranker healthcheck calls `GET /health`.

`/healthz` is a cheap process liveness check. `/readyz` checks app dependencies.
`/status/` reports collection existence, point count, retrieval mode, and Qdrant errors.

Run the Docker-backed health/readiness smoke test with:

```bash
make smoke
```

## Logs

For local readable logs:

```env
LOG_FORMAT=console
LOG_LEVEL=INFO
```

For JSON logs:

```env
LOG_FORMAT=json
LOG_LEVEL=INFO
```

Follow core service logs:

```bash
docker compose logs -f app reranker_service
```

Request logs include request IDs. Secrets and authorization headers are redacted.

## Cache and Index Revision

The semantic cache is scoped to the current indexed corpus. After successful ingestion,
the app increments the Redis key:

```text
index_revision:{COLLECTION_NAME}
```

Repeated queries may intentionally miss cache after ingestion because the document set
changed. This prevents stale answers from being reused for a new corpus.

Useful settings:

- `USE_CACHE`
- `CACHE_TTL_SECONDS`
- `CACHE_SIMILARITY_THRESHOLD`
- `COLLECTION_NAME`

## Provider Switching

When changing `DENSE_PROVIDER`, `EMBEDDING_MODEL`, or `EMBEDDING_DIM`:

1. Recreate or rename the Qdrant collection.
2. Clear Redis or start with fresh volumes.
3. Re-ingest documents.

Embedding spaces are not interchangeable.

## Common Failures

- Missing provider key: app starts, but query or ingestion fails when provider calls run.
- Wrong embedding dimension: Qdrant insert/search can fail or return invalid results.
- Reranker first start is slow: model files may be downloading.
- Docker socket unavailable: start OrbStack or Docker Desktop before running Docker checks.
- Stale local `.venv`: recreate with `rm -rf .venv && uv sync`.

## Production Adoption Notes

Before internet-facing deployment:

- Put the app behind TLS and a trusted gateway.
- Enable platform auth or the built-in API-key guard.
- Add rate limits and request body limits at the edge.
- Keep Qdrant, Redis, and reranker private.
- Add tenant/document authorization filters before retrieval.
- Export logs, metrics, and traces to your operations platform.
