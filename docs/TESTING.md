# Testing

The test suite is designed to prove the template's safety and correctness boundaries
without requiring live model providers, Qdrant, or Redis for every test.

## Local Commands

Run tracked tests:

```bash
uv run python -m pytest
```

Run the focused lint gate used by CI:

```bash
uv run ruff check \
  app/api/auth.py \
  app/api/validation.py \
  app/api/endpoints/ingest.py \
  app/api/endpoints/query.py \
  app/api/endpoints/status.py \
  app/rag/ingestion.py \
  app/rag/pipeline.py \
  app/utils/cache.py \
  tests/test_api_auth.py \
  tests/test_api_validation.py \
  tests/test_cache_scope.py \
  tests/test_ingestion_revision.py \
  tests/test_status_endpoint.py
```

Validate Docker Compose:

```bash
docker compose config --no-interpolate
```

Build the application images when Docker or OrbStack is running:

```bash
docker compose build app reranker_service
```

## Current Coverage

- API-key auth parsing and failure modes.
- Upload filename, extension, size, and empty-file validation.
- Query validation.
- Path ingestion disabled-by-default behavior.
- Status response shape with Qdrant mocked.
- Cache scope acceptance and rejection.
- Ingestion revision bump after success and no bump after indexing failure.

## CI Scope

GitHub Actions runs tests and focused Ruff checks on the hardened safety surfaces. Full
repository Ruff cleanup remains a separate normalization task because older modules still
have style debt unrelated to the production-readiness slices.

## Known Gaps

- No live Qdrant/Redis integration test yet.
- No end-to-end ingestion/query/cache invalidation smoke test in CI.
- No provider-backed LLM or embedding tests in CI.
- Evaluation quality is not yet enforced as a CI gate.

Recommended next testing improvements:

- Add a Docker-backed smoke test for ingest, query, re-ingest, and cache miss after revision bump.
- Add small fake-provider tests around retrieval and generation orchestration.
- Add a cheap deterministic eval fixture for pull requests.
