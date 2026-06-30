.PHONY: test lint compose-check docker-build check

FOCUSED_RUFF_TARGETS = \
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

test:
	uv run python -m pytest

lint:
	uv run ruff check $(FOCUSED_RUFF_TARGETS)

compose-check:
	docker compose config --no-interpolate

docker-build:
	docker compose build app reranker_service

check: test lint compose-check
