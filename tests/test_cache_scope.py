from app.utils.cache import (
    _payload_from_cache_record,
    app_settings,
    build_cache_scope,
    cache_scope_matches,
)

DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_CACHE_SIMILARITY_THRESHOLD = 0.92


def test_cache_scope_matches_same_index_scope() -> None:
    scope = {
        "collection_name": "hybrid_rag_docs",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": "1536",
        "index_revision": "7",
        "metadata_filter_scope": "{}",
    }

    assert cache_scope_matches(scope, scope)


def test_cache_scope_rejects_different_index_revision() -> None:
    current_scope = {
        "collection_name": "hybrid_rag_docs",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": "1536",
        "index_revision": "8",
        "metadata_filter_scope": "{}",
    }
    cached_scope = {**current_scope, "index_revision": "7"}

    assert not cache_scope_matches(cached_scope, current_scope)


def test_cache_scope_rejects_different_embedding_config() -> None:
    current_scope = {
        "collection_name": "hybrid_rag_docs",
        "embedding_provider": "google",
        "embedding_model": "gemini-embedding-2",
        "embedding_dim": "1536",
        "index_revision": "7",
        "metadata_filter_scope": "{}",
    }
    cached_scope = {
        **current_scope,
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    }

    assert not cache_scope_matches(cached_scope, current_scope)


def test_cache_scope_rejects_different_metadata_filters() -> None:
    current_scope = {
        "collection_name": "hybrid_rag_docs",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": "1536",
        "index_revision": "7",
        "metadata_filter_scope": '{"tenant_id":"acme"}',
    }
    cached_scope = {**current_scope, "metadata_filter_scope": "{}"}

    assert not cache_scope_matches(cached_scope, current_scope)


def test_build_cache_scope_defaults_to_unfiltered_scope() -> None:
    scope = build_cache_scope("9")

    assert scope["metadata_filter_scope"] == "{}"


def test_payload_from_cache_record_returns_scoped_payload() -> None:
    record = {
        "answer": (
            '{"cache_scope":{"collection_name":"docs","embedding_provider":"openai",'
            '"embedding_model":"text-embedding-3-small","embedding_dim":"1536",'
            '"index_revision":"1","metadata_filter_scope":"{}"},'
            '"payload":{"answer":"cached","sources":[]}}'
        )
    }

    payload, scope = _payload_from_cache_record(record)

    assert payload == {"answer": "cached", "sources": []}
    assert scope == {
        "collection_name": "docs",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": "1536",
        "index_revision": "1",
        "metadata_filter_scope": "{}",
    }


def test_payload_from_cache_record_rejects_legacy_unscoped_payload() -> None:
    payload, scope = _payload_from_cache_record({"answer": '{"answer":"legacy"}'})

    assert payload is None
    assert scope is None


def test_cache_config_defaults_are_available() -> None:
    assert app_settings.CACHE_TTL_SECONDS == DEFAULT_CACHE_TTL_SECONDS
    assert app_settings.CACHE_SIMILARITY_THRESHOLD == DEFAULT_CACHE_SIMILARITY_THRESHOLD
