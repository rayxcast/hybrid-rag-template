import pytest
from fastapi import HTTPException
from llama_index.core.vector_stores import FilterOperator

from app.rag.metadata_filters import (
    metadata_filter_scope,
    to_llama_metadata_filters,
    validate_metadata_filters,
)

HTTP_BAD_REQUEST = 400


def test_validate_metadata_filters_accepts_scalar_values() -> None:
    filters = validate_metadata_filters({
        " tenant_id ": "acme",
        "document_id": 123,
        "published": True,
        "deleted_at": None,
    })

    assert filters == {
        "tenant_id": "acme",
        "document_id": 123,
        "published": True,
        "deleted_at": None,
    }


def test_validate_metadata_filters_rejects_nested_values() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_metadata_filters({"tenant": {"id": "acme"}})

    assert exc_info.value.status_code == HTTP_BAD_REQUEST


def test_metadata_filter_scope_is_stable_by_key_order() -> None:
    first = metadata_filter_scope({"tenant_id": "acme", "document_id": "handbook"})
    second = metadata_filter_scope({"document_id": "handbook", "tenant_id": "acme"})

    assert first == second


def test_to_llama_metadata_filters_uses_equality_filters() -> None:
    filters = to_llama_metadata_filters({"tenant_id": "acme", "document_id": "handbook"})

    assert filters is not None
    assert [(item.key, item.value, item.operator) for item in filters.filters] == [
        ("document_id", "handbook", FilterOperator.EQ),
        ("tenant_id", "acme", FilterOperator.EQ),
    ]
