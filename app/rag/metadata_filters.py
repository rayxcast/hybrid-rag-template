"""Validation and conversion helpers for retrieval metadata filters."""

import json

from fastapi import HTTPException
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

type MetadataFilterValue = str | int | float | bool | None
type MetadataFilterInput = dict[str, MetadataFilterValue]


def validate_metadata_filters(
    raw_filters: dict[str, object] | None,
) -> MetadataFilterInput:
    if raw_filters is None:
        return {}

    filters: MetadataFilterInput = {}
    for raw_key, raw_value in raw_filters.items():
        key = raw_key.strip()
        if not key:
            raise HTTPException(status_code=400, detail="Metadata filter keys must not be empty")
        if isinstance(raw_value, list | dict):
            raise HTTPException(
                status_code=400,
                detail="Metadata filter values must be scalar equality values",
            )
        if not isinstance(raw_value, str | int | float | bool) and raw_value is not None:
            raise HTTPException(
                status_code=400,
                detail="Metadata filter values must be string, number, boolean, or null",
            )
        filters[key] = raw_value

    return filters


def metadata_filter_scope(filters: MetadataFilterInput | None) -> str:
    normalized = filters or {}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def to_llama_metadata_filters(filters: MetadataFilterInput | None) -> MetadataFilters | None:
    if not filters:
        return None

    return MetadataFilters(
        filters=[
            MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
            for key, value in sorted(filters.items())
        ]
    )
