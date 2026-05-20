from __future__ import annotations

from numbers import Real
from typing import Any


SOURCE_KEYS = (
    "file_name",
    "filename",
    "file_path",
    "path",
    "page_label",
    "page",
    "page_number",
    "chunk_id",
    "node_id",
    "document_id",
    "doc_id",
    "score",
    "similarity",
    "rerank_score",
    "dense_score",
    "sparse_score",
)


def _round_float(value: Any) -> Any:
    if isinstance(value, Real) and not isinstance(value, bool):
        return round(value, 4)
    return value


def _metadata(node_with_score: Any) -> dict:
    node = getattr(node_with_score, "node", None)
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _node_id(node_with_score: Any) -> str | None:
    node = getattr(node_with_score, "node", None)
    for target in (node_with_score, node):
        for attr in ("id_", "node_id", "id"):
            value = getattr(target, attr, None)
            if value:
                return str(value)
    return None


def source_name(metadata: dict) -> str | None:
    for key in ("file_name", "filename", "file_path", "path", "document_id", "doc_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def summarize_node(node_with_score: Any, rank: int, stage: str) -> dict:
    metadata = _metadata(node_with_score)
    summary = {
        "rank": rank,
        "stage": stage,
        "chunk_id": _node_id(node_with_score),
        "score": _round_float(getattr(node_with_score, "score", None)),
    }

    for key in SOURCE_KEYS:
        value = metadata.get(key)
        if value is not None and value != "":
            summary[key] = _round_float(value)

    rerank_score = metadata.get("rerank_score")
    if rerank_score is not None:
        summary["rerank_score"] = _round_float(rerank_score)

    if source_name(metadata):
        summary["source"] = source_name(metadata)

    return {key: value for key, value in summary.items() if value is not None}


def summarize_nodes(nodes: list[Any], stage: str, limit: int | None = None) -> list[dict]:
    selected = nodes[:limit] if limit else nodes
    return [
        summarize_node(node_with_score, rank=index + 1, stage=stage)
        for index, node_with_score in enumerate(selected)
    ]


def summarize_sources(nodes: list[Any]) -> list[dict]:
    sources = []
    for node_with_score in nodes:
        metadata = _metadata(node_with_score)
        source = {
            key: _round_float(metadata[key])
            for key in SOURCE_KEYS
            if key in metadata and metadata[key] not in (None, "")
        }
        source.setdefault("chunk_id", _node_id(node_with_score))
        score = getattr(node_with_score, "score", None)
        if score is not None:
            source["score"] = _round_float(score)
        sources.append({key: value for key, value in source.items() if value is not None})
    return sources


def attach_rerank_score(node_with_score: Any, score: float) -> None:
    node = getattr(node_with_score, "node", None)
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        metadata["rerank_score"] = float(score)
