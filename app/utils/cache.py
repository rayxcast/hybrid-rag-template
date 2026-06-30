"""Redis semantic cache scoped to the current indexed corpus."""

import json
import re
import uuid
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog
from llama_index.core import Settings
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from app.config import app_settings, configure_llm_settings

logger = structlog.get_logger()
MAX_COSINE_DISTANCE = 2.0

redis_client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)
configure_llm_settings()

SCHEMA = IndexSchema.from_dict(
    {
        "index": {"name": "semantic_cache", "prefix": "cache:"},
        "fields": [
            {"name": "query_text", "type": "text"},
            {"name": "answer", "type": "text"},
            {"name": "collection_name", "type": "text"},
            {"name": "embedding_provider", "type": "text"},
            {"name": "embedding_model", "type": "text"},
            {"name": "embedding_dim", "type": "text"},
            {"name": "index_revision", "type": "text"},
            {"name": "metadata_filter_scope", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": app_settings.EMBEDDING_DIM,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    }
)


def normalize_query(query: str) -> str:
    query = query.lower().strip()
    query = re.sub(r'[?.!,;:"]+', "", query)
    return re.sub(r"\s+", " ", query)


def index_revision_key(collection_name: str | None = None) -> str:
    collection = collection_name or app_settings.COLLECTION_NAME
    return f"index_revision:{collection}"


def build_cache_scope(
    index_revision: str | int,
    metadata_filter_scope: str | None = None,
) -> dict[str, str]:
    return {
        "collection_name": app_settings.COLLECTION_NAME,
        "embedding_provider": app_settings.DENSE_PROVIDER,
        "embedding_model": app_settings.EMBEDDING_MODEL,
        "embedding_dim": str(app_settings.EMBEDDING_DIM),
        "index_revision": str(index_revision),
        "metadata_filter_scope": metadata_filter_scope or "{}",
    }


def cache_scope_matches(cached_scope: dict[str, Any] | None, current_scope: dict[str, str]) -> bool:
    if not cached_scope:
        return False
    return all(str(cached_scope.get(key)) == value for key, value in current_scope.items())


async def get_index_revision(collection_name: str | None = None) -> str:
    key = index_revision_key(collection_name)
    await redis_client.setnx(key, "0")
    revision = await redis_client.get(key)
    return str(revision or "0")


async def bump_index_revision(collection_name: str | None = None) -> str:
    key = index_revision_key(collection_name)
    revision = await redis_client.incr(key)
    return str(revision)


async def get_current_cache_scope(metadata_filter_scope: str | None = None) -> dict[str, str]:
    revision = await get_index_revision()
    return build_cache_scope(revision, metadata_filter_scope=metadata_filter_scope)


async def get_connected_index() -> AsyncSearchIndex:
    index = AsyncSearchIndex(SCHEMA)
    await index.set_client(redis_client)
    return index


async def init_cache_index() -> None:
    try:
        await get_index_revision()
        index = await get_connected_index()
        await index.create(overwrite=False)
        logger.info("Semantic cache index initialized or already exists")
    except Exception as error:
        logger.error("Failed to initialize semantic cache index", error=str(error))
        raise


def _payload_from_cache_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        decoded = json.loads(record["answer"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, None

    if not isinstance(decoded, dict):
        return None, None

    payload = decoded.get("payload")
    scope = decoded.get("cache_scope")
    if not isinstance(payload, dict) or not isinstance(scope, dict):
        return None, None

    return payload, scope


async def get_semantic(
    query: str,
    cache_scope: dict[str, str],
    threshold: float | None = None,
    return_embedding: bool = False,
) -> tuple[dict[str, Any] | None, float, list[float] | None] | tuple[dict[str, Any] | None, float]:
    """Look up a semantic cache entry for the active index scope."""
    try:
        score_threshold = (
            threshold if threshold is not None else app_settings.CACHE_SIMILARITY_THRESHOLD
        )
        norm_query = normalize_query(query)
        q_emb = await Settings.embed_model.aget_text_embedding(norm_query)

        vector_query = VectorQuery(
            vector=q_emb,
            vector_field_name="embedding",
            return_fields=[
                "answer",
                "collection_name",
                "embedding_provider",
                "embedding_model",
                "embedding_dim",
                "index_revision",
                "metadata_filter_scope",
            ],
            num_results=10,
            return_score=True,
        )

        index = await get_connected_index()
        results = await index.query(vector_query)

        for result in results or []:
            distance_str = result.get("vector_distance")
            if distance_str is None:
                logger.warning("No vector_distance in semantic cache result", result=result)
                continue

            distance = float(distance_str)
            similarity = (
                1 - (distance / MAX_COSINE_DISTANCE)
                if distance <= MAX_COSINE_DISTANCE
                else 0.0
            )
            if similarity < score_threshold:
                continue

            cached_payload, cached_scope = _payload_from_cache_record(result)
            if not cache_scope_matches(cached_scope, cache_scope):
                logger.info(
                    "Semantic cache candidate rejected due to index scope mismatch",
                    query=query[:50],
                    similarity=similarity,
                    current_scope=cache_scope,
                    cached_scope=cached_scope,
                )
                continue

            logger.info(
                "Semantic cache hit",
                query=query[:50],
                distance=distance,
                similarity=similarity,
                cache_scope=cache_scope,
            )
            if return_embedding:
                return cached_payload, similarity, q_emb
            return cached_payload, similarity

        logger.debug("Semantic cache miss")
        if return_embedding:
            return None, 0.0, q_emb
        return None, 0.0

    except Exception as error:
        logger.error("Semantic cache get failed", error=str(error), exc_info=True)
        if return_embedding:
            return None, 0.0, None
        return None, 0.0


async def set_semantic(
    query: str,
    answer: object,
    cache_scope: dict[str, str],
    ttl: int | None = None,
    embedding: list[float] | None = None,
) -> None:
    """Store a semantic cache entry for the active index scope."""
    try:
        norm_query = normalize_query(query)
        emb_list = embedding or await Settings.embed_model.aget_text_embedding(norm_query)
        emb_bytes = np.array(emb_list, dtype=np.float32).tobytes()

        payload = {
            "cache_scope": cache_scope,
            "payload": answer,
        }
        key = f"cache:{uuid.uuid4().hex[:12]}"
        index = await get_connected_index()
        await index.load(
            [
                {
                    "id": key,
                    "query_text": norm_query,
                    "answer": json.dumps(payload),
                    "embedding": emb_bytes,
                    **cache_scope,
                }
            ]
        )
        await redis_client.expire(key, ttl or app_settings.CACHE_TTL_SECONDS)
        logger.info("Semantic cache set", query=query[:50], cache_scope=cache_scope)
    except Exception as error:
        logger.error("Semantic cache set failed", error=str(error))
