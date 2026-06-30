import httpx
import redis.asyncio as redis
from fastapi import APIRouter, Response
from fastapi import status as http_status
from qdrant_client import QdrantClient

from app.config import app_settings

router = APIRouter(tags=["status"])


def _qdrant_status() -> dict[str, object]:
    client = QdrantClient(url=app_settings.QDRANT_URL)
    collection_exists = False
    point_count = 0
    qdrant_error = None

    try:
        collection_exists = client.collection_exists(app_settings.COLLECTION_NAME)
        if collection_exists:
            count_result = client.count(
                collection_name=app_settings.COLLECTION_NAME,
                exact=False,
            )
            point_count = count_result.count
    except Exception as error:
        qdrant_error = str(error)
    finally:
        close = getattr(client, "close", None)
        if close:
            close()

    return {
        "ok": qdrant_error is None,
        "collection_exists": collection_exists,
        "index_ready": collection_exists and point_count > 0,
        "point_count": point_count,
        "error": qdrant_error,
    }


async def _redis_status() -> dict[str, object]:
    client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)
    error = None
    try:
        await client.ping()
    except Exception as exc:
        error = str(exc)
    finally:
        await client.aclose()

    return {"ok": error is None, "error": error}


async def _reranker_status() -> dict[str, object]:
    if not app_settings.USE_RERANKER or app_settings.RERANKER_PROVIDER != "remote":
        return {"ok": True, "enabled": False, "error": None}

    error = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{app_settings.RERANKER_URL}/health")
            response.raise_for_status()
    except Exception as exc:
        error = str(exc)

    return {"ok": error is None, "enabled": True, "error": error}


@router.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"status": "ok", "service": app_settings.APP_NAME}


@router.get("/readyz")
async def readyz(response: Response) -> dict[str, object]:
    qdrant = _qdrant_status()
    redis_status = await _redis_status()
    reranker = await _reranker_status()
    dependencies = {
        "qdrant": qdrant,
        "redis": redis_status,
        "reranker": reranker,
    }
    is_ready = all(status["ok"] for status in dependencies.values())
    if not is_ready:
        response.status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if is_ready else "degraded",
        "ready": is_ready,
        "dependencies": dependencies,
    }


@router.get("/status/")
async def status() -> dict[str, object]:
    qdrant = _qdrant_status()

    return {
        "status": "ok" if qdrant["ok"] else "degraded",
        "retrieval_mode": app_settings.RETRIEVAL_MODE,
        "collection_name": app_settings.COLLECTION_NAME,
        "collection_exists": qdrant["collection_exists"],
        "index_ready": qdrant["index_ready"],
        "point_count": qdrant["point_count"],
        "qdrant_error": qdrant["error"],
    }
