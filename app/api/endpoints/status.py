from fastapi import APIRouter
from qdrant_client import QdrantClient

from app.config import app_settings

router = APIRouter(tags=["status"])


@router.get("/status/")
async def status():
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
        "status": "ok" if qdrant_error is None else "degraded",
        "retrieval_mode": app_settings.RETRIEVAL_MODE,
        "collection_name": app_settings.COLLECTION_NAME,
        "collection_exists": collection_exists,
        "index_ready": collection_exists and point_count > 0,
        "point_count": point_count,
        "qdrant_error": qdrant_error,
    }
