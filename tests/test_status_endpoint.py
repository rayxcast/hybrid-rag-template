import pytest
from fastapi import Response

from app.api.endpoints import status as status_endpoint

HTTP_OK = 200
HTTP_SERVICE_UNAVAILABLE = 503


class FakeCountResult:
    count = 42


class FakeQdrantClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name == "hybrid_rag_docs"

    def count(self, collection_name: str, exact: bool) -> FakeCountResult:
        assert collection_name == "hybrid_rag_docs"
        assert exact is False
        return FakeCountResult()

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_status_response_shape_with_mocked_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_endpoint, "QdrantClient", FakeQdrantClient)

    response = await status_endpoint.status()

    assert response == {
        "status": "ok",
        "retrieval_mode": "hybrid",
        "collection_name": "hybrid_rag_docs",
        "collection_exists": True,
        "index_ready": True,
        "point_count": 42,
        "qdrant_error": None,
    }


@pytest.mark.asyncio
async def test_healthz_is_liveness_only() -> None:
    response = await status_endpoint.healthz()

    assert response == {"status": "ok", "service": "Hybrid RAG Template"}


@pytest.mark.asyncio
async def test_readyz_reports_dependency_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    async def redis_ok() -> dict[str, object]:
        return {"ok": True, "error": None}

    async def reranker_ok() -> dict[str, object]:
        return {"ok": True, "enabled": True, "error": None}

    monkeypatch.setattr(
        status_endpoint,
        "_qdrant_status",
        lambda: {
            "ok": True,
            "collection_exists": False,
            "index_ready": False,
            "point_count": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(status_endpoint, "_redis_status", redis_ok)
    monkeypatch.setattr(status_endpoint, "_reranker_status", reranker_ok)

    raw_response = Response()
    response = await status_endpoint.readyz(raw_response)

    assert response["status"] == "ok"
    assert response["ready"] is True
    assert raw_response.status_code == HTTP_OK


@pytest.mark.asyncio
async def test_readyz_degrades_when_dependency_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def redis_failed() -> dict[str, object]:
        return {"ok": False, "error": "connection refused"}

    async def reranker_ok() -> dict[str, object]:
        return {"ok": True, "enabled": True, "error": None}

    monkeypatch.setattr(
        status_endpoint,
        "_qdrant_status",
        lambda: {
            "ok": True,
            "collection_exists": False,
            "index_ready": False,
            "point_count": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(status_endpoint, "_redis_status", redis_failed)
    monkeypatch.setattr(status_endpoint, "_reranker_status", reranker_ok)

    raw_response = Response()
    response = await status_endpoint.readyz(raw_response)

    assert response["status"] == "degraded"
    assert response["ready"] is False
    assert raw_response.status_code == HTTP_SERVICE_UNAVAILABLE
