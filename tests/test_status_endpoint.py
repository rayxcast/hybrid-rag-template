import pytest

from app.api.endpoints import status as status_endpoint


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
