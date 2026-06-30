import pytest

from app.rag import ingestion


class FakeStoreProvider:
    async def init_collection_if_needed(self) -> None:
        return None

    async def delete_collection(self) -> dict[str, object]:
        return {
            "collection_name": "hybrid_rag_docs",
            "deleted": True,
            "existed": True,
        }


class FakeIndexer:
    def __init__(self, *, fail: bool = False) -> None:
        self.store_provider = FakeStoreProvider()
        self.fail = fail

    def build_index(self, nodes: list[str]) -> None:
        if self.fail:
            raise RuntimeError("indexing failed")


class FakeSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_nodes_from_documents(self, documents: list[str]) -> list[str]:
        return ["node-1"]


@pytest.mark.asyncio
async def test_ingest_documents_bumps_revision_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions: list[str] = []

    async def fake_bump_index_revision() -> str:
        revisions.append("1")
        return "1"

    monkeypatch.setattr(ingestion, "get_indexer", FakeIndexer)
    monkeypatch.setattr(ingestion, "SentenceSplitter", FakeSplitter)
    monkeypatch.setattr(ingestion, "load_documents", lambda _: ["document"])
    monkeypatch.setattr(ingestion, "bump_index_revision", fake_bump_index_revision)

    result = await ingestion.ingest_documents("unused", request_id="test-request")

    assert revisions == ["1"]
    assert result["trace"]["index_revision"] == "1"


@pytest.mark.asyncio
async def test_ingest_documents_does_not_bump_revision_after_index_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions: list[str] = []

    async def fake_bump_index_revision() -> str:
        revisions.append("1")
        return "1"

    monkeypatch.setattr(ingestion, "get_indexer", lambda: FakeIndexer(fail=True))
    monkeypatch.setattr(ingestion, "SentenceSplitter", FakeSplitter)
    monkeypatch.setattr(ingestion, "load_documents", lambda _: ["document"])
    monkeypatch.setattr(ingestion, "bump_index_revision", fake_bump_index_revision)

    with pytest.raises(RuntimeError, match="indexing failed"):
        await ingestion.ingest_documents("unused", request_id="test-request")

    assert revisions == []
