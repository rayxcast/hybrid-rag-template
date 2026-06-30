from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.validation import (
    require_path_ingest_enabled,
    validate_query_text,
    validate_upload_filename,
    write_limited_upload,
)

HTTP_400_BAD_REQUEST = 400
HTTP_403_FORBIDDEN = 403
HTTP_413_CONTENT_TOO_LARGE = 413
HTTP_415_UNSUPPORTED_MEDIA_TYPE = 415


class AsyncUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def read(self, _: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_validate_upload_filename_rejects_unsupported_extension() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_filename("payload.exe", allowed_extensions={".pdf", ".txt", ".md"})

    assert exc_info.value.status_code == HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_validate_upload_filename_sanitizes_path_segments() -> None:
    assert validate_upload_filename("../report.pdf", allowed_extensions={".pdf"}) == "report.pdf"


@pytest.mark.asyncio
async def test_write_limited_upload_rejects_oversized_file(tmp_path: Path) -> None:
    upload = AsyncUpload([b"12345", b"67890"])

    with pytest.raises(HTTPException) as exc_info:
        await write_limited_upload(upload, tmp_path / "report.txt", max_bytes=8)  # type: ignore[arg-type]

    assert exc_info.value.status_code == HTTP_413_CONTENT_TOO_LARGE


@pytest.mark.asyncio
async def test_write_limited_upload_rejects_empty_file(tmp_path: Path) -> None:
    upload = AsyncUpload([])

    with pytest.raises(HTTPException) as exc_info:
        await write_limited_upload(upload, tmp_path / "empty.txt", max_bytes=8)  # type: ignore[arg-type]

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST


def test_validate_query_text_rejects_blank_query() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_query_text("   ")

    assert exc_info.value.status_code == HTTP_400_BAD_REQUEST


def test_validate_query_text_rejects_too_long_query() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_query_text("x" * 6, max_chars=5)

    assert exc_info.value.status_code == HTTP_413_CONTENT_TOO_LARGE


def test_validate_query_text_strips_valid_query() -> None:
    assert validate_query_text("  What is hybrid search?  ") == "What is hybrid search?"


def test_require_path_ingest_enabled_rejects_disabled_path_ingest() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_path_ingest_enabled(False)

    assert exc_info.value.status_code == HTTP_403_FORBIDDEN


def test_require_path_ingest_enabled_allows_enabled_path_ingest() -> None:
    require_path_ingest_enabled(True)
