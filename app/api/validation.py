from collections.abc import Iterable
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import app_settings


def allowed_upload_extensions(raw_extensions: str | None = None) -> set[str]:
    extensions = raw_extensions or app_settings.ALLOWED_UPLOAD_EXTENSIONS
    return {
        extension.strip().lower()
        for extension in extensions.split(",")
        if extension.strip()
    }


def validate_upload_filename(
    filename: str | None,
    allowed_extensions: Iterable[str] | None = None,
) -> str:
    safe_filename = Path(filename or "").name
    if not safe_filename or safe_filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Upload must include a filename")

    suffix = Path(safe_filename).suffix.lower()
    allowed = set(allowed_extensions or allowed_upload_extensions())
    if suffix not in allowed:
        allowed_display = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported upload type. Allowed extensions: {allowed_display}",
        )

    return safe_filename


def validate_query_text(query: str, max_chars: int | None = None) -> str:
    normalized = query.strip()
    limit = max_chars or app_settings.QUERY_MAX_CHARS

    if not normalized:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    if len(normalized) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Query is too long. Maximum length is {limit} characters",
        )

    return normalized


def require_path_ingest_enabled(allow_path_ingest: bool | None = None) -> None:
    is_enabled = app_settings.ALLOW_PATH_INGEST if allow_path_ingest is None else allow_path_ingest
    if not is_enabled:
        raise HTTPException(
            status_code=403,
            detail="Path ingestion is disabled. Set ALLOW_PATH_INGEST=true to enable it.",
        )


async def write_limited_upload(
    file: UploadFile,
    destination: Path,
    max_bytes: int | None = None,
) -> int:
    limit = max_bytes or app_settings.MAX_UPLOAD_BYTES
    total_bytes = 0

    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload is too large. Maximum size is {limit} bytes",
                )
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Upload must not be empty")

    return total_bytes
