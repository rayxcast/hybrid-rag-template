import tempfile
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.validation import (
    require_path_ingest_enabled,
    validate_upload_filename,
    write_limited_upload,
)
from app.rag.ingestion import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = structlog.get_logger()


@router.post("/")
async def ingest(
    request: Request,
    path: Annotated[str | None, Form(description="Local dir path inside container")] = None,
    file: Annotated[
        UploadFile | None,
        File(
            description=(
                "Single file upload. Leave unselected and untick 'Send empty value' "
                "if using opt-in path ingestion."
            )
        ),
    ] = None,
    recreate: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    request_id = getattr(request.state, "request_id", "no-id")
    logger.info(
        "ingest_request_received",
        request_id=request_id,
        path=path,
        filename=file.filename if file else None,
        recreate=recreate,
    )

    if file:
        safe_filename = validate_upload_filename(file.filename)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / safe_filename
            await write_limited_upload(file, file_path)
            result = await ingest_documents(
                str(Path(tmp_dir)),
                recreate,
                request_id=request_id,
                source_name=safe_filename,
                source_type="upload",
            )
    elif path:
        require_path_ingest_enabled()
        result = await ingest_documents(
            path,
            recreate,
            request_id=request_id,
            source_name=path,
            source_type="path",
        )
    else:
        raise HTTPException(400, "Provide 'path' or 'file'")

    return result
