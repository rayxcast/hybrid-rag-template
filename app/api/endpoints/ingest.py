from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Request
from pathlib import Path
import shutil
import tempfile
import structlog
from app.rag.ingestion import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = structlog.get_logger()

@router.post("/")
async def ingest(
    request: Request,
    path: str = Form(None, description="Local dir path inside container"),
    file: UploadFile = File(None, description="Single file upload. Leave unselected and untick 'Send empty value' if using 'path' instead."),
    recreate: bool = Form(False),
):
    request_id = getattr(request.state, "request_id", "no-id")
    logger.info(
        "ingest_request_received",
        request_id=request_id,
        path=path,
        filename=file.filename if file else None,
        recreate=recreate,
    )

    if file:
        safe_filename = Path(file.filename or "upload").name
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / safe_filename
            with file_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            result = await ingest_documents(
                str(Path(tmp_dir)),
                recreate,
                request_id=request_id,
                source_name=safe_filename,
                source_type="upload",
            )
    elif path:
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
