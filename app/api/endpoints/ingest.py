from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from pathlib import Path
import shutil
import tempfile
from app.rag.ingestion import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/")
async def ingest(
    path: str = Form(None, description="Local dir path inside container"),
    file: UploadFile = File(None, description="Single file upload"),
    recreate: bool = Form(False),
):
    if file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / file.filename
            with file_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            result = await ingest_documents(str(Path(tmp_dir)), recreate)
    elif path:
        result = await ingest_documents(path, recreate)
    else:
        raise HTTPException(400, "Provide 'path' or 'file'")

    return result