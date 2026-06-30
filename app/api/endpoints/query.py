from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel

from app.api.auth import require_api_key
from app.api.validation import validate_query_text
from app.rag.metadata_filters import MetadataFilterInput, validate_metadata_filters
from app.rag.pipeline import HybridRAG

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])
rag = HybridRAG()


class QueryRequest(BaseModel):
    query: str
    metadata_filters: dict[str, object] | None = None


@router.post("/")
async def query_endpoint(
    request: Request,
    req: Annotated[QueryRequest, Body()],
) -> dict[str, object]:
    request_id = getattr(request.state, "request_id", "no-id")
    query = validate_query_text(req.query)
    metadata_filters: MetadataFilterInput = validate_metadata_filters(req.metadata_filters)
    return await rag.query(query, trace_id=request_id, metadata_filters=metadata_filters)
