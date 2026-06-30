from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel

from app.api.auth import require_api_key
from app.api.validation import validate_query_text
from app.rag.pipeline import HybridRAG

router = APIRouter(prefix="/query", tags=["query"], dependencies=[Depends(require_api_key)])
rag = HybridRAG()


class QueryRequest(BaseModel):
    query: str


@router.post("/")
async def query_endpoint(
    request: Request,
    req: Annotated[QueryRequest, Body()],
) -> dict[str, object]:
    request_id = getattr(request.state, "request_id", "no-id")
    query = validate_query_text(req.query)
    return await rag.query(query, trace_id=request_id)
