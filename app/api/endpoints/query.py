from fastapi import APIRouter, Body, Request
from pydantic import BaseModel
from app.rag.pipeline import HybridRAG

router = APIRouter(prefix="/query", tags=["query"])
rag = HybridRAG()

class QueryRequest(BaseModel):
    query: str

@router.post("/")
async def query_endpoint(request: Request, req: QueryRequest = Body(...)):
    request_id = getattr(request.state, "request_id", "no-id")
    return await rag.query(req.query, trace_id=request_id)
