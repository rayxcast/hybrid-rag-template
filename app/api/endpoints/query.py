from fastapi import APIRouter, Body
from pydantic import BaseModel
from app.rag.pipeline import HybridRAG
import uuid

trace_id = str(uuid.uuid4())
router = APIRouter(prefix="/query", tags=["query"])
rag = HybridRAG()

class QueryRequest(BaseModel):
    query: str

@router.post("/")
async def query_endpoint(req: QueryRequest = Body(...)):
    return await rag.query(req.query, trace_id=trace_id)