import os
import time
import logging

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder
import onnxruntime as ort

app = FastAPI()

# At top of main.py, after app = FastAPI()
logging.basicConfig(level=logging.INFO)
# Or better, route to uvicorn.error for visibility
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

def get_execution_providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


model = TextCrossEncoder(
    model_name="jinaai/jina-reranker-v1-tiny-en",
    providers=get_execution_providers(),
)

# Warmup
model.rerank(query="warmup", documents=["warmup"])


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 25


class RerankResponse(BaseModel):
    scores: List[float]


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest): # add async for GPU rerank and batching
    # In rerank endpoint
    logger.info(f"Start rerank in PID {os.getpid()} at {time.time()} | docs: {len(request.documents)}")
    scores = list(
        model.rerank(
            query=request.query,
            documents=request.documents,
        )
    )
    logger.info(f"End rerank in PID {os.getpid()} at {time.time()}")
    return {"scores": scores}

@app.get("/health")
async def health_check():
    return {"status": "ok"}