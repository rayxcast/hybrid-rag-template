from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder
import onnxruntime as ort

app = FastAPI()


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
async def rerank(request: RerankRequest):
    scores = list(
        model.rerank(
            query=request.query,
            documents=request.documents,
        )
    )

    return {"scores": scores}

@app.get("/health")
async def health_check():
    return {"status": "ok"}