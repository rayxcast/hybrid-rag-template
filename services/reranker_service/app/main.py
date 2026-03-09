import asyncio
import logging
import heapq
import time
from typing import List, Tuple

import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import reranker_app_settings


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reranker")

app = FastAPI()


# ---------------------------------------------------
# Execution Providers
# ---------------------------------------------------

def get_execution_providers():

    available = ort.get_available_providers()
    preferred = []

    if "MPSExecutionProvider" in available:
        preferred.append("MPSExecutionProvider")

    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")

    preferred.append("CPUExecutionProvider")

    return preferred


# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = TextCrossEncoder(
    model_name="jinaai/jina-reranker-v1-tiny-en",
    providers=get_execution_providers(),
)

# Warmup
model.rerank(query="warmup", documents=["warmup"])


# ---------------------------------------------------
# Queue
# ---------------------------------------------------

class RerankTask:

    def __init__(self, query: str, docs: List[str], future: asyncio.Future):
        self.query = query
        self.docs = docs
        self.future = future


queue: asyncio.Queue[RerankTask] = asyncio.Queue(
    maxsize=reranker_app_settings.QUEUE_SIZE
)


# ---------------------------------------------------
# Batch Worker
# ---------------------------------------------------

async def batch_worker(worker_id: int):

    while True:

        tasks: List[RerankTask] = []
        pairs: List[Tuple[str, str]] = []
        task_doc_counts: List[int] = []

        pair_count = 0

        # ---------------------------------------------------
        # Get first task (opportunistic batching)
        # ---------------------------------------------------

        try:
            try:
                first = queue.get_nowait()
            except asyncio.QueueEmpty:
                first = await asyncio.wait_for(
                    queue.get(),
                    timeout=reranker_app_settings.BATCH_TIMEOUT,
                )

        except asyncio.TimeoutError:
            continue

        tasks.append(first)

        doc_count = len(first.docs)
        task_doc_counts.append(doc_count)

        pairs.extend((first.query, doc) for doc in first.docs)
        pair_count += doc_count

        # ---------------------------------------------------
        # Fill batch
        # ---------------------------------------------------

        while len(tasks) < reranker_app_settings.MAX_BATCH_REQUESTS:

            try:
                task = queue.get_nowait()

            except asyncio.QueueEmpty:
                break

            doc_count = len(task.docs)

            # enforce pair limit
            if pair_count + doc_count > reranker_app_settings.MAX_BATCH_PAIRS:

                try:
                    queue.put_nowait(task)
                except asyncio.QueueFull:
                    logger.warning("Queue full while reinserting task")

                break

            tasks.append(task)
            task_doc_counts.append(doc_count)

            pairs.extend((task.query, doc) for doc in task.docs)
            pair_count += doc_count

        logger.info(
            f"Worker {worker_id} processing batch "
            f"{len(tasks)} requests / {pair_count} pairs "
            f"| queue={queue.qsize()} "
            f"| max_pairs={reranker_app_settings.MAX_BATCH_PAIRS}"
        )

        # ---------------------------------------------------
        # Inference
        # ---------------------------------------------------

        start = time.perf_counter()

        try:

            scores = list(
                model.rerank_pairs(
                    pairs,
                    batch_size=reranker_app_settings.INTERNAL_BATCH_SIZE,
                )
            )

            # Safety check
            if len(scores) != pair_count:
                raise RuntimeError(
                    f"Score count mismatch: expected {pair_count}, got {len(scores)}"
                )

            idx = 0

            for task, doc_count in zip(tasks, task_doc_counts):

                task_scores = scores[idx: idx + doc_count]
                idx += doc_count

                if not task.future.done():
                    task.future.set_result(task_scores)

                queue.task_done()

        except Exception as e:

            logger.exception("Batch inference failure")

            for task in tasks:
                if not task.future.done():
                    task.future.set_exception(e)

                queue.task_done()

        latency = time.perf_counter() - start

        logger.info(
            f"Worker {worker_id} batch done | latency={latency:.3f}s"
        )


# ---------------------------------------------------
# API
# ---------------------------------------------------

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 25


class RerankResponse(BaseModel):
    results: List[Tuple[int, float]]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):

    if len(request.documents) == 0:
        return {"results": []}

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    task = RerankTask(
        query=request.query,
        docs=request.documents,
        future=future,
    )

    try:
        queue.put_nowait(task)

    except asyncio.QueueFull:
        raise HTTPException(
            status_code=503,
            detail="Reranker overloaded",
        )

    start = time.perf_counter()

    scores = await future

    # Efficient top-k selection
    top = heapq.nlargest(
        request.top_n,
        enumerate(scores),
        key=lambda x: x[1],
    )

    latency = time.perf_counter() - start

    logger.info(
        f"Request done | docs={len(request.documents)} "
        f"| latency={latency:.3f}s"
    )

    return {"results": top}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------
# Startup
# ---------------------------------------------------

@app.on_event("startup")
async def startup():

    workers = reranker_app_settings.WORKERS

    logger.info(f"Starting {workers} reranker workers")

    for i in range(workers):
        asyncio.create_task(batch_worker(i))