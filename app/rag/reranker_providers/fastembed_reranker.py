from fastembed.rerank.cross_encoder import TextCrossEncoder
from .base import BaseReranker
from typing import List
from app.config import app_settings
import onnxruntime as ort


def _get_execution_providers():
    available = ort.get_available_providers()
    # print("ort.get_available_providers():", available)
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FastEmbedReranker(BaseReranker):

    def __init__(self):
        self.model = TextCrossEncoder(
            model_name=app_settings.RERANKER_MODEL,
            providers=_get_execution_providers(),
        )

        # Warmup
        self.model.rerank(query="warmup", documents=["warmup"])

    def rerank(self, query: str, nodes, top_n: int = 25) -> List:
        documents = [node.text for node in nodes]

        scores = list(
            self.model.rerank(
                query=query,
                documents=documents,
            )
        )

        reranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [node for node, _ in reranked[:top_n]]