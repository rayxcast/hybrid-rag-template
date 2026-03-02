from fastembed.rerank.cross_encoder import TextCrossEncoder
from .base import BaseReranker
from typing import List
from app.config import app_settings

class FastEmbedReranker(BaseReranker):

    def __init__(self):
        self.model = TextCrossEncoder(model_name=app_settings.RERANKER_MODEL)

    def rerank(self, query: str, nodes, top_n: int = 25) -> List:
        scores = list(
            self.model.rerank(
                query=query,
                documents=[node.text for node in nodes],
            )
        )

        reranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [node for node, _ in reranked[:top_n]]