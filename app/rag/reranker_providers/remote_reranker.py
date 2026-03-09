import httpx
from .base import BaseReranker
from app.config import app_settings


class RemoteReranker(BaseReranker):

    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=app_settings.RERANKER_URL,
            timeout=15.0,
        )

    async def rerank(self, query, nodes, top_n=25):
        try: 
            documents = [node.text for node in nodes]
            response = await self.client.post(
                "/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )

            scores = response.json()["scores"]
            
            reranked = sorted(
                zip(nodes, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            return [node for node, _ in reranked[:top_n]]
        except Exception as error:
            raise error