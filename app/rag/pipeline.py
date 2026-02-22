from app.rag.generator import LLMGenerator
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.utils.cache import get, set 
from app.config import app_settings
import structlog

logger = structlog.get_logger()

class HybridRAG:
    def __init__(self):
        self.config = app_settings
        self.retriever = Retriever()
        self.reranker = Reranker(app_settings.RERANK_MODEL)
        self.generator = LLMGenerator()
    
    async def query(self, query: str, use_cache: bool = True):
        if use_cache:
            logger.info("Checking cache for query.", query=query)
            cached = await get(query)
            if cached:
                logger.info("Cached query found. Returning cached.", query=query, cached=cached)
                return {**cached, "cached": True}
        
        logger.info("Retrieving nodes for query...")
        retrieved_nodes = self.retriever.retrieve(query)
        logger.info("Top 3 chunks unranked", samples=[n.node.text[:300] for n in retrieved_nodes[:3]])
        
        reranked_nodes = retrieved_nodes
        
        if retrieved_nodes and self.config.USE_RERANKER:
            logger.info("ReRanking retrieved nodes...")
            reranked_nodes = self.reranker.rerank(query, retrieved_nodes, top_n=self.config.RERANK_TOP_N)
            logger.info("Top 3 chunks reranked", samples=[n.node.text[:300] for n in reranked_nodes[:3]])

        final_nodes = reranked_nodes[:self.config.FINAL_CONTEXT_N]

        logger.info("Generating LLM response with final nodes...")
        response = self.generator.generate(query, final_nodes)
        logger.info("LLM response.", response=response)
        
        result = {
            "answer": response["answer"],
            "sources": response["sources"],
            "retrieved_nodes": retrieved_nodes,
            "reranked_nodes": reranked_nodes,
            "mode": self.config.RETRIEVAL_MODE,
            "cached": False,
        }

        if use_cache:
            logger.info("Caching query and answer with Redis...")
            await set(query, {
                "answer": result["answer"],
                "sources": result["sources"],
                "mode": result["mode"],
            })

        return result