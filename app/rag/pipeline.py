from app.core.observability.timing import stage_timer
from app.rag.generator import LLMGenerator
from app.rag.reranker_providers.factory import get_reranker
from app.rag.retriever import Retriever
from app.rag.vectorstores.factory import get_vector_store_provider
from app.utils.cache import get_semantic, set_semantic 
from app.config import app_settings
import structlog
import time

logger = structlog.get_logger()

class HybridRAG:
    def __init__(self):
        self.config = app_settings
        self.retriever = Retriever()
        self.reranker = get_reranker()
        self.generator = LLMGenerator()
        self.vector_store_provider = get_vector_store_provider()
    
    async def query(self, query: str, trace_id: str, cache: bool = True, return_metadata: bool = False):
        
        total_start = time.perf_counter()
        metrics = {}
        
        if self.vector_store_provider.supports_sparse() and app_settings.RETRIEVAL_MODE == "hybrid":
            logger.info("Using hybrid mode.")
        else:
            logger.warning("Using dense mode: Hybrid mode is not supported; Sparse requested but not supported by backend.")

        use_cache = False if not cache else self.config.USE_CACHE # override use_cache if cache==false else default self.config.USE_CACHE

        if use_cache:
            # 1️⃣ Check cached
            with stage_timer("check_cached", logger, trace_id):
                # logger.info("Checking cached for query.", query=query)
                cached, score = await get_semantic(query, threshold=0.92)
            if cached:
                # logger.info("Cached query found. Returning cached.", query=query, cached=cached)
                total_duration = time.perf_counter() - total_start
                logger.info(
                    "cache_pipeline_total_latency",
                    trace_id=trace_id,
                    duration_seconds=round(total_duration, 4),
                )
                return {**cached, "cached": True, "score": score}
        
        # 2️⃣ Retrieval
        with stage_timer("retrieval", logger, trace_id, metrics):
            # logger.info("Retrieving nodes for query...")
            retrieved_nodes = self.retriever.retrieve(query, self.vector_store_provider.supports_sparse())
            # logger.info("Top 3 chunks unranked", samples=[n.node.text[:300] for n in retrieved_nodes[:3]])
        # metrics["retrieval_time"] = round(duration, 4)

        # 3️⃣ Rerank
        reranked_nodes = []
        if retrieved_nodes and self.config.USE_RERANKER and self.reranker:
            with stage_timer("rerank", logger, trace_id, metrics):
                # logger.info("ReRanking retrieved nodes...")
                reranked_nodes = self.reranker.rerank(query, retrieved_nodes, top_n=self.config.RERANK_TOP_N)
                # logger.info("Top 3 chunks reranked", samples=[n.node.text[:300] for n in reranked_nodes[:3]])

        # 4️⃣ Generation
        with stage_timer("generation", logger, trace_id, metrics):
            final_nodes = reranked_nodes[:self.config.FINAL_CONTEXT_N] if self.config.USE_RERANKER and reranked_nodes else retrieved_nodes[:self.config.FINAL_CONTEXT_N]
            # logger.info("Generating LLM response with final nodes...")
            response = self.generator.generate(query, final_nodes)
            # logger.info("LLM response.", response=response)

        result = {
            "answer": response["answer"],
            "sources": response["sources"],
            "mode": self.config.RETRIEVAL_MODE,
            "cached": False
        }

        # print("metrics:", metrics)
        # Conditionally add eval data if testing
        if return_metadata:
            result.update({
                "retrieved_nodes": retrieved_nodes,
                "reranked_nodes": reranked_nodes,
                "latency": metrics, 
            })
    
        if use_cache:
            # 5️⃣ Caching
            with stage_timer("cache_response", logger, trace_id):
                # logger.info("Caching query and answer with Redis...")
                await set_semantic(query, {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "mode": result["mode"],
                })

        total_duration = time.perf_counter() - total_start

        logger.info(
            "rag_pipeline_total_latency",
            trace_id=trace_id,
            duration_seconds=round(total_duration, 4),
        )

        # logger.info("query result", result=result)

        return result