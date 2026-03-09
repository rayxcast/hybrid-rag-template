import asyncio

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
active_requests = 0
active_requests_lock = asyncio.Lock()

class HybridRAG:
    def __init__(self):
        self.config = app_settings
        self.retriever = Retriever()
        self.reranker = get_reranker()
        self.generator = LLMGenerator()
        self.vector_store_provider = get_vector_store_provider()
    
    async def query(self, query: str, trace_id: str, cache: bool = True, return_metadata: bool = False):
        global active_requests

        async with active_requests_lock:
            active_requests += 1
            logger.info("pipeline_active_requests", active_requests=active_requests)

        try:
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
                    cached, score = await get_semantic(query, threshold=0.92)
                if cached:
                    total_duration = time.perf_counter() - total_start
                    logger.info(
                        "cache_pipeline_total_latency",
                        trace_id=trace_id,
                        duration_seconds=round(total_duration, 4),
                    )
                    return {**cached, "cached": True, "score": score}
            
            # 2️⃣ Retrieval
            with stage_timer("retrieval", logger, trace_id, metrics):
                retrieved_nodes = await self.retriever.retrieve(query, self.vector_store_provider.supports_sparse())
            logger.info("retrieved_nodes_count", trace_id=trace_id, count=len(retrieved_nodes))

            # 3️⃣ Rerank
            reranked_nodes = []
            if retrieved_nodes and self.config.USE_RERANKER and self.reranker:
                with stage_timer("rerank", logger, trace_id, metrics):
                    reranked_nodes = await self.reranker.rerank(query, retrieved_nodes, top_n=self.config.RERANK_TOP_N)
                logger.info("reranked_nodes_count", trace_id=trace_id, count=len(reranked_nodes), RERANK_TOP_N=self.config.RERANK_TOP_N)

            # 4️⃣ Generation
            with stage_timer("generation", logger, trace_id, metrics):
                final_nodes = reranked_nodes[:self.config.FINAL_CONTEXT_N] if self.config.USE_RERANKER and reranked_nodes else retrieved_nodes[:self.config.FINAL_CONTEXT_N]
                response = await self.generator.generate(query, final_nodes)
            
            # 4️⃣ Generation Mock
            # await asyncio.sleep(2)
            # logger.info(
            #     "stage_latency",
            #     trace_id=trace_id,
            #     stage="generation",
            #     duration_seconds=2,
            # )
            # metrics["generation"] = 2
            ##########################

            result = {
                # "answer": "response[answer]",
                # "sources": "response[sources]",
                "answer": response["answer"],
                "sources": response["sources"],
                "mode": self.config.RETRIEVAL_MODE,
                "cached": False
            }

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

            return result
        finally:
            async with active_requests_lock:
                active_requests -= 1