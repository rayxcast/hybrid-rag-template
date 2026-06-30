import asyncio
import time

import structlog

from app.config import app_settings
from app.core.observability.timing import stage_timer
from app.rag.generator import LLMGenerator
from app.rag.reranker_providers.factory import get_reranker
from app.rag.retriever import Retriever
from app.rag.trace import summarize_nodes
from app.rag.vectorstores.factory import get_vector_store_provider
from app.utils.cache import get_current_cache_scope, get_semantic, set_semantic

logger = structlog.get_logger()
active_requests = 0
active_requests_lock = asyncio.Lock()

class HybridRAG:
    def __init__(self) -> None:
        self.config = app_settings
        self.retriever = Retriever()
        self.reranker = get_reranker()
        self.generator = LLMGenerator()
        self.vector_store_provider = get_vector_store_provider()

    async def query(  # noqa: PLR0915
        self,
        query: str,
        trace_id: str,
        cache: bool = True,
        return_metadata: bool = False,
    ) -> dict[str, object]:
        global active_requests  # noqa: PLW0603

        external_calls = {
            "embedding_calls": {
                "cache_lookup": 0,
                "retrieval_query": 0,
                "cache_write": 0,
            },
            "llm_calls": {
                "generation": 0,
            },
            "reranker_calls": {
                "remote": 0,
            },
        }
        warnings = [
            "Chunk-level citations only.",
            (
                "Reset/re-ingest when embedding provider, embedding model, "
                "or embedding dimension changes."
            ),
        ]
        trace = {
            "request_id": trace_id,
            "operation": "query",
            "providers": {
                "llm": self.config.LLM_PROVIDER,
                "embedding": self.config.DENSE_PROVIDER,
                "sparse": self.config.SPARSE_PROVIDER,
                "reranker": self.config.RERANKER_PROVIDER,
            },
            "models": {
                "llm": self.config.LLM_MODEL,
                "embedding": self.config.EMBEDDING_MODEL,
                "sparse": self.config.SPARSE_MODEL,
                "reranker": self.config.RERANKER_MODEL,
            },
            "retrieval": {
                "mode": self.config.RETRIEVAL_MODE,
                "top_k": self.config.SIMILARITY_TOP_K,
                "similarity_cutoff": self.config.SIMILARITY_CUTOFF,
                "rerank_top_n": self.config.RERANK_TOP_N,
                "final_context_n": self.config.FINAL_CONTEXT_N,
            },
            "cache": {
                "enabled": False if not cache else self.config.USE_CACHE,
                "hit": False,
            },
            "external_calls": external_calls,
            "warnings": warnings,
        }

        async with active_requests_lock:
            active_requests += 1
            logger.info(
                "pipeline_active_requests",
                trace_id=trace_id,
                active_requests=active_requests,
            )

        try:
            total_start = time.perf_counter()
            metrics = {}

            supports_sparse = self.vector_store_provider.supports_sparse()
            if supports_sparse and app_settings.RETRIEVAL_MODE == "hybrid":
                logger.info(
                    "retrieval_mode_selected",
                    trace_id=trace_id,
                    retrieval_mode="hybrid",
                    llm_provider=self.config.LLM_PROVIDER,
                    embedding_provider=self.config.DENSE_PROVIDER,
                    top_k=self.config.SIMILARITY_TOP_K,
                )
            else:
                warning = "Using dense/default mode because hybrid mode is unavailable or disabled."
                warnings.append(warning)
                logger.warning(
                    "retrieval_mode_fallback",
                    trace_id=trace_id,
                    warning=warning,
                )

            use_cache = False if not cache else self.config.USE_CACHE
            cache_embedding = None
            cache_scope = await get_current_cache_scope()
            trace["cache_scope"] = cache_scope

            if use_cache:
                external_calls["embedding_calls"]["cache_lookup"] = 1
                with stage_timer("check_cached", logger, trace_id, metrics):
                    cached, score, cache_embedding = await get_semantic(
                        query,
                        cache_scope=cache_scope,
                        threshold=self.config.CACHE_SIMILARITY_THRESHOLD,
                        return_embedding=True,
                    )
                if cached:
                    warnings.append("Semantic cache hit; retrieval and generation were skipped.")
                    total_duration = time.perf_counter() - total_start
                    metrics["total"] = round(total_duration, 4)
                    trace.update({
                        "cache": {
                            "enabled": True,
                            "hit": True,
                            "score": round(score, 4),
                            "scope": cache_scope,
                        },
                        "timings": metrics,
                    })
                    logger.info(
                        "cache_pipeline_total_latency",
                        trace_id=trace_id,
                        duration_seconds=round(total_duration, 4),
                        cache_score=round(score, 4),
                    )
                    return {**cached, "cached": True, "score": score, "trace": trace}

            external_calls["embedding_calls"]["retrieval_query"] = 1
            with stage_timer("retrieval", logger, trace_id, metrics):
                retrieved_nodes = await self.retriever.retrieve(query, supports_sparse)
            retrieved_chunks = summarize_nodes(retrieved_nodes, stage="retrieved")
            if retrieved_nodes and not any(
                "dense_score" in chunk or "sparse_score" in chunk for chunk in retrieved_chunks
            ):
                warnings.append(
                    "Dense/SPLADE branch scores were not exposed by the vector store; "
                    "showing fused scores where available."
                )
            if not retrieved_nodes:
                warnings.append(
                    "No chunks passed retrieval/cutoff; answer falls back to a "
                    "document-grounded refusal."
                )
            logger.info(
                "retrieved_nodes",
                trace_id=trace_id,
                count=len(retrieved_nodes),
                chunks=retrieved_chunks,
            )

            reranked_nodes = []
            if retrieved_nodes and self.config.USE_RERANKER and self.reranker:
                if self.config.RERANKER_PROVIDER == "remote":
                    external_calls["reranker_calls"]["remote"] = 1
                with stage_timer("rerank", logger, trace_id, metrics):
                    reranked_nodes = await self.reranker.rerank(
                        query,
                        retrieved_nodes,
                        top_n=self.config.RERANK_TOP_N,
                    )
                logger.info(
                    "reranked_nodes",
                    trace_id=trace_id,
                    count=len(reranked_nodes),
                    rerank_top_n=self.config.RERANK_TOP_N,
                    chunks=summarize_nodes(reranked_nodes, stage="reranked"),
                )

            with stage_timer("generation", logger, trace_id, metrics):
                final_nodes = (
                    reranked_nodes[: self.config.FINAL_CONTEXT_N]
                    if self.config.USE_RERANKER and reranked_nodes
                    else retrieved_nodes[: self.config.FINAL_CONTEXT_N]
                )
                external_calls["llm_calls"]["generation"] = 1 if final_nodes else 0
                response = await self.generator.generate(query, final_nodes)

            result = {
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
                with stage_timer("cache_response", logger, trace_id):
                    external_calls["embedding_calls"]["cache_write"] = 0 if cache_embedding else 1
                    await set_semantic(query, {
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "mode": result["mode"],
                    }, cache_scope=cache_scope, embedding=cache_embedding)

            total_duration = time.perf_counter() - total_start
            metrics["total"] = round(total_duration, 4)
            trace.update({
                "cache": {
                    "enabled": use_cache,
                    "hit": False,
                    "scope": cache_scope,
                    "cache_write_reused_lookup_embedding": bool(use_cache and cache_embedding),
                },
                "timings": metrics,
                "retrieved_count": len(retrieved_nodes),
                "reranked_count": len(reranked_nodes),
                "final_context_count": len(final_nodes),
                "retrieved_chunks": retrieved_chunks,
                "reranked_chunks": summarize_nodes(reranked_nodes, stage="reranked"),
                "final_chunks": summarize_nodes(final_nodes, stage="final"),
            })
            result["trace"] = trace

            logger.info(
                "rag_pipeline_total_latency",
                trace_id=trace_id,
                duration_seconds=round(total_duration, 4),
                retrieved_count=len(retrieved_nodes),
                reranked_count=len(reranked_nodes),
                final_context_count=len(final_nodes),
            )

            return result
        except Exception as error:
            logger.error(
                "rag_pipeline_failed",
                trace_id=trace_id,
                error=str(error),
                exc_info=True,
            )
            raise
        finally:
            async with active_requests_lock:
                active_requests -= 1
