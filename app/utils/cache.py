"""Real Redis semantic cache."""
from llama_index.core import Settings
from app.config import app_settings, configure_llm_settings
import redis.asyncio as redis
import structlog
import json
from typing import Optional

logger = structlog.get_logger()

redis_client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)
configure_llm_settings()

async def get(query: str, threshold: float = 0.85) -> Optional[str]:
    """Semantic cache lookup (embedding hash for approximation)."""
    try:
        query_emb = await Settings.embed_model.aget_text_embedding(query)
        cache_key = f"cache:{hash(tuple(round(x, 4) for x in query_emb))}"  # Rounded for similarity
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info("Cache hit", query=query[:50])
            return json.loads(cached)["answer"]
        return None
    except Exception as e:
        logger.error("Cache get failed", error=str(e))
        return None

async def set(query: str, answer: str, ttl: int = 3600):
    """Store with embed hash key."""
    try:
        query_emb = await Settings.embed_model.aget_text_embedding(query)
        cache_key = f"cache:{hash(tuple(round(x, 4) for x in query_emb))}"
        await redis_client.setex(cache_key, ttl, json.dumps({"answer": answer}))
        logger.info("Cache set", query=(query[:50]+"..."))
    except Exception as e:
        logger.error("Cache set failed", error=str(e))