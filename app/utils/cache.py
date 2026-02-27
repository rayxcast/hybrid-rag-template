"""Real Redis semantic cache with vector similarity."""
import numpy as np  # Add this import at top of cache.py
from llama_index.core import Settings
from app.config import app_settings, configure_llm_settings
import redis.asyncio as redis
import structlog
import json
from typing import Optional, Tuple
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
# from redisvl.query.filter import Tag  # if you ever add metadata filters
import uuid
import re

logger = structlog.get_logger()

redis_client = redis.from_url(app_settings.REDIS_URL, decode_responses=True)
configure_llm_settings()

# Normalize query: lowercase, strip punctuation/spaces (cheap boost to hit rate)
def normalize_query(query: str) -> str:
    query = query.lower().strip()
    query = re.sub(r'[?.!,;:"]+', '', query)  # Remove common punctuation
    query = re.sub(r'\s+', ' ', query)  # Collapse spaces
    return query

# Define schema (dims match your embedding model, e.g., 1536 for OpenAI text-embedding-3-small)
SCHEMA = IndexSchema.from_dict({
    "index": {"name": "semantic_cache", "prefix": "cache:"},
    "fields": [
        {"name": "query_text", "type": "text"},
        {"name": "answer", "type": "text"},
        {"name": "embedding", "type": "vector", "attrs": {
            "dims": 1536,
            "distance_metric": "cosine",
            "algorithm": "hnsw",
            "datatype": "float32"
        }}
    ]
})

async def get_connected_index() -> AsyncSearchIndex:
    """Create index and await client connection."""
    index = AsyncSearchIndex(SCHEMA)
    await index.set_client(redis_client)  # ← This is async → MUST await
    return index

# Create index (run once at app startup)
async def init_cache_index():
    try:
        index = await get_connected_index()
        await index.create(overwrite=False)  # Idempotent
        logger.info("Semantic cache index initialized or already exists")
    except Exception as e:
        logger.error("Failed to initialize semantic cache index", error=str(e))
        raise

async def get_semantic(query: str, threshold: float = 0.92) -> Optional[Tuple[dict, float]]:
    """Semantic cache lookup with vector similarity."""
    try:
        norm_query = normalize_query(query)
        q_emb = await Settings.embed_model.aget_text_embedding(norm_query)
        
        # Create VectorQuery
        vector_query = VectorQuery(
            vector=q_emb,
            vector_field_name="embedding",
            return_fields=["answer"],
            num_results=1,
            return_score=True,          # Ensures vector_distance is returned
            # No distance_metric here – it's in schema
        )
        
        index = await get_connected_index()
        results = await index.query(vector_query)
        
        # logger.info("Raw cache results", results=results)  # ← Keep temporarily for debug
        
        if results and len(results) > 0:
            top_result = results[0]  # dict
            # Get distance (string → float)
            distance_str = top_result.get("vector_distance")
            if distance_str is None:
                logger.warning("No vector_distance in result", result=top_result)
                return None, 0.0
            
            distance = float(distance_str)
            
            # Cosine distance → similarity (0–1, higher better)
            similarity = 1 - (distance / 2) if distance <= 2 else 0.0
            
            if similarity >= threshold:
                logger.info(
                    "Semantic cache hit",
                    query=query[:50],
                    distance=distance,
                    similarity=similarity
                )
                cached_data = json.loads(top_result["answer"])
                return cached_data, similarity  # return similarity score
            
            else:
                logger.debug("Cache miss - similarity too low", similarity=similarity)
        
        logger.debug("Semantic cache miss - no results")
        return None, 0.0
    
    except Exception as e:
        logger.error("Semantic cache get failed", error=str(e), exc_info=True)
        return None, 0.0

async def set_semantic(query: str, answer: str, ttl: int = 3600):
    """Store with vector embedding for similarity search."""
    try:
        norm_query = normalize_query(query)
        emb_list = await Settings.embed_model.aget_text_embedding(norm_query)
        
        # Convert list[float] → bytes (required for Redis vector field)
        emb_bytes = np.array(emb_list, dtype=np.float32).tobytes()
        
        key = f"cache:{uuid.uuid4().hex[:12]}"
        index = await get_connected_index()
        await index.load([{
            "id": key,
            "query_text": norm_query,
            "answer": json.dumps(answer),  # still JSON string
            "embedding": emb_bytes       # ← now bytes, not list
        }])
        await redis_client.expire(key, ttl)
        logger.info("Semantic cache set", query=query[:50] + "...")
    except Exception as e:
        logger.error("Semantic cache set failed", error=str(e))