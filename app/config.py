from pydantic_settings import BaseSettings
from typing import Literal
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.openai import OpenAIEmbedding

class AppSettings(BaseSettings):
    # API
    APP_NAME: str = "Hybrid RAG Template"

    # RAG Settings
    RETRIEVAL_MODE: Literal["dense", "hybrid"] = "hybrid"
    LLM_PROVIDER: Literal["openai", "anthropic", "ollama"] = "openai"
    LLM_MODEL: str = "gpt-5-nano"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBED_BATCH_SIZE: int = 128
    USE_RERANKER: bool = True
    USE_CACHE: bool = True

    SIMILARITY_TOP_K: int = 20
    SIMILARITY_CUTOFF: float = 0.35

    # RERANK MODELS
    # Model,Size,Speed,Quality (approx. relative),Multilingual,License,Best for
    # Xenova/ms-marco-MiniLM-L-6-v2,~80 MB,Very fast,Good / baseline,English-focused,Apache 2.0,"Latency-critical, small infra"
    # Xenova/ms-marco-MiniLM-L-12-v2,~120 MB,Fast,Good+,English-focused,Apache 2.0,"Slightly better quality, still fast"
    # jinaai/jina-reranker-v1-tiny-en,~130 MB,Very fast,Good,English,Apache 2.0,Ultra-low latency English
    # jinaai/jina-reranker-v1-turbo-en,~150 MB,Fast,Good+,English,Apache 2.0,Fast English with better quality
    # BAAI/bge-reranker-base,~1.04 GB,Medium,Very good,Strong multi,MIT,Balanced production choice
    # jinaai/jina-reranker-v2-base-multilingual,~1.1 GB,Medium,Excellent,Very strong,CC-BY-NC-4.0,Multilingual production (non-commercial only if strict)
    RERANK_MODEL: str = "jinaai/jina-reranker-v1-tiny-en" # "light+fast but English only "jinaai/jina-reranker-v1-turbo-en" vs "BAAI/bge-reranker-base" for balanced production choice Multilingual, but much slower + high size (1.1 GB)
    RERANK_TOP_N: int = 15
    FINAL_CONTEXT_N: int = 6
    
    # Services
    QDRANT_URL: str = "http://qdrant:6333"
    REDIS_URL: str = "redis://redis:6379/0"
    COLLECTION_NAME: str = "hybrid_rag_docs"
    
    # Keys (from .env)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLAMA_CLOUD_API_KEY: str | None = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

app_settings = AppSettings()

def configure_llm_settings():
    """Applies global LlamaIndex configuration."""
    Settings.llm = LiteLLM(
        model=f"{app_settings.LLM_PROVIDER}/{app_settings.LLM_MODEL}",
        api_key=app_settings.OPENAI_API_KEY if app_settings.LLM_PROVIDER == "openai" else (app_settings.ANTHROPIC_API_KEY if app_settings.LLM_PROVIDER == "anthropic" else app_settings.LLAMA_CLOUD_API_KEY),
        additional_kwargs={"drop_params": True} 
    )
    Settings.embed_model = OpenAIEmbedding(
        model=app_settings.EMBEDDING_MODEL,
        embed_batch_size=app_settings.EMBED_BATCH_SIZE,  # Critical for ingest speed
    )