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
    USE_RERANKER: bool = True  # Set True later if adding Cohere/BGE

    SIMILARITY_TOP_K: int = 20
    SIMILARITY_CUTOFF: float = 0.35

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # "BAAI/bge-reranker-base" for better quality, but 1.1 GB
    RERANK_TOP_N: int = 15
    FINAL_CONTEXT_N: int = 7
    
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