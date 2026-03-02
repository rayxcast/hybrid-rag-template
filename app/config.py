from pydantic_settings import BaseSettings
from typing import Literal
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from app.rag.embedding_providers.dense.factory import get_dense_provider

class AppSettings(BaseSettings):
    # API
    APP_NAME: str = "Hybrid RAG Template"

    # RAG Settings
    RETRIEVAL_MODE: Literal["dense", "hybrid"] = "hybrid"
    LLM_PROVIDER: Literal["openai", "anthropic", "ollama"] = "openai"
    LLM_MODEL: str = "gpt-4.1-mini"
    USE_RERANKER: bool = True
    USE_CACHE: bool = True

    # Vector storage provider
    VECTOR_STORE_PROVIDER: str = "qdrant"

    # Dense provider
    DENSE_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBED_BATCH_SIZE: int = 128

    CHUNK_SIZE: int = 512 # 512 for better "granularity" for semantic search.
    CHUNK_OVERLAP: int = 100 # 15-20% of CHUNK_SIZE is the gold standard for context continuity.
    EMBEDDING_DIM: int = 1536 # Dimensionality of dense embedding vector; 1536 for OpenAI text-embedding-3-small.

    # Sparse provider
    SPARSE_PROVIDER: str = "fastembed"
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"

    # Reranker provider
    RERANKER_PROVIDER: str = "fastembed"
    RERANKER_MODEL: str = "jinaai/jina-reranker-v1-tiny-en" # "light+fast but English only "jinaai/jina-reranker-v1-turbo-en" vs "BAAI/bge-reranker-base" for balanced production choice Multilingual, but much slower + high size (1.1 GB)

    # Retrieval config
    SIMILARITY_TOP_K: int = 75 # 50 – 100 This is your "Recall" phase. You need enough candidates from both vector and keyword search so the reranker has the "correct" information available to find.
    SIMILARITY_CUTOFF: float = 0.75 # 0.75 is a common industry baseline for "meaningful" similarity in 2026.
    RERANK_TOP_N: int = 25 # 20 – 30 After fusing results, the reranker should evaluate a healthy subset. 15 is slightly narrow; 25 is safer to ensure diverse perspectives are captured before final selection.
    FINAL_CONTEXT_N: int = 7 # 5 – 10 Most modern LLMs perform best with 5–10 highly relevant chunks. Too many chunks can lead to "Lost in the Middle" errors.
    
    # Evals config
    EVAL_LLM_MODEL: str = "gpt-4.1-nano"
    EVAL_LLM_PROVIDER: str = "openai"

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
        temperature=1
    )
 
    Settings.embed_model = get_dense_provider(app_settings.DENSE_PROVIDER).get_dense_model()