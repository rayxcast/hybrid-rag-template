from app.config import app_settings
from .qdrant_hybrid import QdrantHybridStore
from app.rag.embedding_providers.sparse.factory import get_sparse_provider

def get_vector_store_provider():
    provider = app_settings.VECTOR_STORE_PROVIDER

    if provider == "qdrant":
        return QdrantHybridStore(sparse_provider=get_sparse_provider(app_settings.SPARSE_PROVIDER))

    raise ValueError(f"Unsupported vector store provider: {provider}")