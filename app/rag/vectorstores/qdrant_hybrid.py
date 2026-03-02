from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from app.config import app_settings
from .base import BaseVectorStoreProvider

import structlog
logger = structlog.get_logger()

class QdrantHybridStore(BaseVectorStoreProvider):

    def __init__(self, sparse_provider=None):
        self.client = QdrantClient(url=app_settings.QDRANT_URL)
        self.sparse = sparse_provider

    def init_collection_if_needed(self):
        if not self.client.collection_exists(app_settings.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=app_settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=app_settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams()
                },
            )
            logger.info("created_qdrant_collection", name=app_settings.COLLECTION_NAME)

    def get_vector_store(self):
        return QdrantVectorStore(
            client=self.client,
            collection_name=app_settings.COLLECTION_NAME,
            enable_hybrid=True,
            sparse_doc_fn=self.sparse.embed_documents,
            sparse_query_fn=self.sparse.embed_query,
            text_sparse_name="text-sparse",
            use_default_sparse_query_encoder=False,
        )

    def delete_collection(self):
        try:
            self.client.delete_collection(app_settings.COLLECTION_NAME)
            return {
                "deleted": True,
                "collection_name": app_settings.COLLECTION_NAME
            }
        except Exception as error:
            logger.error(
                "delete_collection_failed",
                error=str(error),
                collection=app_settings.COLLECTION_NAME,
            )

        return {
            "deleted": False,
            "collection_name": app_settings.COLLECTION_NAME
        }
    
    def supports_sparse(self) -> bool:
        return True