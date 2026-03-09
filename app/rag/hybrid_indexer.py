
from llama_index.core import VectorStoreIndex, StorageContext
from app.rag.vectorstores.factory import get_vector_store_provider

class HybridIndexer:
    def __init__(self):
        self.store_provider = get_vector_store_provider()
        self.vector_store = self.store_provider.get_vector_store()

    def build_index(self, nodes):
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
            use_async=True
        )