# rag/vectorstores/base.py

from abc import ABC, abstractmethod
from llama_index.core.vector_stores.types import BasePydanticVectorStore

class BaseVectorStoreProvider(ABC):

    @abstractmethod
    def get_vector_store(self) -> BasePydanticVectorStore:
        pass

    @abstractmethod
    def supports_sparse(self) -> bool:
        pass

    def init_collection_if_needed(self):
        """Optional lifecycle hook"""
        pass

    def delete_collection(self):
        """Optional lifecycle hook"""
        pass