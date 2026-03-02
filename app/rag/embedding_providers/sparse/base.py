from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSparseEmbeddingProvider(ABC):

    @abstractmethod
    def embed_documents(
        self, texts: List[str]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        pass

    @abstractmethod
    def embed_query(
        self, texts: List[str]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        pass