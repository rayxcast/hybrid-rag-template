from abc import ABC, abstractmethod
from typing import List

class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query: str, nodes, top_n: int = 25) -> List:
        pass

    def supports_batch(self) -> bool:
        return False