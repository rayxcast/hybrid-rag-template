from abc import ABC, abstractmethod
from typing import Any

class BaseDenseEmbedProvider(ABC):

    @abstractmethod
    def get_dense_model(self) -> Any: 
        pass
