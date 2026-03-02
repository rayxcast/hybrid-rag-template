from fastembed import SparseTextEmbedding
from .base import BaseSparseEmbeddingProvider
from typing import List, Tuple

class SparseEmbeddingProvider(BaseSparseEmbeddingProvider):
    def __init__(self):
        from app.config import app_settings
        self.sparse_model = SparseTextEmbedding(model_name=app_settings.SPARSE_MODEL)

    def embed_documents(
        self, texts: List[str]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        embeddings = list(self.sparse_model.embed(texts))
        indices_list = []
        values_list = []

        for emb in embeddings:
            indices_list.append(emb.indices.tolist())
            values_list.append(emb.values.tolist())

        return indices_list, values_list

    def embed_query(
        self, texts: List[str]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        emb = next(self.sparse_model.embed(texts))
        return [emb.indices.tolist()], [emb.values.tolist()]