from llama_index.embeddings.openai import OpenAIEmbedding
from .base import BaseDenseEmbedProvider

class OpenAIProvider(BaseDenseEmbedProvider):
    def __init__(self):
        from app.config import app_settings
        self.model = app_settings.EMBEDDING_MODEL
        self.batch_size = app_settings.EMBED_BATCH_SIZE

    def get_dense_model(self) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            model=self.model,
            embed_batch_size=self.batch_size,  # Critical for ingest speed
        )