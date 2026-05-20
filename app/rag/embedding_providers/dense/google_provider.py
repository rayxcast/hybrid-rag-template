from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

from .base import BaseDenseEmbedProvider


class GoogleProvider(BaseDenseEmbedProvider):
    def __init__(self):
        from app.config import app_settings

        self.model = app_settings.EMBEDDING_MODEL
        self.batch_size = app_settings.EMBED_BATCH_SIZE
        self.output_dimensionality = app_settings.EMBEDDING_DIM
        self.api_key = app_settings.GOOGLE_API_KEY or None

    def get_dense_model(self) -> GoogleGenAIEmbedding:
        return GoogleGenAIEmbedding(
            model_name=self.model,
            api_key=self.api_key,
            embed_batch_size=self.batch_size,
            embedding_config=EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            ),
        )
