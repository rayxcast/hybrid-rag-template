"""Hybrid RAG query engine: retrieval + generation + flags + fallback."""
from app.config import app_settings, configure_llm_settings
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from app.rag.ingestion import get_vector_store
import yaml
import structlog

logger = structlog.get_logger()

# Load prompts
with open("app/rag/prompts.yaml") as f:
    prompts = yaml.safe_load(f)
qa_prompt = PromptTemplate(prompts["v1"]["qa"])

class Retriever:
    def __init__(self):
        configure_llm_settings()
        self.config = app_settings

    def retrieve(self, query: str):
        try:
            index = VectorStoreIndex.from_vector_store(get_vector_store())
            mode = "hybrid" if self.config.RETRIEVAL_MODE == "hybrid" else "default"
            node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=self.config.SIMILARITY_CUTOFF)]

            retriever = index.as_retriever(
                similarity_top_k=self.config.SIMILARITY_TOP_K,
                node_postprocessors=node_postprocessors,
                vector_store_query_mode=mode
            )
            
            retrieved_nodes = retriever.retrieve(query)
            if retrieved_nodes:
                return retrieved_nodes

        except Exception as e:
            logger.error("retrieval_failed", error=str(e), exc_info=True)

        return []