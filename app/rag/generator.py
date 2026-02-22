from app.config import configure_llm_settings
from llama_index.core import PromptTemplate
import yaml
import structlog

logger = structlog.get_logger()

# Load prompts
with open("app/rag/prompts.yaml") as f:
    prompts = yaml.safe_load(f)
qa_prompt = PromptTemplate(prompts["v1"]["qa"])

class LLMGenerator:
    def __init__(self):
        configure_llm_settings()
        # Grab the model from the global settings once
        from llama_index.core import Settings 
        self.llm = Settings.llm
    
    def generate(self, query: str, final_nodes: list):
        try:
            if not final_nodes:
                # logger.warning("No relevant nodes retrieved for query", query=query)
                # fallback_response = llm.complete(f"Answer directly: {query}")
                answer = "The answer is not present in the provided documents."
                sources = []
            else:
                # Build simple context from retrieved nodes
                context_str = "\n\n".join([n.node.text for n in final_nodes])
                prompt = qa_prompt.format(context_str=context_str, query_str=query)
                response = self.llm.complete(prompt)
                answer = response.text
                sources = [n.node.metadata for n in final_nodes]

        except Exception as e:
            logger.error("retrieval_failed", error=str(e), exc_info=True)
            answer = "The answer is not present in the provided documents."
            sources = []

        return {"answer": answer, "sources": sources}