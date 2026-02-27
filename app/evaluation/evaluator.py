import re
import json
import time
from typing import Dict, Any, List, Optional
from app.config import app_settings, configure_llm_settings
import structlog

logger = structlog.get_logger()

class RAGEvaluator:
    def __init__(self, rag_pipeline):
        configure_llm_settings()
        self.rag = rag_pipeline
        from llama_index.core import Settings
        self.llm = Settings.llm

    # ----------------------------
    # Utility
    # ----------------------------

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", "", text)   # remove ALL whitespace
        return text

    @staticmethod
    def parse_llm_json_response(response_content: str) -> dict | None:
        # 1. Clean the response: remove markdown formatting
        cleaned_content = response_content.replace('```json', '').replace('```', '').strip()
        
        # 2. Extract the JSON block
        match = re.search(r'\{[\s\S]*\}', cleaned_content)
        
        if match:
            json_string = match.group(0)
            
            # FIX: Replace Python-style literals with JSON-style literals
            # Be careful not to replace these if they are inside quotes as strings
            json_string = json_string.replace(': True', ': true').replace(': False', ': false').replace(': None', ': null')
            
            try:
                # 3. Parse the JSON string
                return json.loads(json_string)
            except json.JSONDecodeError:
                # If the simple replace failed, try a more robust approach using ast.literal_eval
                # which natively understands Python-style True/False/None
                try:
                    import ast
                    return ast.literal_eval(json_string)
                except (ValueError, SyntaxError):
                    print("Error decoding JSON/Python dict from response.")
                    return None
        else:
            print("No JSON object found in the response.")
            return None

        
    # ----------------------------
    # Retrieval Recall
    # ----------------------------

    def retrieval_recall(
        self,
        retrieved_nodes: List[Any],
        expected_substring: Optional[str],
    ) -> Optional[bool]:

        if not expected_substring:
            return True

        expected_norm = self.normalize(expected_substring)

        for node in retrieved_nodes:
            chunk = self.normalize(node.node.text)
            if expected_norm in chunk:
                return True

        return False

    # ----------------------------
    # faithfulness evaluation with LLM as judge
    # ----------------------------

    def faithfulness(self, question, answer, retrieved_nodes, should_refuse=False):
        context = "\n\n".join([n.node.text for n in retrieved_nodes[:app_settings.FINAL_CONTEXT_N]])
        #    {"" if not should_refuse else "Does the answer correctly refuse due to lack of evidence?"}
        prompt = f"""
        You are evaluating factual grounding.
        1. Based only on the provided context, is the answer correct and sufficiently complete?
        2. {"Is every fact in the answer directly supported by the provided context?" if not should_refuse else "Does the answer correctly refuse due to lack of evidence?"}
        
        Return your final answer strictly in JSON:
        {{
            "passed": True or False
        }}

        Question:
        {question}

        Context:
        {context}

        Answer:
        {answer}
        """

        response = self.llm.complete(prompt)
        respJSON = self.parse_llm_json_response(response.text)
        # logger.info("parse_llm_json_response", response=response.text, respJSON=respJSON)

        if respJSON:
            passed = respJSON["passed"]
            if isinstance(passed, bool):
                return passed
            elif isinstance(passed, str):
                passed = passed.strip().lower()
                return True if passed == "true" else False

        passed = response.text.strip().lower()    
        return True if "true" in passed else False

    # ----------------------------
    # Full Case Evaluation
    # ----------------------------

    async def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.rag.query(
            case["question"],
            cache=False,  # Important for deterministic eval
            return_metadata=True
        )

        answer = result["answer"]
        retrieved_nodes = result["reranked_nodes"] if app_settings.USE_RERANKER and result["reranked_nodes"] else result["retrieved_nodes"] 

        retrieval = self.retrieval_recall(
            retrieved_nodes,
            case.get("expected_contains"),
        )

        # ----------------------------
        # PASS LOGIC
        # ----------------------------
        start = time.time()
        faithfulness = self.faithfulness(
            case["question"],
            answer,
            retrieved_nodes,
            case["should_refuse"]
        )
        judge_time = time.time() - start

        return {
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "retrieval_recall": retrieval,
            "latency": {
                **result["latency"],
                "judge_time": round(judge_time, 2),
            },
            "passed": faithfulness,
        }