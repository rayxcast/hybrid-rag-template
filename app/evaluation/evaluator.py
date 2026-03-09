import re
import json
import time
from typing import Dict, Any, List, Optional
from llama_index.llms.litellm import LiteLLM
from app.config import app_settings
import structlog
import uuid

logger = structlog.get_logger()

class RAGEvaluator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.llm = LiteLLM(
            model=f"{app_settings.EVAL_LLM_PROVIDER}/{app_settings.EVAL_LLM_MODEL}",
            api_key=app_settings.OPENAI_API_KEY if app_settings.LLM_PROVIDER == "openai" else (app_settings.ANTHROPIC_API_KEY if app_settings.LLM_PROVIDER == "anthropic" else app_settings.LLAMA_CLOUD_API_KEY),
            temperature=1
        )

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

    def llm_as_judge(self, question, answer, retrieved_nodes, should_refuse=False):
        context = "\n\n".join([n.node.text for n in retrieved_nodes[:app_settings.FINAL_CONTEXT_N]])
        prompt = f"""
            You are an expert AI Auditor specializing in hybrid Retrieval-Augmented Generation (RAG) systems. 
            Your goal is to evaluate the quality and factual grounding of a response based on a specific Query and the provided Context. Based only on the provided context, is the answer correct and sufficiently complete?

            ### Evaluation Criteria
            1. **Faithfulness (0.0 - 1.0):** Are all claims in the answer supported by the context? 
            2. **Answer Relevance (0.0 - 1.0):** Does the answer address the user's intent? 
            3. **Context Relevance (0.0 - 1.0):** Was the retrieved context necessary and sufficient? 

            ### Output Format
            Return ONLY a JSON object with this schema:
            {{
                "reasoning": "A concise step-by-step explanation of your judgment.",
                "faithfulness": float,
                "answer_relevance": float,
                "context_relevance": float,
                "passed": boolean
            }}

            ### Input Data
            - Query: {question}
            - Context: {context}
            - Generated Answer: {answer}
        """

        response = self.llm.complete(prompt)
        return self.parse_llm_json_response(response.text)

    # ----------------------------
    # Full Case Evaluation
    # ----------------------------

    async def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        try: 
            trace_id = str(uuid.uuid4())
            result = await self.rag.query(
                case["question"],
                trace_id,
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
            eval = self.llm_as_judge(
                case["question"],
                answer,
                retrieved_nodes,
                case["should_refuse"]
            )
            judge_time = time.time() - start
            
            passed = 0
            score = 0

            try:
                score = (eval["faithfulness"]*0.5)+(eval["answer_relevance"]*0.3)+(eval["context_relevance"]*0.2)
                passed = eval["passed"] and score >= 0.8
            except Exception as error:
                logger.error("failed_eval_passed_calculation", error=error, question=case["question"])

            return {
                "id": case["id"],
                "question": case["question"],
                "answer": answer,
                "retrieval_recall": retrieval,
                "latency": {
                    **result["latency"],
                    "judge": round(judge_time, 2),
                },
                "eval": eval,
                "score": score,
                "passed": passed,
            }
        except Exception as error:
            logger.error("failed_eval", error=error, question=case["question"])
            raise error