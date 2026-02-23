import asyncio
import time
import json
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.eval_dataset import EVAL_SET
from app.rag.pipeline import HybridRAG

rag = HybridRAG()
evaluator = RAGEvaluator(rag)

results = []

async def main():
    for case in EVAL_SET:
        start_time = time.time()
        r = await evaluator.evaluate_case(case)
        end_time = time.time()
        results.append(r)
        print(f"Case '{case["question"]}': {r} | execution time: {end_time - start_time:.2f} seconds")


    accuracy = sum(r["passed"] for r in results) / len(results)

    summary = {
        "avg_accuracy": f"{accuracy:.2%}",
        "avg_latency": {
            "retrieval_time": round(sum(m['latency']["retrieval_time"] for m in results) / len(results), 2),
            "rerank_time": round(sum(m['latency']["rerank_time"] for m in results) / len(results), 2),
            "generation_time": round(sum(m['latency']["generation_time"] for m in results) / len(results), 2),
            "judge_time": round(sum(m['latency']["judge_time"] for m in results) / len(results), 2)
        },
        "total_retrieval_recall": sum(1 for m in results if m['retrieval_recall']),
        "total_passed": sum(1 for m in results if m['passed'])
    }
    
    print(f"\nOverall accuracy:", summary)
    to_save = {
        "results": results,
        "overall": summary
    }
    with open("eval_results.json", "w") as f:
        json.dump(to_save, f, indent=2)

if __name__ == "__main__":
    start_time = time.time()
    # This is the entry point for the asyncio event loop
    asyncio.run(main())
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")