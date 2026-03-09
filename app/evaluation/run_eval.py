import asyncio
import time
import json
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.eval_dataset import EVAL_SET
from app.rag.pipeline import HybridRAG

rag = HybridRAG()
evaluator = RAGEvaluator(rag)

CONCURRENCY_LIMIT = 16  # Adjust based on your machine / reranker capacity

async def evaluate_with_timer(semaphore, case):
    async with semaphore:
        start_time = time.time()
        try:
            result = await evaluator.evaluate_case(case)
            latency = time.time() - start_time
            print(f"Case '{case['question']}' | execution time: {latency:.2f} seconds")
            return result
        except Exception as exc:
            latency = time.time() - start_time
            print(f"Error in case '{case['question']}': {exc} | time: {latency:.2f} seconds")
            return None  # or a failed result dict if you prefer

async def main():
    start_time_global = time.time()

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Create tasks for all cases
    tasks = [evaluate_with_timer(semaphore, case) for case in EVAL_SET]

    # Run concurrently
    completed = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter out None (failed) cases and collect valid results
    results = [r for r in completed if r is not None]

    if not results:
        print("No successful evaluations completed.")
        return

    # Calculate summary
    accuracy = sum(r["passed"] for r in results) / len(results)

    summary = {
        "avg_accuracy": f"{accuracy:.2%}",
        "avg_latency": {
            "retrieval_time": round(sum(m['latency']["retrieval"] for m in results) / len(results), 2),
            "rerank_time": round(sum(m['latency']["rerank"] for m in results) / len(results), 2),
            "generation_time": round(sum(m['latency']["generation"] for m in results) / len(results), 2),
            "judge_time": round(sum(m['latency']["judge"] for m in results) / len(results), 2),
        },
        "total_retrieval_recall": sum(1 for m in results if m.get('retrieval_recall', False)),
        "total_passed": sum(1 for m in results if m['passed']),
        "num_cases_evaluated": len(results),
        "num_cases_failed": len(EVAL_SET) - len(results),
    }

    print("\n" + "="*50)
    print("EVALUATION SUMMARY (parallel, max concurrency = 16)")
    print(f"Overall accuracy:      {summary['avg_accuracy']}")
    print(f"Cases evaluated:        {summary['num_cases_evaluated']}/{len(EVAL_SET)}")
    print(f"Cases failed:           {summary['num_cases_failed']}")
    print("Average latencies:")
    for key, val in summary["avg_latency"].items():
        print(f"  {key:18}: {val:>5.2f} s")
    print(f"Total passed:           {summary['total_passed']}/{len(results)}")
    print("="*50)

    # Save results
    to_save = {
        "results": results,
        "overall": summary,
        "metadata": {
            "concurrency_limit": CONCURRENCY_LIMIT,
            "total_time_seconds": round(time.time() - start_time_global, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    try:
        with open("eval_results/eval_results_parallel.json", "w") as f:
            json.dump(to_save, f, indent=2)
        print("Results saved to eval_results/eval_results_parallel.json")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

    total_time = time.time() - start_time_global
    print(f"Total execution time (parallel): {total_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())