"""
Benchmark: which embedding model retrieves better on our golden question set?

For each question in the golden set:
1. Embed the question with model X
2. Retrieve top-K chunks from Qdrant collection for model X
3. Check if any retrieved chunk's doc_title matches the expected titles
4. Record hit/miss at K=1, 3, 5, 10

Metrics:
- Recall@K = fraction of questions where a correct chunk appeared in top K
- Latency = wall-clock time per query (median across the set)

Higher recall = better retrieval. Lower latency = faster serving.
The tradeoff is the whole point of this benchmark.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from src.rag_playground.db import get_client
from src.rag_playground.golden_set import GoldenEntry, load_golden_set
from src.rag_playground.indexer import EMBEDDING_MODELS, load_model


K_VALUES = [1, 3, 5, 10]


@dataclass
class QueryResult:
    question: str
    expected: list[str]
    retrieved_titles: list[str]      # Top-10 doc titles retrieved
    latency_ms: float
    hits_at_k: dict[int, bool] = field(default_factory=dict)


def evaluate_model(model_key: str, golden: list[GoldenEntry]) -> dict:
    """Run the golden set through one model and return per-question + summary results."""
    cfg = EMBEDDING_MODELS[model_key]
    model = load_model(model_key)
    client = get_client()

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_key}  (collection: {cfg['collection']})")
    print("=" * 60)

    results: list[QueryResult] = []

    for entry in golden:
        # Embed the query with the SAME model that was used to index
        t0 = time.perf_counter()
        query_vec = model.encode(
            entry.question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Retrieve top-K chunks (we use max K = 10 to compute all recall@K in one query)
        search_result = client.query_points(
            collection_name=cfg["collection"],
            query=query_vec,
            limit=max(K_VALUES),
            with_payload=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Get the doc_title of each retrieved chunk
        retrieved_titles = [p.payload["doc_title"] for p in search_result.points]

        # Compute recall@K for each K
        hits_at_k = {}
        for k in K_VALUES:
            top_k_titles = retrieved_titles[:k]
            hit = any(t in entry.expected_doc_titles for t in top_k_titles)
            hits_at_k[k] = hit

        results.append(QueryResult(
            question=entry.question,
            expected=entry.expected_doc_titles,
            retrieved_titles=retrieved_titles,
            latency_ms=latency_ms,
            hits_at_k=hits_at_k,
        ))

        # Print per-question result for transparency
        top1 = retrieved_titles[0] if retrieved_titles else "(none)"
        marker = "✓" if hits_at_k[1] else "✗"
        print(f"  {marker} [{latency_ms:5.1f}ms] Q: {entry.question[:55]}")
        print(f"         Top-1: {top1}")

    # Compute aggregate metrics
    recall_at_k = {}
    for k in K_VALUES:
        n_hits = sum(1 for r in results if r.hits_at_k[k])
        recall_at_k[k] = n_hits / len(results)

    latencies = [r.latency_ms for r in results]

    summary = {
        "model": model_key,
        "hf_id": cfg["hf_id"],
        "dim": cfg["dim"],
        "n_questions": len(results),
        "recall_at_k": recall_at_k,
        "latency_ms": {
            "median": median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
    }

    return {"summary": summary, "per_question": results}


def print_comparison(summaries: list[dict]) -> None:
    """Pretty-print a side-by-side comparison table."""
    print(f"\n{'=' * 78}")
    print("BENCHMARK COMPARISON")
    print("=" * 78)
    header = f"{'Model':<12} " + " ".join(f"R@{k:<2}   " for k in K_VALUES) + "Latency(ms)"
    print(header)
    print("-" * 78)
    for s in summaries:
        row = f"{s['model']:<12} "
        row += " ".join(f"{s['recall_at_k'][k]:.2f}  " for k in K_VALUES)
        row += f"{s['latency_ms']['median']:6.1f} (median)"
        print(row)
    print("=" * 78)


def save_results(results_by_model: dict, output_path: Path) -> None:
    """Save full results to JSON for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for model_key, data in results_by_model.items():
        serializable[model_key] = {
            "summary": data["summary"],
            "per_question": [
                {
                    "question": r.question,
                    "expected": r.expected,
                    "retrieved_titles": r.retrieved_titles,
                    "latency_ms": r.latency_ms,
                    "hits_at_k": r.hits_at_k,
                }
                for r in data["per_question"]
            ],
        }
    with output_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    golden = load_golden_set()
    print(f"Golden set: {len(golden)} questions\n")

    results_by_model = {}
    summaries = []

    for model_key in ["minilm", "bge_large"]:
        data = evaluate_model(model_key, golden)
        results_by_model[model_key] = data
        summaries.append(data["summary"])

    print_comparison(summaries)
    save_results(results_by_model, Path("data/benchmark_results.json"))
