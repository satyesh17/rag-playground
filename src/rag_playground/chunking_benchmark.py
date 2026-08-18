"""
Day 9 benchmark: which chunking strategy retrieves better?

Embedding model held constant (MiniLM) across all strategies.
Golden set: same 15 questions as Day 8.

Metrics:
- Recall@K for K = 1, 3, 5, 10
- Median query latency
- Average chunk count returned in top-K (proxy for how "spread" the retrieval is)
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from src.rag_playground.chunker import CHUNKING_STRATEGIES
from src.rag_playground.db import get_client
from src.rag_playground.golden_set import GoldenEntry, load_golden_set
from src.rag_playground.indexer import EMBEDDING_MODELS, load_model


K_VALUES = [1, 3, 5, 10]


@dataclass
class QueryResult:
    question: str
    expected: list[str]
    retrieved_titles: list[str]
    latency_ms: float
    hits_at_k: dict[int, bool] = field(default_factory=dict)


def evaluate_strategy(
    strategy: str,
    collection_name: str,
    model,
    golden: list[GoldenEntry],
) -> dict:
    """Run the golden set against one chunking strategy's collection."""
    client = get_client()

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {strategy}  (collection: {collection_name})")
    print("=" * 60)

    results: list[QueryResult] = []

    for entry in golden:
        t0 = time.perf_counter()
        query_vec = model.encode(
            entry.question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vec,
            limit=max(K_VALUES),
            with_payload=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_titles = [p.payload["doc_title"] for p in search_result.points]

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

        marker = "✓" if hits_at_k[1] else "✗"
        top1 = retrieved_titles[0] if retrieved_titles else "(none)"
        print(f"  {marker} [{latency_ms:5.1f}ms] Q: {entry.question[:50]}")
        print(f"         Top-1: {top1}")

    recall_at_k = {
        k: sum(1 for r in results if r.hits_at_k[k]) / len(results)
        for k in K_VALUES
    }
    latencies = [r.latency_ms for r in results]

    summary = {
        "strategy": strategy,
        "description": CHUNKING_STRATEGIES[strategy]["description"],
        "collection": collection_name,
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
    """Pretty-print side-by-side comparison."""
    print(f"\n{'=' * 78}")
    print("CHUNKING STRATEGY COMPARISON (MiniLM held constant)")
    print("=" * 78)
    header = f"{'Strategy':<12} " + " ".join(f"R@{k:<3}" for k in K_VALUES) + "  Latency(ms)"
    print(header)
    print("-" * 78)
    for s in summaries:
        row = f"{s['strategy']:<12} "
        row += " ".join(f"{s['recall_at_k'][k]:.2f} " for k in K_VALUES)
        row += f"  {s['latency_ms']['median']:5.1f} (median)"
        print(row)
    print("=" * 78)


def save_results(results_by_strategy: dict, output_path: Path) -> None:
    """Persist full results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for strategy, data in results_by_strategy.items():
        serializable[strategy] = {
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
    print(f"Golden set: {len(golden)} questions")

    # Load MiniLM once — same model for all strategies
    minilm = load_model("minilm")

    results_by_strategy = {}
    summaries = []

    for strategy in CHUNKING_STRATEGIES:
        collection = f"day9_{strategy}"
        data = evaluate_strategy(strategy, collection, minilm, golden)
        results_by_strategy[strategy] = data
        summaries.append(data["summary"])

    print_comparison(summaries)
    save_results(results_by_strategy, Path("data/chunking_benchmark_results.json"))
