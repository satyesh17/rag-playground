"""
Eval runner — unified interface for running any retrieval pipeline against
the golden set and producing structured metrics.

Design: separate "run pipeline" from "score results". This lets us:
- Score the same run with different metrics (cheap vs expensive)
- Save raw results for debugging failed evals
- Add new pipelines without duplicating scoring logic
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

from sentence_transformers import SentenceTransformer

from src.rag_playground.db import get_client
from src.rag_playground.golden_set_adversarial import (
    load_adversarial_golden_set,
)
from src.rag_playground.hybrid_retriever import HybridRetriever
from src.rag_playground.reranker import CrossEncoderReranker

K_VALUES = [1, 3, 5, 10]


@dataclass
class QueryResult:
    question: str
    expected_titles: list[str]
    retrieved_titles: list[str]
    retrieved_texts: list[str]
    latency_ms: float
    category: str = ""

    def hit_at_k(self, k: int) -> bool:
        return any(t in self.expected_titles for t in self.retrieved_titles[:k])


@dataclass
class EvalMetrics:
    """Aggregate metrics across all golden set questions."""

    approach: str
    n_questions: int
    recall_at_k: dict[int, float] = field(default_factory=dict)
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    per_category_recall_at_1: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "n_questions": self.n_questions,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "per_category_recall_at_1": self.per_category_recall_at_1,
        }


def _compute_metrics(approach: str, results: list[QueryResult]) -> EvalMetrics:
    """Aggregate per-question results into summary metrics."""
    recall_at_k = {k: sum(1 for r in results if r.hit_at_k(k)) / len(results) for k in K_VALUES}
    latencies = sorted(r.latency_ms for r in results)
    med = median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]

    # Per-category recall@1 helps diagnose which failure modes regressed
    from collections import defaultdict

    category_hits: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        category_hits[r.category].append(r.hit_at_k(1))
    per_cat = {cat: sum(hits) / len(hits) for cat, hits in category_hits.items()}

    return EvalMetrics(
        approach=approach,
        n_questions=len(results),
        recall_at_k=recall_at_k,
        median_latency_ms=med,
        p95_latency_ms=p95,
        per_category_recall_at_1=per_cat,
    )


def run_dense_only(
    collection: str = "day9_fixed",
) -> tuple[list[QueryResult], EvalMetrics]:
    """Dense-only retrieval baseline."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = get_client()
    golden = load_adversarial_golden_set()

    results = []
    for entry in golden:
        t0 = time.perf_counter()
        qvec = model.encode(entry.question, normalize_embeddings=True).tolist()
        res = client.query_points(
            collection_name=collection, query=qvec, limit=10, with_payload=True
        )
        latency = (time.perf_counter() - t0) * 1000
        results.append(
            QueryResult(
                question=entry.question,
                expected_titles=entry.expected_doc_titles,
                retrieved_titles=[p.payload["doc_title"] for p in res.points],
                retrieved_texts=[p.payload["text"] for p in res.points],
                latency_ms=latency,
                category=entry.category,
            )
        )

    return results, _compute_metrics("dense", results)


def run_hybrid(collection: str = "day9_fixed") -> tuple[list[QueryResult], EvalMetrics]:
    """Hybrid dense + BM25 with RRF."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = HybridRetriever(collection, model)
    golden = load_adversarial_golden_set()

    results = []
    for entry in golden:
        t0 = time.perf_counter()
        res = retriever.retrieve(entry.question, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        results.append(
            QueryResult(
                question=entry.question,
                expected_titles=entry.expected_doc_titles,
                retrieved_titles=[r.doc_title for r in res],
                retrieved_texts=[r.text for r in res],
                latency_ms=latency,
                category=entry.category,
            )
        )

    return results, _compute_metrics("hybrid", results)


def run_hybrid_rerank(
    collection: str = "day9_fixed",
) -> tuple[list[QueryResult], EvalMetrics]:
    """Hybrid retrieval + cross-encoder reranking."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = HybridRetriever(collection, model)
    reranker = CrossEncoderReranker()
    golden = load_adversarial_golden_set()

    results = []
    for entry in golden:
        t0 = time.perf_counter()
        hybrid = retriever.retrieve(entry.question, top_k=20)
        reranked = reranker.rerank(entry.question, hybrid, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        results.append(
            QueryResult(
                question=entry.question,
                expected_titles=entry.expected_doc_titles,
                retrieved_titles=[r.doc_title for r in reranked],
                retrieved_texts=[r.text for r in reranked],
                latency_ms=latency,
                category=entry.category,
            )
        )

    return results, _compute_metrics("hybrid_rerank", results)


APPROACHES = {
    "dense": run_dense_only,
    "hybrid": run_hybrid,
    "hybrid_rerank": run_hybrid_rerank,
}


def run_all_approaches(
    output_dir: Path = Path("data/eval_runs"),
) -> dict[str, EvalMetrics]:
    """Run every approach, save raw results and summary metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, EvalMetrics] = {}

    for name, runner in APPROACHES.items():
        print(f"\n{'=' * 60}\nRunning approach: {name}\n{'=' * 60}")
        results, metrics = runner()

        # Save raw per-question results
        raw_path = output_dir / f"{name}_raw.json"
        with raw_path.open("w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save aggregate metrics
        summary_path = output_dir / f"{name}_metrics.json"
        with summary_path.open("w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        all_metrics[name] = metrics
        print(
            f"  R@1: {metrics.recall_at_k[1]:.2f}, R@5: {metrics.recall_at_k[5]:.2f}, "
            f"latency: {metrics.median_latency_ms:.0f}ms"
        )

    return all_metrics


if __name__ == "__main__":
    metrics = run_all_approaches()

    print(f"\n{'=' * 78}\nEVAL SUMMARY\n{'=' * 78}")
    header = f"{'Approach':<16} " + " ".join(f"R@{k:<2}   " for k in K_VALUES) + "Latency(ms)"
    print(header)
    print("-" * 78)
    for name, m in metrics.items():
        row = f"{name:<16} "
        row += " ".join(f"{m.recall_at_k[k]:.2f}  " for k in K_VALUES)
        row += f"{m.median_latency_ms:6.0f} (median)"
        print(row)
