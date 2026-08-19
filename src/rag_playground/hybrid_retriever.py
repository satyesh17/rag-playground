"""
Hybrid retriever combining dense (Qdrant) + sparse (BM25) via RRF.

Reciprocal Rank Fusion (RRF):
    For each candidate, RRF score = sum over lists of 1 / (k + rank)
    - Rank starts at 1 (not 0)
    - k is a constant, typically 60
    - Documents in higher ranks in more lists win

Why RRF:
- No score calibration needed (dense = 0-1, BM25 = unbounded)
- No hyperparameters to tune
- Simple and reproducible

Trade-offs:
- Ignores actual score magnitudes (only uses ranks)
- Can't weight one retriever over another
- Simpler alternatives (weighted sum) can outperform when scores are well-calibrated
"""

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.rag_playground.db import get_client
from src.rag_playground.sparse_retriever import BM25Retriever


@dataclass
class HybridResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    rrf_score: float
    dense_rank: int | None  # None if not retrieved by dense
    sparse_rank: int | None  # None if not retrieved by BM25


class HybridRetriever:
    """Combines dense (Qdrant) and sparse (BM25) via Reciprocal Rank Fusion."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: SentenceTransformer,
        candidates_per_retriever: int = 50,
        rrf_k: int = 60,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.qdrant = get_client()
        self.bm25 = BM25Retriever(collection_name)
        self.candidates_per_retriever = candidates_per_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> list[HybridResult]:
        """Retrieve top-K after fusing dense + sparse via RRF."""
        # Dense retrieval
        query_vec = self.embedding_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        dense_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=self.candidates_per_retriever,
            with_payload=True,
        )

        # Sparse retrieval
        sparse_result = self.bm25.retrieve(query, top_k=self.candidates_per_retriever)

        # Build rank maps keyed by chunk_id
        dense_ranks: dict[str, int] = {}
        chunk_data: dict[str, dict] = {}  # chunk_id -> payload

        for rank, point in enumerate(dense_result.points, start=1):
            chunk_id = point.payload["chunk_id"]
            dense_ranks[chunk_id] = rank
            chunk_data[chunk_id] = point.payload

        sparse_ranks: dict[str, int] = {}
        for rank, r in enumerate(sparse_result, start=1):
            sparse_ranks[r.chunk_id] = rank
            if r.chunk_id not in chunk_data:
                # Chunk found only by BM25 — reconstruct payload from result
                chunk_data[r.chunk_id] = {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "doc_title": r.doc_title,
                    "text": r.text,
                }

        # Compute RRF scores
        all_ids = set(dense_ranks) | set(sparse_ranks)
        rrf_scores: dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score descending, take top-K
        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for chunk_id in top_ids:
            payload = chunk_data[chunk_id]
            results.append(
                HybridResult(
                    chunk_id=chunk_id,
                    doc_id=payload["doc_id"],
                    doc_title=payload["doc_title"],
                    text=payload["text"],
                    rrf_score=rrf_scores[chunk_id],
                    dense_rank=dense_ranks.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                )
            )
        return results


if __name__ == "__main__":
    # Smoke test
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = HybridRetriever("day9_fixed", model)

    test_queries = [
        "What does the acronym TAI stand for in metrology?",  # BM25 should win
        "Which conflict of the 1860s reshaped American federalism?",  # Dense should win
        "Who founded the school of thought based on syllogistic logic?",  # Both miss?
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(
                f"  {r.rrf_score:.4f} | dense={r.dense_rank or '-'} sparse={r.sparse_rank or '-'} | "
                f"{r.doc_title}"
            )
