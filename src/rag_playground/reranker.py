"""
Cross-encoder reranker.

A cross-encoder reads the QUERY and each CANDIDATE together and scores
their relevance. This is fundamentally different from bi-encoders (which
embed query and candidate independently):

- Bi-encoder (used in dense retrieval): query and candidate get embedded
  separately, then compared. Fast — can pre-embed the entire corpus.
  Weakness: no query-candidate interaction during embedding.

- Cross-encoder (used here for reranking): query and candidate feed into
  the same model together. Slower — every (query, candidate) pair needs
  its own forward pass. Strength: rich query-candidate interaction.

Why this matters:
Cross-encoders can distinguish "chunk mentions X" from "chunk is about X".
That's exactly the problem hybrid retrieval couldn't solve.

Trade-off:
Retrieving with cross-encoders over millions of docs is impractical (one
forward pass per doc). So the production pattern is:
  1. Retrieve top-K with bi-encoders/BM25 (fast, imperfect)
  2. Rerank top-K with cross-encoder (slow, precise)
"""
from dataclasses import dataclass

from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    original_rank: int
    reranker_score: float
    reranked_rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker for scoring (query, candidate) pairs."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        print(f"Loading cross-encoder reranker: {model_name}...")
        # First call downloads the model (~2GB), subsequent calls use HF cache
        self.model = CrossEncoder(model_name)
        print("  Ready.")

    def rerank(
        self,
        query: str,
        candidates: list,   # list of objects with .text, .chunk_id, .doc_id, .doc_title
        top_k: int = 10,
    ) -> list[RerankedResult]:
        """
        Rescore candidates by query relevance, return top-K by new score.

        `candidates` should be anything with these attributes: chunk_id, doc_id,
        doc_title, text. HybridResult from hybrid_retriever.py fits perfectly.
        """
        if not candidates:
            return []

        # Build (query, candidate_text) pairs
        pairs = [(query, c.text) for c in candidates]

        # Score all pairs in one batched call
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to candidates and sort descending
        indexed = list(enumerate(zip(candidates, scores, strict=True)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)

        results = []
        for new_rank, (original_rank, (candidate, score)) in enumerate(indexed[:top_k], start=1):
            results.append(RerankedResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                doc_title=candidate.doc_title,
                text=candidate.text,
                original_rank=original_rank + 1,
                reranker_score=float(score),
                reranked_rank=new_rank,
            ))
        return results


if __name__ == "__main__":
    # Smoke test — rerank hybrid retrieval results
    from sentence_transformers import SentenceTransformer
    from src.rag_playground.hybrid_retriever import HybridRetriever

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    retriever = HybridRetriever("day9_fixed", model)
    reranker = CrossEncoderReranker()

    query = "Where did Aristotle's most famous student wage military campaigns?"
    print(f"\nQuery: {query}")

    hybrid_results = retriever.retrieve(query, top_k=10)
    print("\nHybrid top-5 (before rerank):")
    for r in hybrid_results[:5]:
        print(f"  RRF={r.rrf_score:.4f} | {r.doc_title}")

    reranked = reranker.rerank(query, hybrid_results, top_k=5)
    print("\nAfter reranking (top-5):")
    for r in reranked:
        print(f"  rerank_score={r.reranker_score:+.3f} | orig_rank={r.original_rank:2d} → new_rank={r.reranked_rank} | {r.doc_title}")
