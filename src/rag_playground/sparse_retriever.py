"""
BM25 sparse retriever.

BM25 doesn't understand semantics — it scores documents by keyword overlap
weighted by term rarity. Fast, deterministic, zero ML inference cost.

Design:
- Load chunks from Qdrant (same source as dense retrieval)
- Build BM25 index once at construction
- Score against query at retrieval time
- Return top-K chunks by BM25 score

Trade-offs vs dense:
- Wins on exact keyword matches (acronyms, formulas, proper nouns)
- Loses on paraphrases and semantic reasoning
- Zero embedding cost — pure text math
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.rag_playground.db import get_client


@dataclass
class SparseResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    score: float


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-word chars."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """BM25 retriever over chunks stored in Qdrant."""

    def __init__(self, collection_name: str):
        """Load all chunks from a Qdrant collection and build BM25 index."""
        self.collection_name = collection_name
        client = get_client()

        # Load ALL chunks — scroll through pages
        print(f"Loading chunks from {collection_name} for BM25 index...")
        all_chunks = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for sparse
            )
            all_chunks.extend(batch)
            if offset is None:
                break

        # Store payloads and build tokenized corpus
        self.payloads = [p.payload for p in all_chunks]
        tokenized_corpus = [_tokenize(p["text"]) for p in self.payloads]

        print(f"  Building BM25 index over {len(tokenized_corpus)} chunks...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("  Done.")

    def retrieve(self, query: str, top_k: int = 10) -> list[SparseResult]:
        """Return top-K chunks scored by BM25."""
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            p = self.payloads[idx]
            results.append(
                SparseResult(
                    chunk_id=p["chunk_id"],
                    doc_id=p["doc_id"],
                    doc_title=p["doc_title"],
                    text=p["text"],
                    score=float(scores[idx]),
                )
            )
        return results


if __name__ == "__main__":
    # Quick smoke test
    retriever = BM25Retriever("day9_fixed")

    test_queries = [
        "What does TAI stand for?",
        "Who wrote Nicomachean Ethics?",
        "E=mc squared",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  {r.score:6.2f} | {r.doc_title:40s} | {r.text[:50]}...")
