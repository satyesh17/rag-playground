# RAG Retrieval Benchmark — Day 8

## Setup

- **Corpus:** ~14,000 chunks from 500 Wikipedia articles
- **Chunking:** Fixed-size 500 chars, 50 char overlap
- **Vector DB:** Qdrant (local Docker), cosine similarity
- **Questions:** 15 hand-crafted golden set, mixed categories (definitional, historical, causal, descriptive)

## Models Compared

| Model | Dimensions | Model Size | HF ID |
|---|---|---|---|
| MiniLM | 384 | ~90MB | sentence-transformers/all-MiniLM-L6-v2 |
| BGE-large | 1024 | ~1.3GB | BAAI/bge-large-en-v1.5 |

## Results

| Model | R@1 | R@3 | R@5 | R@10 | Latency (median) |
|---|---|---|---|---|---|
| MiniLM | 1.00 | 1.00 | 1.00 | 1.00 | 21ms |
| BGE-large | 0.93 | 1.00 | 1.00 | 1.00 | 90ms |

## Analysis

**Surprise:** MiniLM matched or beat BGE-large on this benchmark despite BGE being the "better" model per published leaderboards.

**Why:** The golden set uses questions with vocabulary distinctive to each source article (e.g. "Anatoly Karpov", "A Modest Proposal"). Both models retrieve accurately when queries share surface features with the source. BGE's semantic-understanding advantage would show up on harder retrieval:
- Paraphrased queries ("which Soviet grandmaster dominated 1970s chess?")
- Multi-hop questions
- Ambiguous or noisy queries

**Production implication:** For queries that closely resemble source vocabulary, MiniLM at 4x the speed is the right choice. BGE's advantage justifies its cost only for harder query patterns.

## Failure Analysis

_[Add the BGE miss details once you have them from the diagnostic script.]_

## Next Steps

- Day 9: Benchmark 4 chunking strategies (fixed, recursive, semantic, document-aware)
- Day 10: Add hybrid search (BM25 + dense) with reranking
- Day 11: Wire RAGAS evals into CI

## Reproduce

```bash
git clone https://github.com/satyesh17/rag-playground
cd rag-playground
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Download corpus, chunk, index, benchmark
python -m src.rag_playground.data_loader
python -m src.rag_playground.indexer
python -m src.rag_playground.benchmark
```
## Failure Analysis (BGE @ K=1)

BGE-large missed only one question at K=1: "Who was Alexander the Great and what did he conquer?"

Top-5 retrieved:
1. Anatolia
2. ✓ Alexander the Great
3. Aegean Sea
4. ✓ Alexander the Great
5. ✓ Alexander the Great

**Not a bug — a signal.** BGE interpreted "what did he conquer" as pointing at regions he conquered, not the biography. Anatolia (major Alexander conquest, mentioned prominently in the Anatolia article) surfaced first.

MiniLM prioritized surface match to "Alexander the Great" and returned the biography. For a biographical lookup, MiniLM's answer is more useful. For a research assistant exploring related context, BGE's answer is arguably richer.

**Lesson:** "Correct" retrieval is application-dependent. Semantic understanding can appear as a "miss" when the metric assumes surface match. Real RAG evaluation needs application-specific gold labels, not just topic labels.
EOF
---

# Day 9 — Chunking Strategy Benchmark

## Setup

- **Corpus:** All 480 Wikipedia articles
- **Embedding model:** MiniLM (held constant)
- **Vector DB:** Qdrant, cosine similarity
- **Questions:** Same 15-question golden set as Day 8
- **Size clamping:** min 100 chars, max 1500 chars

## Strategies Compared

| Strategy | Boundary rule | Chunks produced |
|---|---|---|
| Fixed | 500 chars with 50-char overlap | 28,151 |
| Recursive | Paragraphs → lines → sentences → words | 35,744 |
| Sentence | 5 sentences per chunk, 1 sentence overlap | 23,240 |
| Paragraph | 2 paragraphs per chunk | 15,383 |

## Results

| Strategy | R@1 | R@3 | R@5 | R@10 | Latency (median) |
|---|---|---|---|---|---|
| Fixed | 1.00 | 1.00 | 1.00 | 1.00 | 19ms |
| Recursive | 1.00 | 1.00 | 1.00 | 1.00 | 31ms |
| Sentence | 1.00 | 1.00 | 1.00 | 1.00 | 17ms |
| Paragraph | 1.00 | 1.00 | 1.00 | 1.00 | 17ms |

## Analysis — Recall Tied, Latency Differed

All four strategies achieve perfect recall on the golden set. The retrieval task is easy: golden set queries share vocabulary with source article titles, so any reasonable chunking finds the right document.

**Where they differ:**

- **Recursive is 60% slower** (31ms vs 17-19ms baseline)
- **Sentence and paragraph** produce fewer, larger chunks → smaller index → faster lookup
- **Fixed** is middle ground

## Real Takeaway

Chunking strategy is a latency/storage decision when queries are easy, and a quality decision when queries are hard.

The Day 9 golden set is too easy to differentiate strategies on recall. Fixed-size 500-char remains the sensible default: consistent, fast, no bug-prone splitting logic. Where chunking WILL matter (Day 10+):

- Paraphrased queries (where surface vocabulary doesn't match source)
- Multi-hop questions (answer requires combining chunks)
- Long-form content where answers span chunk boundaries

## The Learning About Benchmarks

Two days of RAG benchmarks, both showed ceiling effects (Day 8: models tied at 1.00, Day 9: strategies tied at 1.00). Pattern: **when your benchmark ceilings, you don't have a result — you have a signal that your test isn't hard enough.**

Day 10 will introduce a harder golden set with:
- Paraphrased queries (no vocabulary overlap with source)
- Multi-hop questions
- Reranking to compress top-K

That's where real differentiation will emerge.

## Next

- Day 10: hybrid search (BM25 + dense) with Cohere reranking
- Day 11: RAGAS evals in CI with fail-build gates
