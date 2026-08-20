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

---

# Day 10 — Hybrid Search + Reranking (with adversarial golden set)

## Setup

- **Corpus:** 480 Wikipedia articles, 28,151 fixed-size chunks
- **Embedding model:** MiniLM (all-MiniLM-L6-v2)
- **BM25:** rank-bm25, tokenized on words, lowercase
- **Hybrid fusion:** Reciprocal Rank Fusion (k=60)
- **Reranker:** BAAI/bge-reranker-v2-m3 (cross-encoder)
- **Golden set:** 14 adversarial questions across 6 categories designed to break dense-only retrieval

## Results

| Approach | R@1 | R@3 | R@5 | R@10 | Latency (median) |
|---|---|---|---|---|---|
| Dense (MiniLM) | 0.50 | 0.60 | 0.73 | 0.80 | 20ms |
| BM25 | 0.57 | 0.71 | 0.86 | 0.86 | 52ms |
| Hybrid (RRF) | 0.71 | 0.71 | 0.79 | 0.93 | 74ms |
| Hybrid + Rerank | 0.71 | 0.86 | 0.93 | 1.00 | 961ms |

## Key Findings

### 1. Complementary retrievers → RRF captures the ceiling

Dense and BM25 failed on different questions (5 both hit, 2 dense-only, 3 BM25-only, 4 both missed). RRF captured every complementary win — hitting exactly the theoretical ceiling of 0.71 at R@1.

**Lesson:** verify complementarity before assuming hybrid helps. If your retrievers fail on the same questions, hybrid ≈ average of both.

### 2. Reranking helps recall, NOT precision (at K=1)

Reranking pushed R@5 from 0.79 → 0.93 and R@10 from 0.93 → 1.00. But R@1 stayed at 0.71 — same as hybrid alone.

**Why:** cross-encoder rerankers score by query-chunk topical similarity, not by whether the chunk contains the answer. When the query's most distinctive keyword (e.g., "Aristotle") appears in a WRONG-but-topically-similar article, reranking often reinforces that mistake.

**Concrete example:** query "Where did Aristotle's most famous student wage military campaigns?" retrieves an Aristotle chunk discussing his relationship with Alexander. The reranker scores this HIGHER than the Alexander article, because it satisfies both "Aristotle" (query subject) and "campaigns" (query activity). Technically correct semantic overlap. Functionally wrong answer.

### 3. Latency cost is brutal for the R@1 case

13x slower (74ms → 961ms) for zero R@1 improvement. For downstream LLMs using top-5 context, reranking's R@5 gain (0.79 → 0.93) justifies the cost. For top-1 systems, gains are illusory.

## The Four Remaining Failures Reveal Structural Limits

Even with hybrid + rerank, R@1 caps at 0.71. The four remaining misses share a pattern: **the query's surface keywords match a wrong-but-adjacent article better than the right one.**

- Aristotle's student → keyword "Aristotle" > answer "Alexander"
- E=mc² → formula tokens > physicist name
- 4th century BCE conqueror → period keywords > specific person
- Union general baseball myth → Civil War context > individual name

These require capabilities beyond retrieval+rerank:
- **Query rewriting** (LLM rewrites "Aristotle's student" → "Alexander the Great")
- **Answer scoring** (rerank based on answer-containment, not topical relevance)
- **Multi-step agentic retrieval** (evaluate results, refine query, retry)

Day 13 (agentic RAG) will target these.

## Production Implications

- **Encyclopedic corpus + top-5 LLM context:** hybrid + rerank is worth 900ms
- **Encyclopedic corpus + top-1 (chat, cost-sensitive):** hybrid alone is nearly as good, 13x faster
- **Domain corpus (legal, medical, technical):** must re-run this comparison; keyword density differs

## Content

The story of Day 10 is not "reranking wins." It's:
1. Verify retriever complementarity empirically before building hybrid
2. Reranking's benefit is at K=3+, not K=1
3. Both retrieval and rerank optimize topical similarity; when questions require answer-containment reasoning, both fail together
4. That failure mode is why agentic RAG exists

---

# Day 11 — Eval Gate In CI

## What Shipped

A GitHub Actions workflow that automatically blocks PRs whose retrieval quality regresses below configured thresholds.

**Three CI jobs:**
- Lint & format (ruff)
- Unit tests (pytest — 5 module-level import checks)
- Eval threshold gate (reads committed metrics, exits non-zero on threshold violation)

**Threshold design:**
- Global floors on R@5 and R@10 (catches broad regressions)
- Per-approach floors (dense, hybrid, hybrid_rerank each held to their own bar)
- Per-category floors on R@1 (catches category-specific regressions hidden in aggregate)

**Gate mechanics:**
- Fast: reads committed `data/eval_runs/*_metrics.json` rather than re-running the full pipeline
- Contributor responsibility: when retrieval code changes, run evals locally and commit updated metrics — PR shows both code and metric diffs

## Verification (Chaos Test)

After building the gate, deliberately regressed `hybrid_rerank_metrics.json`:
- R@5: 0.93 → 0.60 (below approach floor 0.85 and global floor 0.70)
- R@10: 1.00 → 0.70 (below approach floor 0.95 and global floor 0.80)

Pushed as a PR. CI eval-gate job correctly failed with 4 threshold violations. PR unmergeable until metrics restored.

**This verification matters.** A CI gate that always passes is worse than no gate — false confidence in a check that doesn't check. Real production teams verify every new gate by proving it blocks. Half the CI gates in most codebases don't actually gate anything; nobody tested them.

## Deferred: LLM-Judged Metrics

Attempted RAGAS 0.4.3 for context_recall, faithfulness, and answer_relevancy. Library is currently broken — hard-imports a langchain_community module that no longer exists in current versions. Documented in TECHDEBT.md.

The recall@K metric is a strong proxy for retrieval quality. LLM-judged metrics matter more once we add answer generation to the pipeline (Day 13 agentic RAG).

## Content

Real production RAG lesson from Day 11: after building a CI gate, verify it actually blocks. Deliberately regress a metric, push, confirm CI turns red. If the chaos test also passes, your gate is theater. This discipline catches broken checks before they hide real regressions.
