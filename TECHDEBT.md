# Known Tech Debt

## RAGAS integration deferred (Day 11)

**Current state:** `src/rag_playground/ragas_eval.py` exists but is not wired
into CI. Any attempt to import RAGAS 0.4.3 in the current environment fails
with `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`.

**Why:** RAGAS 0.4.3 hard-imports a langchain_community path that was
removed in later langchain_community versions. This is a package-level bug,
not a configuration issue on our end.

**Impact:** No LLM-judged metrics (context_recall, faithfulness,
answer_relevancy) in CI today. The recall@K metric catches most retrieval
regressions but doesn't catch:
- Hallucinations in generated answers
- Off-topic answers
- Answers that fail to use the retrieved context

**Fix options (any of these):**
1. Wait for RAGAS to fix the langchain_community import (upstream issue)
2. Downgrade langchain_community and pin all langchain-family versions
3. Roll our own LLM-as-judge using Groq directly (~200 lines of code)

**Priority:** Medium. recall@K is a strong proxy for retrieval quality.
LLM-judged metrics matter more when we add answer generation to the
pipeline (Day 13 agentic RAG).

## Reranker regresses temporal queries (Day 11)

**Observation:** Cross-encoder reranker scores temporal queries lower than
dense retrieval alone. Dense R@1 for temporal = 1.00, reranker R@1 = 0.50.

**Why:** Cross-encoders trained on general relevance data (MS MARCO) score
topical overlap, not temporal/date reasoning. Query "Which 4th century BCE
conqueror shaped Hellenistic civilization?" gets Ancient Philosophy over
Alexander the Great because "ancient philosophy" is more topically dense.

**Fix:** Query rewriting or answer-scoring reranker. Both require agentic
retrieval, which is Day 13.

**Current mitigation:** eval_thresholds.yaml sets the temporal floor at
0.50 (matching current behavior) so CI doesn't false-alarm.
