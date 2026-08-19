"""
RAGAS-based LLM-judged evaluation.

Adds three metrics beyond your existing recall@K:
- context_recall: does the retrieved context contain enough info to answer?
- faithfulness: is the generated answer grounded in the context (no hallucination)?
- answer_relevancy: does the answer address the question?

These require LLM inference — we use Groq for cheap/fast judging.

Design notes:
- Uses only top-5 retrieved chunks as context (matches typical LLM RAG usage)
- Generates an answer per question, then scores the answer
- Runs sequentially (RAGAS supports async but sequential is easier to debug)
- Costs ~$0.02 per full run (14 questions × ~3 LLM calls each)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def _get_groq_llm():
    """Return a Groq LLM client for LLM-as-judge scoring."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not set — check .env")
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Fast, free-tier, good for judging
        temperature=0.0,  # Deterministic for eval consistency
    )


def generate_answer(llm, question: str, context_chunks: list[str]) -> str:
    """Generate an answer using the retrieved context. Standard RAG prompt."""
    context = "\n\n---\n\n".join(context_chunks[:5])  # Top-5 chunks
    prompt = f"""Answer the question based only on the context provided.
If the context does not contain the answer, say "I don't have enough information to answer."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def evaluate_approach_with_ragas(
    approach_name: str,
    raw_results_path: Path,
) -> dict:
    """
    Load raw retrieval results for one approach, generate answers,
    score with RAGAS metrics, return summary.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_recall,
        faithfulness,
    )

    from src.rag_playground.reference_answers import get_reference

    # Load raw retrieval results
    with raw_results_path.open() as f:
        raw = json.load(f)

    llm = _get_groq_llm()

    # For each question: generate an answer using retrieved chunks
    print(f"\nGenerating answers for {approach_name} ({len(raw)} questions)...")
    ragas_records = []
    for i, r in enumerate(raw, 1):
        print(f"  [{i}/{len(raw)}] {r['question'][:60]}...")
        answer = generate_answer(llm, r["question"], r["retrieved_texts"])
        ragas_records.append(
            {
                "question": r["question"],
                "answer": answer,
                "contexts": r["retrieved_texts"][:5],  # Top-5 chunks as context
                "ground_truth": get_reference(r["question"]),
                "reference": get_reference(r["question"]),  # RAGAS >= 0.2 uses this
            }
        )

    dataset = Dataset.from_list(ragas_records)

    print("\nScoring with RAGAS metrics (this takes 2-3 minutes)...")
    result = evaluate(
        dataset=dataset,
        metrics=[context_recall, faithfulness, answer_relevancy],
        llm=llm,
    )

    # Aggregate metrics — RAGAS returns per-question scores; we average
    metrics_dict = {
        "approach": approach_name,
        "n_questions": len(raw),
        "context_recall": float(result["context_recall"]),
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }
    return metrics_dict


def run_ragas_on_all_approaches(
    eval_dir: Path = Path("data/eval_runs"),
) -> dict[str, dict]:
    """Run RAGAS on all approaches that have raw results."""
    all_ragas = {}
    for raw_file in sorted(eval_dir.glob("*_raw.json")):
        approach = raw_file.stem.replace("_raw", "")
        print(f"\n{'=' * 60}\nRAGAS eval for: {approach}\n{'=' * 60}")

        try:
            metrics = evaluate_approach_with_ragas(approach, raw_file)
            all_ragas[approach] = metrics

            # Save alongside existing metrics
            out_path = eval_dir / f"{approach}_ragas.json"
            with out_path.open("w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_ragas[approach] = {"error": str(e)}

    return all_ragas


if __name__ == "__main__":
    results = run_ragas_on_all_approaches()

    print(f"\n{'=' * 78}\nRAGAS SUMMARY (LLM-judged metrics)\n{'=' * 78}")
    print(f"{'Approach':<16} {'context_recall':<16} {'faithfulness':<14} {'answer_rel.':<12}")
    print("-" * 78)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<16} ERROR: {m['error']}")
        else:
            print(
                f"{name:<16} {m['context_recall']:<16.3f} {m['faithfulness']:<14.3f} {m['answer_relevancy']:<12.3f}"
            )
