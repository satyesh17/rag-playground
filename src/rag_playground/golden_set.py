"""
Golden question set for RAG retrieval benchmarking.

Structure:
Each golden entry: {
    "question": str,                     # The user query
    "expected_doc_titles": list[str],    # Article title(s) that should be retrieved
}

We match by document TITLE rather than exact chunk because:
- Any chunk from the correct article counts as a hit
- More lenient (and realistic) than exact-chunk match

Question categories intentionally varied:
- Definitional ("What is X?")
- Historical ("When did X happen?")
- Causal ("Why does X?")
- Descriptive ("How does X work?")
- Comparative ("What's the difference between X and Y?")

This tests whether the models generalize across question styles.
"""
from dataclasses import dataclass


@dataclass
class GoldenEntry:
    question: str
    expected_doc_titles: list[str]
    category: str = ""   # For error analysis later


GOLDEN_QUESTIONS = [
    # Original 5 that matched
    GoldenEntry(
        question="What is the anarchist political philosophy?",
        expected_doc_titles=["Anarchism"],
        category="definitional",
    ),
    GoldenEntry(
        question="What was Aristotle's contribution to philosophy?",
        expected_doc_titles=["Aristotle"],
        category="descriptive",
    ),
    GoldenEntry(
        question="What is the history of Athens?",
        expected_doc_titles=["Athens", "History of Athens"],
        category="historical",
    ),
    GoldenEntry(
        question="Who was Alexander the Great and what did he conquer?",
        expected_doc_titles=["Alexander the Great"],
        category="descriptive",
    ),
    GoldenEntry(
        question="What is Einstein's theory of relativity?",
        expected_doc_titles=["Albert Einstein", "Theory of relativity"],
        category="definitional",
    ),
    # New ones based on your actual corpus
    GoldenEntry(
        question="What was Jonathan Swift's A Modest Proposal about?",
        expected_doc_titles=["A Modest Proposal"],
        category="descriptive",
    ),
    GoldenEntry(
        question="Who invented baseball and what role did Abner Doubleday play?",
        expected_doc_titles=["Abner Doubleday"],
        category="historical",
    ),
    GoldenEntry(
        question="What causes the American Civil War?",
        expected_doc_titles=["American Civil War"],
        category="causal",
    ),
    GoldenEntry(
        question="What is anime and how does Japanese animation differ from Western?",
        expected_doc_titles=["Anime"],
        category="descriptive",
    ),
    GoldenEntry(
        question="How is the ampere defined as a unit of electric current?",
        expected_doc_titles=["Ampere"],
        category="definitional",
    ),
    GoldenEntry(
        question="What is altruism in behavioral biology?",
        expected_doc_titles=["Altruism"],
        category="definitional",
    ),
    GoldenEntry(
        question="Who was the chess player Anatoly Karpov?",
        expected_doc_titles=["Anatoly Karpov"],
        category="descriptive",
    ),
    GoldenEntry(
        question="What is an alloy and how are they made?",
        expected_doc_titles=["Alloy"],
        category="definitional",
    ),
    GoldenEntry(
        question="What is the geography of Alabama?",
        expected_doc_titles=["Geography of Alabama"],
        category="descriptive",
    ),
    GoldenEntry(
        question="What is International Atomic Time and how is it measured?",
        expected_doc_titles=["International Atomic Time"],
        category="definitional",
    ),
]


def load_golden_set() -> list[GoldenEntry]:
    """Return the current golden question set."""
    return GOLDEN_QUESTIONS


if __name__ == "__main__":
    entries = load_golden_set()
    print(f"Golden set: {len(entries)} questions")
    for i, e in enumerate(entries, 1):
        print(f"  {i:2d}. [{e.category:12s}] {e.question}")
