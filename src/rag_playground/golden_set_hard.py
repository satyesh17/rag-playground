"""
HARD golden question set for RAG benchmarking (Day 10).

Design goals: differentiate retrieval strategies by removing surface
keyword overlap between queries and target article titles.

Question types:
- Paraphrase: uses different vocabulary than the article
- Indirect: refers to the entity by role, description, or era
- Multi-hop: answer requires connecting information across the article
- Analogical: describes what the concept IS without naming it
"""
from dataclasses import dataclass


@dataclass
class GoldenEntry:
    question: str
    expected_doc_titles: list[str]
    category: str = ""
    notes: str = ""


HARD_QUESTIONS = [
    # Paraphrase — no article-title vocabulary
    GoldenEntry(
        question="Which political philosophy rejects hierarchical authority and advocates for stateless societies?",
        expected_doc_titles=["Anarchism"],
        category="paraphrase",
        notes="Article title 'Anarchism' does not appear in query",
    ),
    GoldenEntry(
        question="Who founded the school of thought based on syllogistic logic and virtue ethics?",
        expected_doc_titles=["Aristotle"],
        category="indirect",
        notes="Refers to Aristotle by his intellectual contributions",
    ),
    GoldenEntry(
        question="Which ancient Greek city-state is considered the birthplace of democratic government?",
        expected_doc_titles=["Athens"],
        category="paraphrase",
        notes="Athens described by function, not named",
    ),
    GoldenEntry(
        question="Which Macedonian ruler built one of the largest empires in ancient history before dying at 32?",
        expected_doc_titles=["Alexander the Great"],
        category="indirect",
        notes="Refers to Alexander by nationality and biographical detail",
    ),
    GoldenEntry(
        question="Which physicist explained gravity as the curvature of spacetime?",
        expected_doc_titles=["Albert Einstein"],
        category="paraphrase",
        notes="Refers to Einstein by his theory's mechanism, not name",
    ),
    GoldenEntry(
        question="What 18th century essay ironically suggested eating children as a solution to Irish poverty?",
        expected_doc_titles=["A Modest Proposal"],
        category="indirect",
        notes="Describes the work's content and satirical tone",
    ),
    GoldenEntry(
        question="Which American conflict pitted the industrial north against the agrarian south?",
        expected_doc_titles=["American Civil War"],
        category="paraphrase",
        notes="Describes the war by economic geography",
    ),
    GoldenEntry(
        question="What Japanese art form uses stylized illustration for storytelling and TV series?",
        expected_doc_titles=["Anime"],
        category="paraphrase",
        notes="Anime described by function, not named",
    ),
    GoldenEntry(
        question="What SI unit measures the flow rate of electric charge?",
        expected_doc_titles=["Ampere"],
        category="paraphrase",
        notes="Ampere described by what it measures",
    ),
    GoldenEntry(
        question="What behavior evolves when helping others reduces one's own reproductive fitness?",
        expected_doc_titles=["Altruism"],
        category="paraphrase",
        notes="Altruism described by its evolutionary paradox",
    ),
    GoldenEntry(
        question="Which grandmaster held the world chess title through most of the late Cold War?",
        expected_doc_titles=["Anatoly Karpov"],
        category="indirect",
        notes="Karpov by chess role and era, not name",
    ),
    GoldenEntry(
        question="What material combination is used to make metals stronger, harder, or more corrosion-resistant?",
        expected_doc_titles=["Alloy"],
        category="paraphrase",
        notes="Alloys described by their purpose",
    ),
    GoldenEntry(
        question="Which US state has the Appalachian Mountains in its northeast and Gulf coast in its south?",
        expected_doc_titles=["Geography of Alabama"],
        category="multi-hop",
        notes="Alabama identified by two geographic features",
    ),
    GoldenEntry(
        question="What global time standard uses atomic clocks and forms the basis of UTC?",
        expected_doc_titles=["International Atomic Time"],
        category="paraphrase",
        notes="TAI described by function and relationship to UTC",
    ),
    GoldenEntry(
        question="Who was the American general credited with — but not actually responsible for — inventing baseball?",
        expected_doc_titles=["Abner Doubleday"],
        category="indirect",
        notes="Doubleday by nationality, profession, and the baseball myth",
    ),
]


def load_hard_golden_set() -> list[GoldenEntry]:
    return HARD_QUESTIONS


if __name__ == "__main__":
    entries = load_hard_golden_set()
    print(f"Hard golden set: {len(entries)} questions")
    for i, e in enumerate(entries, 1):
        print(f"  {i:2d}. [{e.category:11s}] {e.question[:75]}")
