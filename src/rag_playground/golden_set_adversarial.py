"""
ADVERSARIAL golden question set (Day 10 v2).

Design goals: break dense-only retrieval so we can measure whether hybrid
and reranking actually help.

Categories designed to fail specific retrieval mechanisms:
- SEMANTIC_DRIFT: uses vocabulary that pulls toward wrong articles
- LEXICAL_ONLY: exact terms only, no semantic scaffolding
- MULTIHOP: answer requires connecting 2+ articles
- TEMPORAL: relies on dates/periods that vocabulary alone can't disambiguate
- NEGATION: uses negation/comparison that confuses dense embeddings
- INDIRECT_ROLE: references entity by role/relationship, not name

If any category doesn't actually degrade the baseline, we drop those questions
and replace with harder ones. Empirical selection.
"""

from dataclasses import dataclass


@dataclass
class GoldenEntry:
    question: str
    expected_doc_titles: list[str]
    category: str = ""
    notes: str = ""


ADVERSARIAL_QUESTIONS = [
    # SEMANTIC_DRIFT — vocabulary points to wrong-but-related articles
    GoldenEntry(
        question="Where did Aristotle's most famous student wage military campaigns after his teacher's death?",
        expected_doc_titles=["Alexander the Great"],
        category="semantic_drift",
        notes="Query mentions Aristotle explicitly — should pull his article, but answer is about student",
    ),
    # LEXICAL_ONLY — exact-term matches where semantics don't help
    GoldenEntry(
        question="What does the acronym TAI stand for in metrology?",
        expected_doc_titles=["International Atomic Time"],
        category="lexical_only",
        notes="Only 'TAI' matches; dense retrieval will fail on this",
    ),
    GoldenEntry(
        question="Who wrote the Nicomachean Ethics?",
        expected_doc_titles=["Aristotle"],
        category="lexical_only",
        notes="Only 'Nicomachean' is a distinctive keyword; dense should still find Aristotle but might drift",
    ),
    GoldenEntry(
        question="What does E=mc squared mean in physics?",
        expected_doc_titles=["Albert Einstein"],
        category="lexical_only",
        notes="Formula string might fail on dense; BM25 should nail exact match",
    ),
    # MULTIHOP — requires linking articles
    GoldenEntry(
        question="Which philosopher's work influenced the school that Aristotle later founded in Athens?",
        expected_doc_titles=["Plato", "Aristotle"],
        category="multihop",
        notes="Plato -> influenced Aristotle -> founded Lyceum in Athens; connection across 2 articles",
    ),
    GoldenEntry(
        question="What form of government did the birthplace of Western philosophy pioneer?",
        expected_doc_titles=["Athens"],
        category="multihop",
        notes="Birthplace of philosophy → Athens → democracy; requires two hops",
    ),
    # TEMPORAL — date/period disambiguation
    GoldenEntry(
        question="Which conflict of the 1860s reshaped American federalism?",
        expected_doc_titles=["American Civil War"],
        category="temporal",
        notes="No 'civil war' keyword; period + geography",
    ),
    GoldenEntry(
        question="Which 4th century BCE conqueror shaped Hellenistic civilization?",
        expected_doc_titles=["Alexander the Great"],
        category="temporal",
        notes="4th century BCE + conqueror + Hellenistic; multiple attractors possible",
    ),
    # NEGATION — comparison and exclusion
    GoldenEntry(
        question="Unlike pure metals, what property makes mixtures of metallic elements industrially useful?",
        expected_doc_titles=["Alloy"],
        category="negation",
        notes="'Unlike pure metals' vs 'mixtures of metallic elements' — negation confuses dense",
    ),
    GoldenEntry(
        question="Which political philosophy is opposed to both capitalism and state socialism?",
        expected_doc_titles=["Anarchism"],
        category="negation",
        notes="Defined by what it's NOT — dense retrieval often trips on this",
    ),
    # INDIRECT_ROLE — entity by role, not identity
    GoldenEntry(
        question="Which chess player defeated Boris Spassky's challenger in the 1970s title match?",
        expected_doc_titles=["Anatoly Karpov"],
        category="indirect_role",
        notes="Karpov defeated Korchnoi (Spassky's challenger); no keywords match Karpov directly",
    ),
    GoldenEntry(
        question="Which Union general has been mythologized as the inventor of America's national pastime?",
        expected_doc_titles=["Abner Doubleday"],
        category="indirect_role",
        notes="Doubleday by role; 'national pastime' = baseball, not stated",
    ),
    # TRICKY MULTIHOP
    GoldenEntry(
        question="What is the SI unit named after the French mathematician who studied electromagnetism?",
        expected_doc_titles=["Ampere"],
        category="multihop",
        notes="Ampere named after André-Marie Ampère; requires connecting person -> unit",
    ),
    GoldenEntry(
        question="Which stylized Japanese animation medium grew from post-war manga culture?",
        expected_doc_titles=["Anime"],
        category="semantic_drift",
        notes="'Manga' will pull toward Manga article; anime is the target",
    ),
]


def load_adversarial_golden_set() -> list[GoldenEntry]:
    return ADVERSARIAL_QUESTIONS


if __name__ == "__main__":
    entries = load_adversarial_golden_set()
    from collections import Counter

    print(f"Adversarial golden set: {len(entries)} questions")
    cats = Counter(e.category for e in entries)
    print("\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat:16s}: {count}")
    print("\nSamples:")
    for i, e in enumerate(entries[:5], 1):
        print(f"  {i}. [{e.category:16s}] {e.question}")
