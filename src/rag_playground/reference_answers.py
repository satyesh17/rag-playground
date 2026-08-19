"""
Ground-truth reference answers for the adversarial golden set.

Currently unused (RAGAS integration deferred due to library issues).
Kept for future use — either RAGAS-based scoring when the library stabilizes,
or a hand-rolled LLM-as-judge implementation.

Reference answers are the highest-leverage artifact in RAG evaluation:
they define what "correct" means for each question.
"""

REFERENCE_ANSWERS: dict[str, str] = {
    "Where did Aristotle's most famous student wage military campaigns after his teacher's death?": "Alexander the Great, Aristotle's most famous student, waged military "
    "campaigns across Persia, Egypt, Central Asia, and into India, "
    "building one of history's largest empires.",
    "What does the acronym TAI stand for in metrology?": "TAI stands for International Atomic Time (Temps Atomique International "
    "in French), a high-precision time standard based on atomic clocks.",
    "Who wrote the Nicomachean Ethics?": "Aristotle wrote the Nicomachean Ethics, one of his most influential "
    "works on moral philosophy and virtue ethics.",
    "What does E=mc squared mean in physics?": "E=mc² is Albert Einstein's mass-energy equivalence formula from special "
    "relativity, showing that mass and energy are interchangeable and that "
    "a small amount of mass can be converted to a very large amount of energy.",
    "Which philosopher's work influenced the school that Aristotle later founded in Athens?": "Plato's work influenced Aristotle, who was Plato's student at the "
    "Academy in Athens before founding his own school, the Lyceum.",
    "What form of government did the birthplace of Western philosophy pioneer?": "Athens, considered a birthplace of Western philosophy, pioneered "
    "democratic government in the classical period.",
    "Which conflict of the 1860s reshaped American federalism?": "The American Civil War (1861-1865) reshaped American federalism by "
    "establishing federal supremacy over states' rights and abolishing slavery.",
    "Which 4th century BCE conqueror shaped Hellenistic civilization?": "Alexander the Great, king of Macedon in the 4th century BCE, shaped "
    "Hellenistic civilization by spreading Greek culture across his vast "
    "empire from Egypt to India.",
    "Unlike pure metals, what property makes mixtures of metallic elements industrially useful?": "Alloys — mixtures of metals with other elements — are more industrially "
    "useful than pure metals because they offer improved strength, hardness, "
    "corrosion resistance, and other tailored properties.",
    "Which political philosophy is opposed to both capitalism and state socialism?": "Anarchism opposes both capitalism and state socialism, advocating for "
    "stateless societies organized without hierarchical authority.",
    "Which chess player defeated Boris Spassky's challenger in the 1970s title match?": "Anatoly Karpov defeated Viktor Korchnoi (who had defeated Boris Spassky "
    "in the Candidates matches) in the 1978 World Chess Championship.",
    "Which Union general has been mythologized as the inventor of America's national pastime?": "Abner Doubleday, a Union general during the American Civil War, has "
    "been mythologized as the inventor of baseball, though modern historians "
    "have discredited this attribution.",
    "What is the SI unit named after the French mathematician who studied electromagnetism?": "The ampere, the SI unit of electric current, is named after André-Marie "
    "Ampère, a French mathematician and physicist who pioneered the study of "
    "electromagnetism.",
    "Which stylized Japanese animation medium grew from post-war manga culture?": "Anime, a distinctive style of Japanese animation, grew from post-war "
    "manga culture and has become a major global entertainment medium.",
}


def get_reference(question: str) -> str:
    """Return the reference answer for a given question."""
    if question not in REFERENCE_ANSWERS:
        raise KeyError(f"No reference answer for: {question}")
    return REFERENCE_ANSWERS[question]
