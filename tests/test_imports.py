"""Sanity: verify all modules import without errors."""


def test_chunker_imports():
    from src.rag_playground.chunker import (
        CHUNKING_STRATEGIES,
    )

    assert len(CHUNKING_STRATEGIES) == 4


def test_golden_set_imports():
    from src.rag_playground.golden_set_adversarial import load_adversarial_golden_set

    entries = load_adversarial_golden_set()
    assert len(entries) == 14


def test_reference_answers_align():
    from src.rag_playground.golden_set_adversarial import load_adversarial_golden_set
    from src.rag_playground.reference_answers import REFERENCE_ANSWERS

    golden = load_adversarial_golden_set()
    for entry in golden:
        assert entry.question in REFERENCE_ANSWERS, f"Missing reference for: {entry.question}"


def test_eval_runner_imports():
    # Just check the module imports — actually running requires Qdrant
    from src.rag_playground import eval_runner

    assert eval_runner.APPROACHES.keys() == {"dense", "hybrid", "hybrid_rerank"}


def test_gate_script_syntax():
    import subprocess

    result = subprocess.run(
        ["python", "-m", "py_compile", "scripts/check_eval_thresholds.py"],
        capture_output=True,
    )
    assert result.returncode == 0, f"Gate script has syntax errors: {result.stderr.decode()}"
