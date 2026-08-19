"""
Eval gate — reads metrics files, compares to thresholds, exits non-zero on
any failure.

Called by CI after running the eval suite. Exit code:
  0 = all thresholds met, PR is mergeable
  1 = one or more thresholds violated, PR blocked
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(2)


def load_thresholds(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_metrics(eval_dir: Path) -> dict[str, dict]:
    """Load all *_metrics.json files from eval_runs/."""
    metrics = {}
    for path in sorted(eval_dir.glob("*_metrics.json")):
        approach = path.stem.replace("_metrics", "")
        with path.open() as f:
            metrics[approach] = json.load(f)
    return metrics


def check_thresholds(metrics: dict[str, dict], thresholds: dict) -> list[str]:
    """Return a list of violation messages. Empty list = all passed."""
    violations = []

    global_thresh = thresholds.get("global", {})
    approach_thresh = thresholds.get("approaches", {})
    category_thresh = thresholds.get("per_category_r1_floor", {})

    for approach, m in metrics.items():
        # Recall@K in metrics is keyed by str, not int
        r_at_k = m.get("recall_at_k", {})
        r5 = float(r_at_k.get("5", 0))
        r10 = float(r_at_k.get("10", 0))

        # Global floors
        if r5 < global_thresh.get("min_recall_at_5", 0):
            violations.append(
                f"[{approach}] R@5 = {r5:.3f} < global floor {global_thresh['min_recall_at_5']}"
            )
        if r10 < global_thresh.get("min_recall_at_10", 0):
            violations.append(
                f"[{approach}] R@10 = {r10:.3f} < global floor {global_thresh['min_recall_at_10']}"
            )

        # Per-approach floors
        specific = approach_thresh.get(approach, {})
        if "min_recall_at_5" in specific and r5 < specific["min_recall_at_5"]:
            violations.append(
                f"[{approach}] R@5 = {r5:.3f} < approach floor {specific['min_recall_at_5']}"
            )
        if "min_recall_at_10" in specific and r10 < specific["min_recall_at_10"]:
            violations.append(
                f"[{approach}] R@10 = {r10:.3f} < approach floor {specific['min_recall_at_10']}"
            )

        # Per-category R@1 floors
        cat_recall = m.get("per_category_recall_at_1", {})
        for category, floor in category_thresh.items():
            if category in cat_recall:
                val = float(cat_recall[category])
                if val < floor:
                    violations.append(
                        f"[{approach}] category {category} R@1 = {val:.3f} < floor {floor}"
                    )

    return violations


def main():
    thresholds_path = Path("eval_thresholds.yaml")
    eval_dir = Path("data/eval_runs")

    if not thresholds_path.exists():
        print(f"ERROR: {thresholds_path} not found")
        sys.exit(2)
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} not found — did you run eval_runner?")
        sys.exit(2)

    thresholds = load_thresholds(thresholds_path)
    metrics = load_metrics(eval_dir)

    if not metrics:
        print(f"ERROR: no *_metrics.json files in {eval_dir}")
        sys.exit(2)

    print(f"Checking {len(metrics)} approaches against thresholds...\n")

    # Print current state
    for approach, m in metrics.items():
        r = m.get("recall_at_k", {})
        print(
            f"  {approach:16s}  R@1={r.get('1', 0):.2f}  R@5={r.get('5', 0):.2f}  R@10={r.get('10', 0):.2f}"
        )

    violations = check_thresholds(metrics, thresholds)

    print()
    if violations:
        print(f"❌ EVAL GATE FAILED — {len(violations)} threshold violation(s):\n")
        for v in violations:
            print(f"  - {v}")
        print()
        print("If this reflects an intentional quality improvement, update thresholds.")
        print("If not, this PR is blocked until retrieval quality is restored.")
        sys.exit(1)
    else:
        print(f"✅ EVAL GATE PASSED — all {len(metrics)} approaches meet thresholds")
        sys.exit(0)


if __name__ == "__main__":
    main()
