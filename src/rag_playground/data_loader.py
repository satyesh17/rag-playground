"""
Download and prepare a document corpus for RAG benchmarking.

Uses HuggingFace's Wikipedia snapshot. Why Wikipedia:
- Real, varied prose (not synthetic)
- Diverse topics (tests generalization)
- Free and reliable

We take a small subset (~500 articles) for fast experimentation.
Production RAG would index millions.
"""
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def load_wikipedia_subset(
    n_articles: int = 500,
    output_path: Path = Path("data/corpus.jsonl"),
) -> list[dict]:
    """
    Download N Wikipedia articles and save as JSONL.

    Each record: {"id": str, "title": str, "text": str}

    Returns the list of records for immediate use.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n_articles} Wikipedia articles from HuggingFace...")
    # Streaming mode — doesn't download the whole 20GB corpus,
    # just streams what we need.
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    records = []
    for i, article in enumerate(tqdm(ds, total=n_articles, desc="Downloading")):
        if i >= n_articles:
            break

        # Filter: skip disambiguation pages and stubs (too short to be useful)
        text = article["text"]
        if len(text) < 500 or "may refer to:" in text[:200].lower():
            continue

        records.append({
            "id": str(article["id"]),
            "title": article["title"],
            "text": text,
        })

    # Save as JSONL for easy inspection
    import json
    with output_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} articles to {output_path}")
    return records


if __name__ == "__main__":
    records = load_wikipedia_subset(n_articles=500)
    print(f"\nFirst article: {records[0]['title']}")
    print(f"Length: {len(records[0]['text'])} chars")
    print(f"Preview: {records[0]['text'][:200]}...")
