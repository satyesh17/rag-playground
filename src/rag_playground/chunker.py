"""
Text chunking strategies for RAG.

Today: fixed-size chunking (baseline).
Day 9 will add: recursive, semantic, document-aware.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata linking back to its source."""
    chunk_id: str          # Unique ID for this chunk
    doc_id: str            # ID of the source document
    doc_title: str         # Human-readable source name
    text: str              # The chunk text itself
    chunk_index: int       # Position within the source document (0, 1, 2, ...)


def fixed_size_chunks(
    doc_id: str,
    doc_title: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Split text into fixed-size character chunks with overlap.

    Why overlap:
    Splitting at exact boundaries risks cutting a sentence mid-thought.
    Overlap ensures every semantic unit appears fully in at least one chunk.

    Example: chunk_size=500, overlap=50 → chunks are 500 chars each,
    with the last 50 chars of chunk N reappearing as the first 50 chars of chunk N+1.
    """
    if not text:
        return []

    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(Chunk(
            chunk_id=f"{doc_id}::chunk_{idx}",
            doc_id=doc_id,
            doc_title=doc_title,
            text=chunk_text,
            chunk_index=idx,
        ))

        start += step
        idx += 1

        # Stop if we've captured the tail
        if end >= len(text):
            break

    return chunks


def chunk_corpus(
    records: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Apply fixed_size_chunks to every document in a corpus."""
    all_chunks = []
    for r in records:
        all_chunks.extend(fixed_size_chunks(
            doc_id=r["id"],
            doc_title=r["title"],
            text=r["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        ))
    return all_chunks


if __name__ == "__main__":
    # Quick smoke test
    from src.rag_playground.data_loader import load_wikipedia_subset
    import json
    from pathlib import Path

    # Reuse existing corpus if it exists
    if Path("data/corpus.jsonl").exists():
        records = []
        with open("data/corpus.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        print(f"Loaded {len(records)} existing records")
    else:
        records = load_wikipedia_subset(n_articles=500)

    chunks = chunk_corpus(records[:5])  # Chunk first 5 docs
    print(f"\nProduced {len(chunks)} chunks from 5 docs")
    print(f"Sample chunk: {chunks[0].chunk_id}")
    print(f"  Source: {chunks[0].doc_title}")
    print(f"  Length: {len(chunks[0].text)} chars")
    print(f"  Preview: {chunks[0].text[:150]}...")
