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


# ============================================================================
# NEW CHUNKERS (Day 9)
# ============================================================================

def recursive_chunks(
    doc_id: str,
    doc_title: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Recursive character splitting — tries progressively finer boundaries.

    Strategy: split on '\\n\\n' (paragraphs) first. If a chunk is still too big,
    split on '\\n' (lines). Then '. ' (sentences). Then ' ' (words).

    This preserves semantic units better than fixed-size splitting because
    it avoids cutting mid-sentence when a paragraph boundary would do.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_recursive(text: str, seps: list[str]) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if len(text) <= chunk_size:
            return [text]

        # Try each separator in order
        for sep in seps:
            if sep and sep in text:
                parts = text.split(sep)
                # Rejoin parts up to chunk_size
                result = []
                current = ""
                for part in parts:
                    trial = current + (sep if current else "") + part
                    if len(trial) <= chunk_size:
                        current = trial
                    else:
                        if current:
                            result.append(current)
                        # If this single part is bigger than chunk_size,
                        # recurse with finer separators
                        if len(part) > chunk_size:
                            result.extend(_split_recursive(part, seps[seps.index(sep) + 1:]))
                            current = ""
                        else:
                            current = part
                if current:
                    result.append(current)
                return result

        # No separator worked — hard-split by chunk_size
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    raw_chunks = _split_recursive(text, separators)

    # Apply overlap: append last `overlap` chars of previous chunk to next
    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        if idx > 0 and overlap > 0:
            prev_tail = raw_chunks[idx - 1][-overlap:]
            chunk_text = prev_tail + chunk_text

        chunks.append(Chunk(
            chunk_id=f"{doc_id}::rec_{idx}",
            doc_id=doc_id,
            doc_title=doc_title,
            text=chunk_text,
            chunk_index=idx,
        ))

    return chunks


def sentence_chunks(
    doc_id: str,
    doc_title: str,
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Sentence-based chunking — groups N sentences per chunk.

    Simple but effective. Preserves complete sentences at the cost of
    variable chunk sizes.

    Not "semantic" chunking in the strict sense (which uses embedding
    similarity to decide where to split), but a solid baseline that's
    much better than mid-word splits.
    """
    if not text:
        return []

    # Naive sentence splitting — good enough for prose.
    # Production would use nltk.sent_tokenize or spaCy.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    idx = 0
    step = max_sentences - overlap_sentences

    for start in range(0, len(sentences), step):
        end = min(start + max_sentences, len(sentences))
        chunk_text = " ".join(sentences[start:end])

        chunks.append(Chunk(
            chunk_id=f"{doc_id}::sent_{idx}",
            doc_id=doc_id,
            doc_title=doc_title,
            text=chunk_text,
            chunk_index=idx,
        ))
        idx += 1

        if end >= len(sentences):
            break

    return chunks


def paragraph_chunks(
    doc_id: str,
    doc_title: str,
    text: str,
    max_paragraphs: int = 2,
) -> list[Chunk]:
    """
    Paragraph-based chunking — groups N paragraphs per chunk.

    Wikipedia articles are naturally structured into paragraphs.
    This uses the author's own semantic units — often the most sensible
    boundary for well-structured prose.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for idx, i in enumerate(range(0, len(paragraphs), max_paragraphs)):
        chunk_text = "\n\n".join(paragraphs[i:i + max_paragraphs])

        chunks.append(Chunk(
            chunk_id=f"{doc_id}::para_{idx}",
            doc_id=doc_id,
            doc_title=doc_title,
            text=chunk_text,
            chunk_index=idx,
        ))

    return chunks


# ============================================================================
# STRATEGY DISPATCH
# ============================================================================

CHUNKING_STRATEGIES = {
    "fixed": {
        "func": fixed_size_chunks,
        "kwargs": {"chunk_size": 500, "overlap": 50},
        "description": "Fixed-size 500 char with 50 char overlap",
    },
    "recursive": {
        "func": recursive_chunks,
        "kwargs": {"chunk_size": 500, "overlap": 50},
        "description": "Recursive character splitting on paragraph/sentence/word",
    },
    "sentence": {
        "func": sentence_chunks,
        "kwargs": {"max_sentences": 5, "overlap_sentences": 1},
        "description": "5 sentences per chunk, 1 sentence overlap",
    },
    "paragraph": {
        "func": paragraph_chunks,
        "kwargs": {"max_paragraphs": 2},
        "description": "2 paragraphs per chunk, no overlap",
    },
}


def chunk_corpus_with_strategy(
    records: list[dict],
    strategy: str,
    apply_clamping: bool = True,
) -> list[Chunk]:
    """Chunk a corpus using the named strategy, with optional size clamping."""
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Options: {list(CHUNKING_STRATEGIES)}")

    cfg = CHUNKING_STRATEGIES[strategy]
    func = cfg["func"]
    kwargs = cfg["kwargs"]

    all_chunks = []
    for r in records:
        all_chunks.extend(func(
            doc_id=r["id"],
            doc_title=r["title"],
            text=r["text"],
            **kwargs,
        ))

    if apply_clamping:
        all_chunks = clamp_chunks(all_chunks, min_chars=100, max_chars=1500)

    return all_chunks


# ============================================================================
# SIZE CLAMPING (Day 9 fix — real-world chunk hygiene)
# ============================================================================

def clamp_chunks(
    chunks: list[Chunk],
    min_chars: int = 100,
    max_chars: int = 1500,
) -> list[Chunk]:
    """
    Enforce chunk size bounds via drop-and-split (simpler than merge logic):
    - Chunks smaller than min_chars are DROPPED (rarely carry meaning anyway)
    - Chunks larger than max_chars are HARD-SPLIT into <= max_chars pieces

    Design trade-off: dropping tiny chunks means we lose a few (maybe 1-2%)
    but ALL retained chunks are guaranteed within bounds. That's a better
    trade than complex merge logic with edge cases.
    """
    if not chunks:
        return []

    result = []

    for chunk in chunks:
        text = chunk.text.strip()

        # Drop chunks that are too small (after stripping)
        if len(text) < min_chars:
            continue

        # Chunks within bounds pass through unchanged
        if len(text) <= max_chars:
            result.append(Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                text=text,
                chunk_index=chunk.chunk_index,
            ))
            continue

        # Oversized chunks: hard-split at max_chars boundaries
        piece_idx = 0
        pos = 0
        while pos < len(text):
            piece_text = text[pos:pos + max_chars]

            # Only keep pieces that meet min_chars (last piece can be small)
            if len(piece_text) >= min_chars:
                result.append(Chunk(
                    chunk_id=f"{chunk.chunk_id}_p{piece_idx}",
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    text=piece_text,
                    chunk_index=chunk.chunk_index,
                ))
                piece_idx += 1

            pos += max_chars

    return result
