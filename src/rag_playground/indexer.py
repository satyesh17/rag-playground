"""
Embed chunks and load them into Qdrant.

Supports two use cases:
- Day 8: benchmark embedding models (one strategy, multiple models)
- Day 9: benchmark chunking strategies (one model, multiple strategies)

Collections are named to reflect what's inside:
- "wiki_minilm" — Day 8 collection, MiniLM model, fixed chunking
- "wiki_bge_large" — Day 8 collection, BGE-large model, fixed chunking
- "day9_fixed" / "day9_recursive" / "day9_sentence" / "day9_paragraph"
  — Day 9 collections, MiniLM model, varying chunking
"""
import uuid
from dataclasses import asdict

from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.rag_playground.chunker import Chunk
from src.rag_playground.db import get_client


# Day 8 config — kept for backward compat
EMBEDDING_MODELS = {
    "minilm": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "collection": "wiki_minilm",
    },
    "bge_large": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "collection": "wiki_bge_large",
    },
}


def load_model(model_key: str) -> SentenceTransformer:
    """Load a sentence-transformers model by our short key."""
    cfg = EMBEDDING_MODELS[model_key]
    print(f"Loading {cfg['hf_id']} (dim={cfg['dim']})...")
    return SentenceTransformer(cfg["hf_id"])


def create_named_collection(collection_name: str, dim: int) -> None:
    """Create (or recreate) a Qdrant collection with the given name and dim."""
    client = get_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created collection: {collection_name} (dim={dim})")


def index_chunks_into(
    collection_name: str,
    chunks: list[Chunk],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> None:
    """Embed and upsert chunks into a specific Qdrant collection."""
    client = get_client()

    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Indexing {collection_name}"):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []
        for chunk, vector in zip(batch, vectors, strict=True):
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.tolist(),
                payload=asdict(chunk),
            ))

        client.upsert(collection_name=collection_name, points=points)

    count = client.count(collection_name, exact=True).count
    print(f"  → {count} vectors in {collection_name}")


# ============================================================================
# DAY 8 MODE — index full corpus with each model, fixed chunking
# ============================================================================

def day8_index_all_models() -> None:
    """Day 8 flow: full corpus, fixed chunking, two models."""
    import json
    from src.rag_playground.chunker import chunk_corpus_with_strategy

    with open("data/corpus.jsonl") as f:
        records = [json.loads(line) for line in f]

    chunks = chunk_corpus_with_strategy(records, "fixed")
    print(f"Total chunks: {len(chunks)}")

    for model_key in ["minilm", "bge_large"]:
        cfg = EMBEDDING_MODELS[model_key]
        create_named_collection(cfg["collection"], cfg["dim"])
        model = load_model(model_key)
        index_chunks_into(cfg["collection"], chunks, model)


# ============================================================================
# DAY 9 MODE — sample corpus, MiniLM, four chunking strategies
# ============================================================================

def day9_index_all_strategies(n_articles: int = 10000) -> None:
    """Day 9 flow: sample of corpus, MiniLM, all 4 chunking strategies."""
    import json
    import random
    from src.rag_playground.chunker import chunk_corpus_with_strategy, CHUNKING_STRATEGIES

    with open("data/corpus.jsonl") as f:
        records = [json.loads(line) for line in f]

    # Use a deterministic sample so results are reproducible
    random.seed(42)
    sampled = random.sample(records, min(n_articles, len(records)))
    print(f"Sampled {len(sampled)} articles from {len(records)} total\n")

    # Load MiniLM once — we'll use it for every strategy
    minilm = load_model("minilm")
    minilm_dim = EMBEDDING_MODELS["minilm"]["dim"]

    for strategy in CHUNKING_STRATEGIES:
        chunks = chunk_corpus_with_strategy(sampled, strategy)
        collection = f"day9_{strategy}"
        print(f"\n=== Strategy: {strategy} — {len(chunks)} chunks ===")
        create_named_collection(collection, minilm_dim)
        index_chunks_into(collection, chunks, minilm)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "day9"
    if mode == "day8":
        day8_index_all_models()
    else:
        day9_index_all_strategies()
