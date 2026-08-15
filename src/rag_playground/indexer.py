"""
Embed chunks and load them into Qdrant.

Each embedding model gets its own collection because vectors from
different models are incompatible (different dimensions AND different spaces).

Collections are named: "wiki_<model_short_name>"
"""
import uuid
from dataclasses import asdict

from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.rag_playground.chunker import Chunk
from src.rag_playground.db import get_client


# Configuration for each embedding model we'll benchmark
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
    # First call downloads the model; subsequent calls use HF cache
    return SentenceTransformer(cfg["hf_id"])


def create_collection(model_key: str) -> None:
    """Create (or recreate) the Qdrant collection for this model."""
    cfg = EMBEDDING_MODELS[model_key]
    client = get_client()

    # If the collection already exists, delete and recreate.
    # Fresh state = no confusion during benchmarking.
    if client.collection_exists(cfg["collection"]):
        client.delete_collection(cfg["collection"])
        print(f"Deleted existing collection: {cfg['collection']}")

    client.create_collection(
        collection_name=cfg["collection"],
        vectors_config=VectorParams(
            size=cfg["dim"],
            distance=Distance.COSINE,   # Cosine similarity, as discussed
        ),
    )
    print(f"Created collection: {cfg['collection']} (dim={cfg['dim']})")


def index_chunks(
    model_key: str,
    chunks: list[Chunk],
    batch_size: int = 64,
) -> None:
    """Embed and upsert chunks into Qdrant."""
    cfg = EMBEDDING_MODELS[model_key]
    model = load_model(model_key)
    client = get_client()

    print(f"Indexing {len(chunks)} chunks with {model_key}...")

    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Batches ({model_key})"):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        # Embed in one batched call — much faster than one-at-a-time
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = []
        for chunk, vector in zip(batch, vectors, strict=True):
            points.append(PointStruct(
                # Qdrant needs a UUID or int for the point ID.
                # We hash the chunk_id to get a deterministic UUID.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.tolist(),
                payload=asdict(chunk),  # Store the whole chunk as metadata
            ))

        client.upsert(collection_name=cfg["collection"], points=points)

    # Verify by counting
    count = client.count(cfg["collection"], exact=True).count
    print(f"Indexed {count} vectors in {cfg['collection']}")


if __name__ == "__main__":
    # Load the corpus and chunk it
    import json

    with open("data/corpus.jsonl") as f:
        records = [json.loads(line) for line in f]

    from src.rag_playground.chunker import chunk_corpus
    chunks = chunk_corpus(records)
    print(f"Total chunks: {len(chunks)}")

    # Index with both models
    for model_key in ["minilm", "bge_large"]:
        print(f"\n{'=' * 60}")
        print(f"Indexing with: {model_key}")
        print("=" * 60)
        create_collection(model_key)
        index_chunks(model_key, chunks)
