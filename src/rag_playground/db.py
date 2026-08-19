"""
Qdrant client wrapper.

Why this file exists:
Every part of the RAG pipeline (indexing, querying, benchmarks) needs to talk
to Qdrant. Centralizing the connection here means:
1. One place to configure host/port
2. One place to switch to remote Qdrant later
3. Consistent client instance across the codebase
"""

from qdrant_client import QdrantClient


def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Return a Qdrant client connected to a local or remote instance."""
    return QdrantClient(host=host, port=port)


def health_check(client: QdrantClient) -> bool:
    """Verify Qdrant is reachable. Returns True if healthy, False otherwise."""
    try:
        # get_collections is a no-op that hits the server — good health check
        client.get_collections()
        return True
    except Exception:
        return False
