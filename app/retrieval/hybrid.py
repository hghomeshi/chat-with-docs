"""
Hybrid retrieval: Dense (semantic) + Sparse (BM25) using Qdrant's native hybrid search.

Why hybrid?
  - Dense search misses exact technical terms: "RFC 7519", "HNSW", "ALS matrix factorisation"
  - BM25 misses paraphrased / conceptual queries: "how does the model learn user preferences?"
  - Qdrant handles RRF fusion natively, eliminating the need to pull all documents into memory.

After fusion, top_k candidates are reranked by a cross-encoder for precision.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import RetrievedChunk
from app.ingestion.embedder import embed_query
from app.ingestion.vector_store import hybrid_search, get_client

logger = get_logger(__name__)

async def hybrid_retrieve(
    query: str,
    collection: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Runs native hybrid retrieval (dense + sparse) using Qdrant.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k

    dense_query_vector, sparse_query_vector = await embed_query(query)
    client = get_client()
    
    results = await hybrid_search(
        client=client,
        collection=collection,
        query_vector=dense_query_vector,
        sparse_query_vector=sparse_query_vector,
        top_k=top_k
    )
    
    logger.info("Hybrid retrieval complete", query=query[:60], returned=len(results))
    return results
