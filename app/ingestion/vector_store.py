"""
Qdrant vector store interface.

Why Qdrant?
  - Rust-based engine: fast ANN search with HNSW indexing
  - Rich payload filtering (filter by filename, date, doc_id, etc.)
  - Easy local (Docker) → Qdrant Cloud migration — same client, just swap URL
  - Named vectors: extensible to multi-vector setups (e.g. title + body)

ChromaDB was considered — excellent for pure local prototyping — but lacks
production-grade filtering and horizontal scaling story.

pgvector was considered — great if you already have Postgres — but adds
infra coupling and requires manual HNSW tuning.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.core.models import DocumentChunk, RetrievedChunk

logger = get_logger(__name__)


def get_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


async def ensure_collection(client: AsyncQdrantClient, collection: str) -> None:
    """Create collection if it does not exist — idempotent."""
    settings = get_settings()
    collections_response = await client.get_collections()
    existing = [c.name for c in collections_response.collections]
    if collection not in existing:
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        logger.info("Created Qdrant collection with dense and sparse vectors", collection=collection)
    else:
        logger.debug("Collection already exists", collection=collection)


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection: str,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    sparse_embeddings: list[dict[str, list[float] | list[int]]] | None = None,
) -> None:
    """Upsert chunks + their vectors into Qdrant."""
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
        vector_dict: dict[str, Any] = {"dense": embedding}
        if sparse_embeddings and i < len(sparse_embeddings):
            vector_dict["sparse"] = {
                "indices": sparse_embeddings[i]["indices"],
                "values": sparse_embeddings[i]["values"],
            }
            
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=vector_dict,
                payload={
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
        )
    try:
        await client.upsert(collection_name=collection, points=points)
        logger.info("Upserted chunks", collection=collection, count=len(points))
    except Exception as exc:
        raise RetrievalError(f"Qdrant upsert failed: {exc}") from exc


async def hybrid_search(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    sparse_query_vector: dict[str, list[float] | list[int]],
    top_k: int,
    filename_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Native hybrid search using Qdrant."""
    qdrant_filter = None
    if filename_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename_filter))]
        )
    try:
        from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector
        
        prefetch = [
            Prefetch(
                query=query_vector,
                using="dense",
                limit=top_k,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_query_vector["indices"],  # type: ignore
                    values=sparse_query_vector["values"],  # type: ignore
                ),
                using="sparse",
                limit=top_k,
            )
        ]
        
        response = await client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points if hasattr(response, "points") else response
    except Exception as exc:
        raise RetrievalError(f"Qdrant hybrid search failed: {exc}") from exc

    return [
        RetrievedChunk(
            chunk_id=str(hit.id),  # type: ignore
            doc_id=hit.payload["doc_id"],  # type: ignore
            filename=hit.payload["filename"],  # type: ignore
            page=hit.payload.get("page"),  # type: ignore
            text=hit.payload["text"],  # type: ignore
            score=hit.score,  # type: ignore
            metadata={k: v for k, v in hit.payload.items()  # type: ignore
                       if k not in ("doc_id", "filename", "page", "text")},
        )
        for hit in results
    ]



