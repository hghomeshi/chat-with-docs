"""
Ingestion orchestration — ties parser → chunker → embedder → vector store together.
Called by the API /ingest endpoint and can also be run as a standalone CLI script.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import IngestResponse
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import parse_document
from app.ingestion.vector_store import ensure_collection, get_client, upsert_chunks

logger = get_logger(__name__)


async def ingest_document(file_path: Path, collection: str | None = None) -> IngestResponse:
    """
    Full ingestion pipeline for a single document.

    Steps:
      1. Parse → raw page text
      2. Chunk → DocumentChunk list
      3. Embed → float vectors (dense + sparse)
      4. Upsert → Qdrant

    Returns an IngestResponse with counts for API / CLI feedback.
    """
    settings = get_settings()
    collection = collection or settings.qdrant_collection

    logger.info("Starting ingestion", filename=file_path.name, collection=collection)

    doc_id, pages = parse_document(file_path)
    chunks = chunk_pages(doc_id=doc_id, filename=file_path.name, pages=pages)

    if not chunks:
        logger.warning("No chunks produced — document may be empty", filename=file_path.name)
        return IngestResponse(
            doc_id=doc_id,
            filename=file_path.name,
            chunks_indexed=0,
            message="Document parsed but no text content found.",
        )

    dense_embeddings, sparse_embeddings = await embed_chunks(chunks)

    client = get_client()
    await ensure_collection(client, collection)
    await upsert_chunks(client, collection, chunks, dense_embeddings, sparse_embeddings)

    logger.info("Ingestion complete", doc_id=doc_id, chunks=len(chunks))
    return IngestResponse(
        doc_id=doc_id,
        filename=file_path.name,
        chunks_indexed=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{file_path.name}'.",
    )
