"""
Embedding layer.

Model: text-embedding-3-small (OpenAI)
  - 1536 dimensions, strong benchmark performance, cost-efficient
  - Upgrade path: text-embedding-3-large (3072 dims) for higher recall on dense corpora

Batching: OpenAI allows up to 2048 inputs per request — we batch in groups of 100
to stay well within rate limits and handle large ingestion jobs gracefully.
"""
from __future__ import annotations

from openai import AsyncOpenAI
from fastembed import SparseTextEmbedding

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.core.models import DocumentChunk

logger = get_logger(__name__)

BATCH_SIZE = 100

_sparse_model: SparseTextEmbedding | None = None

def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse embedding model")
        _sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return _sparse_model


async def embed_chunks(chunks: list[DocumentChunk]) -> tuple[list[list[float]], list[dict[str, list[float] | list[int]]]]:
    """
    Returns a tuple of (dense_embeddings, sparse_embeddings), one per chunk, in the same order.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    all_dense_embeddings: list[list[float]] = []
    all_sparse_embeddings: list[dict[str, list[float] | list[int]]] = []

    sparse_model = _get_sparse_model()

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c.text for c in batch]
        try:
            # Dense embeddings
            response = await client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            batch_dense = [item.embedding for item in response.data]
            all_dense_embeddings.extend(batch_dense)
            
            # Sparse embeddings
            batch_sparse_gen = sparse_model.embed(texts)
            for sparse_vec in batch_sparse_gen:
                all_sparse_embeddings.append({
                    "indices": sparse_vec.indices.tolist(),
                    "values": sparse_vec.values.tolist()
                })
                
            logger.info(
                "Embedded batch",
                batch_start=batch_start,
                batch_size=len(batch),
                model=settings.embedding_model,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed at batch {batch_start}: {exc}") from exc

    return all_dense_embeddings, all_sparse_embeddings


async def embed_query(text: str) -> tuple[list[float], dict[str, list[float] | list[int]]]:
    """Embed a single query string into dense and sparse vectors."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    sparse_model = _get_sparse_model()
    
    try:
        # Dense
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=[text],
        )
        dense_vec = response.data[0].embedding
        
        # Sparse
        sparse_gen = list(sparse_model.embed([text]))
        sparse_vec = {
            "indices": sparse_gen[0].indices.tolist(),
            "values": sparse_gen[0].values.tolist()
        }
        
        return dense_vec, sparse_vec
    except Exception as exc:
        raise EmbeddingError(f"Query embedding failed: {exc}") from exc
