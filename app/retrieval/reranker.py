"""
Cross-encoder reranking.

After hybrid retrieval returns top_k=20 candidates, a cross-encoder rescores
each (query, chunk) pair and keeps top_n=5 for context assembly.

Why rerank?
  Embedding models score query–chunk relevance independently (bi-encoder).
  Cross-encoders jointly encode the pair — dramatically better precision,
  at the cost of O(k) inference calls. Worth it on k=20, not feasible on k=10k.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (sentence-transformers)
  - Fast (~50ms for 20 pairs on CPU), runs locally, no API cost
  - Upgrade path: Cohere Rerank API for higher quality on specialized domains

Skipped: ColBERT late-interaction — excellent but adds complexity beyond scope.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import RetrievedChunk

logger = get_logger(__name__)

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model", model=_RERANK_MODEL)
        _cross_encoder = CrossEncoder(_RERANK_MODEL)
    return _cross_encoder


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank chunks using a cross-encoder; return top_n with updated scores.
    Falls back gracefully to original ordering if cross-encoder unavailable.
    """
    settings = get_settings()
    top_n = top_n or settings.retrieval_top_n

    if not chunks:
        return []

    try:
        model = _get_cross_encoder()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )
        result = []
        for chunk, score in ranked[:top_n]:
            chunk.score = float(score)
            result.append(chunk)

        logger.info("Reranking complete", input=len(chunks), output=len(result))
        return result

    except Exception as exc:
        logger.warning("Cross-encoder reranking failed, using original order", error=str(exc))
        return chunks[:top_n]
