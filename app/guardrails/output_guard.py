"""
Output guardrails.

Ensures the LLM response:
  1. Always includes at least one source citation
  2. Never returns an empty answer
  3. Triggers a "no context" fallback when retrieved chunks are below threshold

This is enforced at the Pydantic model level (QueryResponse) + post-generation check.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolationError
from app.core.logging import get_logger
from app.core.models import QueryResponse, RetrievedChunk, Source

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I don't have enough relevant information in the provided documents "
    "to answer this question confidently. Please upload relevant documents "
    "or rephrase your question."
)


def check_retrieval_quality(chunks: list[RetrievedChunk]) -> bool:
    """
    Returns True if at least one chunk exceeds the similarity threshold.
    Below threshold → return NO_CONTEXT_ANSWER instead of hallucinating.
    """
    settings = get_settings()
    return any(c.score >= settings.similarity_threshold for c in chunks)


def build_no_context_response(chunks: list[RetrievedChunk]) -> QueryResponse:
    logger.warning(
        "All retrieved chunks below similarity threshold",
        max_score=max((c.score for c in chunks), default=0.0),
        threshold=get_settings().similarity_threshold,
    )
    return QueryResponse(
        answer=NO_CONTEXT_ANSWER,
        sources=[],
    )


def validate_output(answer: str, sources: list[Source]) -> QueryResponse:
    """Final validation of the generated answer."""
    if not answer or not answer.strip():
        raise GuardrailViolationError("LLM returned empty answer.")

    return QueryResponse(answer=answer.strip(), sources=sources)
