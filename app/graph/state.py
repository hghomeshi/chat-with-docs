"""
LangGraph state schema — typed dict that flows through all graph nodes.
"""
from __future__ import annotations

from typing import TypedDict, Any

from app.core.models import QueryResponse, RetrievedChunk


class GraphState(TypedDict, total=False):
    # Input
    question: str
    collection: str

    # Mid-graph
    rewritten_question: str | None
    query_rewritten: bool
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    proceed_to_generate: bool

    # Output
    response: QueryResponse
