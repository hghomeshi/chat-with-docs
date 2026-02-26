from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any


class DocumentChunk(BaseModel):
    """A single chunk of text with its provenance metadata."""
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    """A chunk that has been retrieved and scored."""
    score: float = 0.0


class IngestRequest(BaseModel):
    filename: str


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4096)
    collection: str = "documents"
    top_n: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    filename: str
    page: int | None
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_rewritten: bool = False
    rewritten_question: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm: str