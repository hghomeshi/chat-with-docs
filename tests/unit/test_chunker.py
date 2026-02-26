"""Unit tests for the chunking module — no API calls, uses recursive mode."""
from __future__ import annotations

from app.ingestion.chunker import chunk_pages

SAMPLE_PAGES = [
    {
        "page": 1,
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation. It allows models "
            "to ground their outputs in external documents rather than relying solely "
            "on parametric knowledge. This reduces hallucination significantly.\n\n"
            "The retrieval component typically uses dense embeddings from transformer "
            "models to find semantically similar passages. The top-k passages are then "
            "concatenated and passed as context to the language model along with the "
            "original query. The model is expected to synthesise an answer grounded "
            "in this context."
        ),
    }
]


def test_chunk_pages_returns_chunks():
    chunks = chunk_pages(
        doc_id="test_doc",
        filename="test.txt",
        pages=SAMPLE_PAGES,
        mode="recursive",
    )
    assert len(chunks) >= 1


def test_chunk_pages_metadata():
    chunks = chunk_pages(
        doc_id="test_doc",
        filename="test.txt",
        pages=SAMPLE_PAGES,
        mode="recursive",
    )
    for chunk in chunks:
        assert chunk.doc_id == "test_doc"
        assert chunk.filename == "test.txt"
        assert len(chunk.text) >= 30
        assert chunk.chunk_id  # non-empty UUID


def test_chunk_pages_empty_page():
    chunks = chunk_pages(
        doc_id="empty_doc",
        filename="empty.txt",
        pages=[{"page": 1, "text": "   "}],
        mode="recursive",
    )
    assert chunks == []


def test_chunk_pages_short_fragments_discarded():
    chunks = chunk_pages(
        doc_id="frag_doc",
        filename="frag.txt",
        pages=[{"page": 1, "text": "Hi."}],
        mode="recursive",
    )
    # "Hi." is 3 chars — below 30-char threshold
    assert chunks == []
