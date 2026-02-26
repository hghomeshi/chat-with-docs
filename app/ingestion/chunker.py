"""
Chunking strategy.

Default: SemanticChunker (LangChain) — splits on embedding-similarity breakpoints
so a concept explanation is never cut in half.

Fallback: RecursiveCharacterTextSplitter — fast, deterministic, used in tests.

Why not fixed-size only?  Technical docs commonly have multi-paragraph concepts;
fixed-size chunks create artificial breaks that fragment the answer context and
degrade retrieval precision.

Known limitation: tables and code blocks are treated as plain text — next iteration
would detect and preserve these via regex or Unstructured.io element types.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal, Protocol

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import DocumentChunk

logger = get_logger(__name__)

ChunkMode = Literal["semantic", "recursive"]


class TextSplitter(Protocol):
    def split_text(self, text: str) -> list[str]:
        ...


def _make_recursive_splitter(chunk_size: int, chunk_overlap: int) -> TextSplitter:
    separators = ["\n\n", "\n", ". ", " ", ""]
    try:
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    except Exception as exc:
        logger.warning(
            "LangChain text splitter unavailable, using fallback",
            error=str(exc),
        )
        return _FallbackRecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )


class _FallbackRecursiveSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        splits = [text]
        for sep in self.separators:
            splits = self._split_on_separator(splits, sep)
        return self._merge_splits(splits)

    def _split_on_separator(self, chunks: list[str], separator: str) -> list[str]:
        if separator == "":
            return chunks

        next_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                next_chunks.append(chunk)
                continue
            parts = chunk.split(separator)
            if len(parts) == 1:
                next_chunks.append(chunk)
                continue
            for idx, part in enumerate(parts):
                if not part:
                    continue
                if idx < len(parts) - 1:
                    part = f"{part}{separator}"
                next_chunks.append(part)
        return next_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        total = 0
        step = max(self.chunk_size - self.chunk_overlap, 1)

        def flush() -> None:
            nonlocal current, total
            if not current:
                return
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if self.chunk_overlap > 0 and chunk:
                overlap = chunk[-self.chunk_overlap :]
                current = [overlap] if overlap else []
                total = len(overlap)
            else:
                current = []
                total = 0

        for split in splits:
            if not split:
                continue
            if len(split) > self.chunk_size and "" not in self.separators:
                for idx in range(0, len(split), step):
                    piece = split[idx : idx + self.chunk_size]
                    if total + len(piece) > self.chunk_size and current:
                        flush()
                    current.append(piece)
                    total += len(piece)
                continue

            if total + len(split) > self.chunk_size and current:
                flush()
            current.append(split)
            total += len(split)

        flush()
        return chunks


def chunk_pages(
    doc_id: str,
    filename: str,
    pages: list[dict],
    mode: ChunkMode | None = None,
) -> list[DocumentChunk]:
    """
    Takes parsed pages ({page, text}) and returns a flat list of DocumentChunks.
    """
    settings = get_settings()
    mode = mode or ("semantic" if settings.semantic_chunking else "recursive")
    min_chunk_chars = settings.min_chunk_chars

    chunks: list[DocumentChunk] = []

    for page_info in pages:
        page_num = page_info.get("page")
        text = page_info["text"].strip()
        if not text:
            continue

        raw_chunks = _split_text(text, mode, settings.chunk_size, settings.chunk_overlap)

        for raw in raw_chunks:
            chunk_text = raw.strip()
            if len(chunk_text) < min_chunk_chars:
                # discard noise fragments
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=filename,
                    page=page_num,
                    text=chunk_text,
                    metadata={"mode": mode},
                )
            )

    logger.info("Chunking complete", filename=filename, total_chunks=len(chunks), mode=mode)
    return chunks


def _split_text(text: str, mode: ChunkMode, chunk_size: int, chunk_overlap: int) -> list[str]:
    if mode == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings

            settings = get_settings()
            from pydantic import SecretStr
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=SecretStr(settings.openai_api_key),
            )
            splitter: Any = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
            docs = splitter.create_documents([text])
            return [d.page_content for d in docs]
        except Exception as exc:
            logger.warning("SemanticChunker failed, falling back to recursive", error=str(exc))
            # graceful fallback — never fail the ingestion
            mode = "recursive"

    splitter = _make_recursive_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)