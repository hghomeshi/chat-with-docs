"""
Document parsing layer.

Supports:
  - PDF  → PyMuPDF (fitz) — fast, handles multi-column layouts
  - TXT / MD → plain read
  - DOCX     → python-docx

Extension point: swap in Unstructured.io for complex mixed-format docs.
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from app.core.exceptions import DocumentParsingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    sha.update(path.read_bytes())
    return sha.hexdigest()[:16]


def parse_pdf(path: Path) -> list[dict]:
    """Return list of {page, text} dicts from a PDF."""
    pages = []
    try:
        doc = fitz.open(str(path))
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_num, "text": text})
        doc.close()
    except Exception as exc:
        raise DocumentParsingError(f"PDF parse failed for {path.name}: {exc}") from exc
    return pages


def parse_txt(path: Path) -> list[dict]:
    try:
        return [{"page": None, "text": path.read_text(encoding="utf-8")}]
    except Exception as exc:
        raise DocumentParsingError(f"TXT parse failed for {path.name}: {exc}") from exc


def parse_docx(path: Path) -> list[dict]:
    try:
        doc = DocxDocument(str(path))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page": None, "text": full_text}]
    except Exception as exc:
        raise DocumentParsingError(f"DOCX parse failed for {path.name}: {exc}") from exc


_PARSERS = {
    "application/pdf": parse_pdf,
    "text/plain": parse_txt,
    "text/markdown": parse_txt,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
}


def parse_document(path: Path) -> tuple[str, list[dict]]:
    """
    Returns (doc_id, pages) where each page is {page, text}.
    doc_id is a hash of the file bytes — deterministic, collision-resistant.
    """
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        # fall back to suffix-based detection
        suffix_map = {".pdf": "application/pdf", ".txt": "text/plain",
                      ".md": "text/markdown", ".docx":
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
        mime = suffix_map.get(path.suffix.lower(), "text/plain")

    parser = _PARSERS.get(mime)
    if parser is None:
        raise DocumentParsingError(f"Unsupported file type: {mime} ({path.name})")

    doc_id = _hash_file(path)
    logger.info("Parsing document", filename=path.name, doc_id=doc_id, mime=mime)
    pages = parser(path)
    logger.info("Parsed document", filename=path.name, pages=len(pages))
    return doc_id, pages