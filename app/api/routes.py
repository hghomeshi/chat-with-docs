"""
FastAPI routes for ingestion and query endpoints.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.exceptions import (
	ChatWithDocsError,
	DocumentParsingError,
	EmbeddingError,
	GuardrailViolationError,
	RetrievalError,
)
from app.core.logging import get_logger
from app.core.models import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.graph.pipeline import rag_graph
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import get_client

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
	settings = get_settings()
	qdrant_status = "ok"
	try:
		client = get_client()
		await client.get_collections()
	except Exception as exc:
		logger.warning("Qdrant health check failed", error=str(exc))
		qdrant_status = "unavailable"

	llm_status = "configured" if settings.openai_api_key else "missing_api_key"

	status = "ok" if qdrant_status == "ok" else "degraded"
	return HealthResponse(status=status, qdrant=qdrant_status, llm=llm_status)


@router.post("/ingest", response_model=IngestResponse, status_code=201, tags=["ingestion"])
async def ingest(file: Annotated[UploadFile, File(...)]) -> IngestResponse:
	if not file.filename:
		raise HTTPException(status_code=400, detail="Filename is required.")

	try:
		suffix = Path(file.filename).suffix
		with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
			contents = await file.read()
			tmp.write(contents)
			temp_path = Path(tmp.name)

		response = await ingest_document(temp_path)
		return response
	except (DocumentParsingError, EmbeddingError, RetrievalError, GuardrailViolationError) as exc:
		logger.warning("Ingestion failed", error=str(exc))
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except ChatWithDocsError as exc:
		logger.error("Ingestion failed", error=str(exc))
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	except Exception as exc:
		logger.error("Unexpected ingestion error", error=str(exc))
		raise HTTPException(status_code=500, detail="Unexpected ingestion error.") from exc


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query_docs(request: QueryRequest) -> QueryResponse:
	try:
		result = await rag_graph.ainvoke({
			"question": request.question,
			"collection": request.collection,
		})
		response = result.get("response")
		if not response:
			raise HTTPException(status_code=500, detail="No response generated.")
		return response
	except GuardrailViolationError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except RetrievalError as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc
	except ChatWithDocsError as exc:
		logger.error("Query failed", error=str(exc))
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	except Exception as exc:
		logger.error("Unexpected query error", error=str(exc))
		raise HTTPException(status_code=500, detail="Unexpected query error.") from exc


@router.get("/documents", tags=["documents"])
async def list_documents(collection: str | None = None) -> dict:
	settings = get_settings()
	collection = collection or settings.qdrant_collection

	try:
		client = get_client()
		records, _ = await client.scroll(
			collection_name=collection,
			limit=10_000,
			with_payload=True,
			with_vectors=False,
		)
		records_list = [{"id": str(r.id), **(r.payload or {})} for r in records]
	except Exception as exc:
		logger.warning("Failed to list documents", error=str(exc))
		raise HTTPException(status_code=503, detail="Qdrant unavailable.") from exc

	docs: dict[tuple[str | None, str], dict[str, str | None]] = {}
	for record in records_list:
		filename = record.get("filename")
		if not filename:
			continue
		doc_id = record.get("doc_id")
		docs.setdefault((doc_id, filename), {"doc_id": doc_id, "filename": filename})

	return {"documents": list(docs.values())}
