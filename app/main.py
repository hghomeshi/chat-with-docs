"""
FastAPI application entry point.

Startup:
  - Configure structured logging
  - Optionally enable LangSmith tracing via env vars
  - Register routes with /api/v1 prefix

CORS is open by default (development) — tighten via CORS_ORIGINS env var in production.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()

# Enable LangSmith tracing if configured
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

app = FastAPI(
    title="Chat With Your Docs",
    description="Production-grade RAG API with hybrid retrieval, reranking, and self-corrective LangGraph pipeline.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["ops"])
async def root() -> dict:
    return {"message": "Chat With Your Docs API", "docs": "/docs"}
