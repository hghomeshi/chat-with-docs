# Chat With Your Docs

A production-grade conversational RAG system. Ask questions over your PDF, TXT, MD,
and DOCX documents. Built with FastAPI, LangGraph, Qdrant, and OpenAI.

---

## Quick Setup

```bash
# 1. Clone and install
git clone https://github.com/your-handle/chat-with-docs
cd chat-with-docs
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# → Set OPENAI_API_KEY in .env

# 3. Start (Docker — recommended)
docker-compose up --build

# API:  http://localhost:8000/docs
# UI:   http://localhost:8501

# Or run locally (requires Qdrant running separately)
uvicorn app.main:app --reload
streamlit run ui/app.py
```

**Guardrails (optional env overrides):**

```bash
PII_GUARD_PROVIDER=presidio
ENABLE_PII_SCRUBBING=true
INJECTION_GUARD_PROVIDER=local
LOCAL_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection
LOCAL_INJECTION_BLOCK_THRESHOLD=0.6
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Client (Browser)                     │
│                      Streamlit UI :8501                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                    FastAPI API :8000                        │
│              POST /ingest   POST /query                     │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
  ┌──────────▼──────────┐   ┌─────────▼─────────────────────┐
  │  Ingestion Pipeline │   │   LangGraph RAG Pipeline      │
  │  parse → chunk      │   │                               │
  │  embed → upsert     │   │  validate_input               │
  └──────────┬──────────┘   │       ↓                       │
             │              │  retrieve (hybrid RRF)        │
  ┌──────────▼──────────┐   │       ↓                       │
  │   Qdrant (Docker)   │◄──│  grade_documents              │
  │   HNSW + payloads   │   │       ↓ (if poor quality)     │
  └─────────────────────┘   │  rewrite_query ──────────┐    │
                            │       ↓ (if ok)          │    │
                            │  generate (GPT-4o-mini)  │    │
                            │  [rerank → context →     │    │
                            │   guardrails → answer]   │    │
                            │       ↓                  │    │
                            │  validate_output         │    │
                            └──────────────────────────┴────┘
```

---

## RAG / LLM Decisions

### Chunking
**SemanticChunker** (LangChain) over fixed-size. Splits on embedding-similarity
breakpoints so concept explanations aren't cut mid-paragraph. Falls back to
`RecursiveCharacterTextSplitter` if the API is unavailable. Chunk size: 512 tokens,
50-token overlap.

### Embedding Model
**text-embedding-3-small** (OpenAI). Best cost/quality ratio for English technical docs.
Upgrade path: `text-embedding-3-large` for domain-dense corpora.

### LLM
**GPT-4o-mini** — cost-efficient, 128k context window, strong instruction-following.
The system prompt strictly instructs the model to cite sources and say
"INSUFFICIENT_CONTEXT" if context is inadequate — the output guard detects this
and returns a graceful fallback.

### Vector Database
**Qdrant** (self-hosted via Docker). Chosen over ChromaDB (limited production filters),
pgvector (Postgres coupling), and Pinecone (paid, SaaS-only). Qdrant Cloud is a
zero-code-change upgrade path.

### Retrieval
**Hybrid search (dense + sparse BM25) using Qdrant's native hybrid search**, then **cross-encoder reranking**.
- Dense: catches paraphrased/conceptual queries
- Sparse (BM25 via Splade): catches exact technical terms (model names, acronyms, version strings)
- RRF: parameter-light fusion, handled natively by Qdrant without pulling documents into memory
- Cross-encoder (`ms-marco-MiniLM-L-6-v2`): re-scores top-20 → top-5, runs locally

### Orchestration
**LangGraph** over a plain chain. The graph detects low-quality retrieval and
rewrites the query before generation — one retry maximum to prevent loops.
This is the key architectural decision: failure paths are first-class, not afterthoughts.

### Prompt Engineering
System prompt enforces: use only provided context, cite sources, return a sentinel
string (`INSUFFICIENT_CONTEXT`) when context is inadequate. The output guard detects
the sentinel and substitutes a user-friendly message. Prompts live in `app/graph/prompts.py`.

### Guardrails
- **Input**: Length check, PII scrubbing via **Presidio** (email + phone), prompt-injection detection via
    a **local Transformers classifier** (default: `protectai/deberta-v3-base-prompt-injection`) with regex fallback.
- **Output**: Pydantic-typed response, similarity threshold check, INSUFFICIENT_CONTEXT sentinel

**Note:** The first run will download the local injection model weights from Hugging Face if they are not cached.

**Guardrail configuration (env):**
- `PII_GUARD_PROVIDER=presidio|regex`
- `ENABLE_PII_SCRUBBING=true|false`
- `INJECTION_GUARD_PROVIDER=local|regex|rebuff`
- `LOCAL_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection`
- `LOCAL_INJECTION_BLOCK_THRESHOLD=0.6`

### Observability
- **Structured JSON logging** via `structlog` with correlation IDs per request
- **LangSmith tracing** (optional, enable via env) — traces every graph node
- **RAGAS evaluation** as a CI artefact — faithfulness + answer relevancy on a golden set

---

## Productionising on AWS

-----------------------------------------------------------------------------------------
| Concern           | Local (this repo)     | AWS Production                            |
|-------------------|-----------------------|-------------------------------------------|
| Vector DB         | Qdrant (Docker)       | Qdrant Cloud or OpenSearch k-NN           |
| API               | Docker Compose        | ECS Fargate (auto-scaling)                |
| Ingestion trigger | Manual POST /ingest   | S3 event → Lambda → ECS task              |
| Caching           | None                  | ElastiCache (Redis) semantic cache        |
| Secrets           | `.env` file           | AWS Secrets Manager                       |
| CI/CD             | GitHub Actions        | GitHub Actions → ECR → ECS rolling deploy |
| Observability     | structlog + LangSmith | CloudWatch + LangSmith                    |
| Auth              | None (noted below)    | Cognito + API Gateway                     |
-----------------------------------------------------------------------------------------

**Scaling notes:**
- Fargate scales on CPU/memory; set min=2 for HA
- Qdrant scales via collection sharding (Qdrant Cloud manages this)
- ElastiCache semantic cache: hash query embedding → cache hit skips LLM entirely
- Rate limiting: FastAPI middleware + API Gateway usage plans

---

## Engineering Standards

**Applied:**
- Type hints throughout; Pydantic v2 for all I/O schemas
- Fully Async FastAPI and LangGraph pipeline for high concurrency
- `pydantic-settings` for environment-based config with a committed `.env.example`
- `structlog` structured JSON logging
- `ruff` (linting) + `mypy` (type checking) via `pre-commit` hooks
- Unit tests for chunking, guardrails, and RRF logic
- RAGAS evaluation suite as a CI gate
- Graceful degradation: semantic chunking falls back to recursive; reranking falls back to score order
- Resilient LLM calls: `tenacity` retry logic with exponential backoff for transient API failures

**Deliberately skipped (and why):**
- **Authentication / multi-tenancy** — out of scope for this submission; noted above
- **Streaming responses** — SSE adds complexity; buffered responses sufficient for demo
- **Full integration tests** — time-boxed; golden set covers critical paths end-to-end
- **Helm charts / Terraform** — Docker Compose covers the submission scope; IaC documented above
- **Table/image extraction from PDFs** — PyMuPDF extracts text only; Unstructured.io noted as upgrade

---

## How I Used AI Coding Tools

I used **GitHub Copilot** during development.

**Where AI tools accelerated me:**
- Boilerplate: Dockerfile, docker-compose, FastAPI route scaffolding, pytest stubs
- Repetitive patterns: Pydantic model definitions, logging calls, try/except wrappers
- Looking up API signatures (Qdrant client, LangGraph edge syntax)

**Where I wrote without AI assistance:**
- All architectural decisions and their rationale (this README)
- The LangGraph graph topology — the conditional retry loop is a deliberate design choice
- Prompt engineering — the system prompt, sentinel strategy, and output guard are mine
- Test cases — the guardrail tests and RRF unit test were written to spec, not generated
- The trade-off reasoning (vector DB selection, hybrid vs pure semantic, chunking strategy)

**My do's and don'ts with AI coding assistants:**
- Do's: 
    - Accept boilerplate, reject logic — always read what's generated before committing
    - Use AI for first drafts, own the review — generated code goes through the same ruff/mypy as handwritten code
    - Regenerate when AI fills in values it shouldn't know (hardcoded paths, wrong model names)
- Don'ts:
    - Never let AI write tests based on its own implementation — tests must reflect intent, not implementation
    - Never accept generated README prose— it sounds generic; this document reflects my actual reasoning
---

## What I'd Do Differently With More Time

1. **GraphRAG** — build a knowledge graph over inter-document references for multi-hop reasoning
2. **Streaming responses** — SSE via FastAPI `StreamingResponse` for real-time token delivery
3. **Document metadata filters** — expose filename/date filters to the user in the UI
4. **Fine-tuned embeddings** — fine-tune `text-embedding-3-small` on domain-specific corpus
5. **Async ingestion queue** — Celery + Redis for large batch uploads that currently block
6. **Human-in-the-loop feedback** — thumbs up/down feeds back into the RAGAS golden set automatically
7. **Multi-modal support** — PyMuPDF extracts images; pass to GPT-4o vision for figure/table understanding
8. **Auth** — JWT-based multi-tenant isolation so each user only queries their own documents

---

## Known Limitations

- Tables and code blocks in PDFs are treated as plain text — layout information is lost
- BM25 index is rebuilt in-memory on every query — acceptable at <10k chunks, needs caching at scale
- No document deduplication beyond hash-based doc_id — re-uploading the same file will not create duplicate chunks but re-processes the full pipeline
- Cross-encoder reranking loads a ~90MB model on first request — warm-up in production via startup event
