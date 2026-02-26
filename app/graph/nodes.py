"""
LangGraph node functions — each node is a pure function that transforms GraphState.

Graph flow:
  START
    → validate_input_node
    → retrieve_node          (hybrid retrieval)
    → grade_documents_node   (check quality; if poor → rewrite_query_node)
    → rewrite_query_node     (optional; loops back to retrieve_node once)
    → generate_node          (LLM generation)
    → validate_output_node
  END

Why LangGraph over a plain chain?
  A chain has no conditional logic — it always retrieves and generates.
  LangGraph lets us detect low-quality retrieval and trigger query rewriting
  before generation, which meaningfully improves answer quality on vague queries.
  This is the architectural insight, not just tooling choice.
"""
from __future__ import annotations

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import RetrievedChunk, Source
from app.graph.prompts import (
    QUERY_REWRITE_PROMPT,
    RELEVANCE_GRADE_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from app.graph.state import GraphState
from app.guardrails.input_guard import validate_input
from app.guardrails.output_guard import (
    build_no_context_response,
    check_retrieval_quality,
    validate_output,
)
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import rerank

logger = get_logger(__name__)


def _get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_with_retry(client: AsyncOpenAI, **kwargs):
    return await client.chat.completions.create(**kwargs)


# ── Node 1: Validate & sanitise input ─────────────────────────────────────────

async def validate_input_node(state: GraphState) -> GraphState:
    cleaned = validate_input(state["question"])
    state["question"] = cleaned
    return state


# ── Node 2: Hybrid retrieval ──────────────────────────────────────────────────

async def retrieve_node(state: GraphState) -> GraphState:
    settings = get_settings()
    query = state.get("rewritten_question") or state["question"]

    chunks = await hybrid_retrieve(
        query=query,
        collection=state["collection"],
        top_k=settings.retrieval_top_k,
    )
    state["retrieved_chunks"] = chunks
    logger.info("Retrieved chunks", count=len(chunks))
    return state


# ── Node 3: Grade retrieved documents ─────────────────────────────────────────

async def grade_documents_node(state: GraphState) -> GraphState:
    """
    Uses a lightweight LLM call to grade each chunk's relevance.
    If fewer than 2 chunks are relevant AND no rewrite has been attempted,
    flags for query rewriting.
    """
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    question = state.get("rewritten_question") or state["question"]

    if state.get("query_rewritten"):
        # Already rewrote once — proceed regardless to avoid infinite loop
        state["proceed_to_generate"] = True
        return state

    client = _get_llm_client()

    relevant_chunks = []
    for chunk in chunks[:10]:  # grade top-10 only for cost efficiency
        try:
            response = await _call_llm_with_retry(
                client=client,
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": RELEVANCE_GRADE_PROMPT.format(
                        question=question, passage=chunk.text[:500]
                    ),
                }],
                max_tokens=5,
                temperature=0.0,
            )
            grade = response.choices[0].message.content.strip().upper()
            if grade.startswith("YES"):
                relevant_chunks.append(chunk)
        except Exception as exc:
            logger.warning("Grading LLM call failed", error=str(exc))
            relevant_chunks.append(chunk)  # assume relevant on error

    state["relevant_chunks"] = relevant_chunks

    if len(relevant_chunks) < 2 and not state.get("query_rewritten"):
        state["proceed_to_generate"] = False
        logger.info("Low relevance — triggering query rewrite", relevant=len(relevant_chunks))
    else:
        state["proceed_to_generate"] = True

    return state


# ── Node 4: Query rewriting (optional) ────────────────────────────────────────

async def rewrite_query_node(state: GraphState) -> GraphState:
    settings = get_settings()
    client = _get_llm_client()
    original = state["question"]

    try:
        response = await _call_llm_with_retry(
            client=client,
            model=settings.llm_model,
            messages=[{
                "role": "user",
                "content": QUERY_REWRITE_PROMPT.format(question=original),
            }],
            max_tokens=128,
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        state["rewritten_question"] = rewritten
        state["query_rewritten"] = True
        logger.info("Query rewritten", original=original, rewritten=rewritten)
    except Exception as exc:
        logger.warning("Query rewrite failed", error=str(exc))
        state["query_rewritten"] = True  # prevent retry loop

    return state


# ── Node 5: LLM generation ────────────────────────────────────────────────────

async def generate_node(state: GraphState) -> GraphState:
    settings = get_settings()
    client = _get_llm_client()

    chunks: list[RetrievedChunk] = state.get("relevant_chunks") or state.get("retrieved_chunks", [])

    # Rerank to top_n before building context
    reranked = rerank(
        query=state.get("rewritten_question") or state["question"],
        chunks=chunks,
        top_n=settings.retrieval_top_n,
    )

    # Check quality threshold
    if not check_retrieval_quality(reranked):
        response = build_no_context_response(reranked)
        state["response"] = response
        return state

    # Build context block
    context_parts = []
    for i, chunk in enumerate(reranked, start=1):
        page_info = f", page {chunk.page}" if chunk.page else ""
        context_parts.append(f"[{i}] ({chunk.filename}{page_info}):\n{chunk.text}")
    context = "\n\n".join(context_parts)

    question = state.get("rewritten_question") or state["question"]

    try:
        completion = await _call_llm_with_retry(
            client=client,
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": USER_PROMPT.format(question=question)},
            ],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        answer_text = completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation failed", error=str(exc))
        answer_text = "An error occurred during answer generation. Please try again."

    if answer_text == "INSUFFICIENT_CONTEXT":
        state["response"] = build_no_context_response(reranked)
        return state

    sources = [
        Source(
            filename=c.filename,
            page=c.page,
            chunk_id=c.chunk_id,
            score=round(c.score, 4),
        )
        for c in reranked
    ]

    state["response"] = validate_output(answer_text, sources)
    state["reranked_chunks"] = reranked
    return state


# ── Node 6: Output validation ──────────────────────────────────────────────────

async def validate_output_node(state: GraphState) -> GraphState:
    # Response is already a validated QueryResponse from generate_node
    # This node is a hook for future post-processing (e.g., toxicity checks)
    return state
