"""
RAGAS evaluation test — runs against a live stack.

Marked as 'evaluation' — skipped in unit test runs, run explicitly in CI via:
  pytest tests/evaluation/ -m evaluation

Requires: OPENAI_API_KEY, running Qdrant, and the sample docs pre-ingested.
"""
from __future__ import annotations

import os

import pytest
from datasets import Dataset

from tests.evaluation.golden_set import GOLDEN_QA


@pytest.mark.evaluation
@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set — skipping evaluation tests",
)
def test_ragas_faithfulness_and_relevancy():
    """
    Runs the golden Q&A set through the live RAG pipeline and evaluates with RAGAS.
    Fails CI if faithfulness < 0.7 or answer_relevancy < 0.7.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        pytest.skip("ragas not installed")

    from app.core.config import get_settings
    from app.graph.pipeline import rag_graph
    from app.graph.state import GraphState

    settings = get_settings()
    questions, answers, contexts, ground_truths = [], [], [], []

    for qa in GOLDEN_QA:
        state: GraphState = {
            "question": qa["question"],
            "collection": settings.qdrant_collection,
            "query_rewritten": False,
        }
        result = rag_graph.invoke(state)
        response = result.get("response")
        reranked = result.get("reranked_chunks", [])

        questions.append(qa["question"])
        answers.append(response.answer if response else "")
        contexts.append([c.text for c in reranked])
        ground_truths.append(qa["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"\nRAGAS scores: {results}")

    assert results["faithfulness"] >= 0.7, f"Faithfulness too low: {results['faithfulness']}"
    assert results["answer_relevancy"] >= 0.7, f"Answer relevancy too low: {results['answer_relevancy']}"
