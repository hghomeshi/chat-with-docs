"""
RAGAS golden set — 10 Q&A pairs for CI evaluation.

These are written against the included sample document (docs/sample_rag_paper.txt).
The CI step runs this suite and fails if faithfulness < 0.7 or answer_relevancy < 0.7.

Intentionally kept small — the goal is catching regressions, not comprehensive coverage.
Expand this set as the document corpus grows.
"""

GOLDEN_QA = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": "RAG combines a retrieval component with a language model generator. "
                        "The retrieval component finds relevant documents, which are passed as "
                        "context to the LLM to ground its output.",
    },
    {
        "question": "What are the main failure modes of naive RAG systems?",
        "ground_truth": "Naive RAG fails due to poor chunking that splits concepts mid-sentence, "
                        "pure semantic search missing exact technical terms, and lack of "
                        "reranking causing low-precision context.",
    },
    {
        "question": "What is Reciprocal Rank Fusion?",
        "ground_truth": "RRF is a rank aggregation method that combines results from multiple "
                        "retrieval systems using the formula score(d) = sum(1 / (k + rank_i(d))), "
                        "where k=60 is the smoothing constant.",
    },
    {
        "question": "Why use a cross-encoder for reranking?",
        "ground_truth": "Cross-encoders jointly encode query and document, giving better relevance "
                        "scores than bi-encoders which score them independently. They are applied "
                        "to a small candidate set (top-k=20) for cost efficiency.",
    },
    {
        "question": "What embedding model is used and why?",
        "ground_truth": "text-embedding-3-small from OpenAI, chosen for its strong benchmark "
                        "performance, 8191-token context window, and low cost per token.",
    },
]
