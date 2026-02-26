"""
LangGraph pipeline definition.

Conditional edge logic:
  After grade_documents_node:
    - If proceed_to_generate=False AND not yet rewritten → rewrite_query_node → retrieve_node
    - Otherwise → generate_node

This implements the self-corrective RAG loop without infinite recursion.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    generate_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    validate_input_node,
    validate_output_node,
)
from app.graph.state import GraphState


def _should_rewrite(state: GraphState) -> str:
    if not state.get("proceed_to_generate", True):
        return "rewrite"
    return "generate"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_output", validate_output_node)

    # Edges
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        _should_rewrite,
        {
            "rewrite": "rewrite_query",
            "generate": "generate",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()


# Module-level compiled graph — singleton, reused across requests
rag_graph = build_rag_graph()
