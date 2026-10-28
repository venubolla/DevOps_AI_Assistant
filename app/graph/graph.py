from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    MAX_RETRIES,
    agent_node,
    grounding_node,
    relevance_node,
    retrieval_failed_node,
    retry_search_node,
    rewrite_query_node,
    scope_check_node,
    tool_node,
    validate_graph_state,
)
from app.graph.state import RAGGraphState


def route_after_agent(
    state: RAGGraphState,
) -> str:
    """
    Decide whether the Agent requested a tool or generated a final answer.
    """
    if state.get("agent_failed", False):
        return "end"

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", []):
        return "tool"

    # A tool-less reply still has to be checked against the knowledge base.
    if state.get("direct_answer", False) and not state.get("retrieved_documents"):
        return "scope_check"

    return "grounding"


def route_after_scope_check(
    state: RAGGraphState,
) -> str:
    """
    Decide whether a tool-less Agent reply is in scope for the knowledge base.
    """

    decision = state.get("scope_decision")

    if decision == "retrieve":
        return "agent"

    if decision in ("conversational", "refused"):
        return "end"

    return "failed"


def route_after_relevance(
    state: RAGGraphState,
) -> str:
    """
    Decide whether retrieval passed validation
    or needs one retry.
    """

    if state.get(
        "relevance_passed",
        False,
    ):
        return "answer"

    retry_count = state.get(
        "retry_count",
        0,
    )

    if retry_count < MAX_RETRIES:
        return "rewrite"

    return "end"


def route_after_grounding(
    state: RAGGraphState,
) -> str:
    """
    Decide whether the generated answer passed grounding validation.
    """

    if state.get("grounding_passed", False):
        return "pass"

    return "fail"


def route_after_state_validation(
    state: RAGGraphState,
) -> str:
    """
    Decide whether graph state is valid enough
    to continue to the Agent.
    """

    if state.get("answer"):
        return "end"

    return "agent"


builder = StateGraph(RAGGraphState)

builder.add_node(
    "validate_state",
    validate_graph_state,
)

builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tool",
    tool_node,
)

builder.add_node(
    "scope_check",
    scope_check_node,
)

builder.add_node(
    "relevance",
    relevance_node,
)

builder.add_node(
    "grounding",
    grounding_node,
)

builder.add_node(
    "rewrite",
    rewrite_query_node,
)

builder.add_node(
    "retry_search",
    retry_search_node,
)

builder.add_node(
    "retrieval_failed",
    retrieval_failed_node,
)


builder.add_edge(
    START,
    "validate_state",
)

builder.add_conditional_edges(
    "validate_state",
    route_after_state_validation,
    {
        "agent": "agent",
        "end": END,
    },
)

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tool": "tool",
        "grounding": "grounding",
        "scope_check": "scope_check",
        "end": END,
    },
)

builder.add_conditional_edges(
    "scope_check",
    route_after_scope_check,
    {
        "agent": "agent",
        "failed": "retrieval_failed",
        "end": END,
    },
)

builder.add_edge(
    "tool",
    "relevance",
)

builder.add_conditional_edges(
    "relevance",
    route_after_relevance,
    {
        "answer": "agent",
        "rewrite": "rewrite",
        "end": "retrieval_failed",
    },
)

builder.add_conditional_edges(
    "grounding",
    route_after_grounding,
    {
        "pass": END,
        "fail": "retrieval_failed",
    },
)

builder.add_edge(
    "rewrite",
    "retry_search",
)

builder.add_edge(
    "retry_search",
    "relevance",
)

builder.add_edge(
    "retrieval_failed",
    END,
)

graph = builder.compile()
