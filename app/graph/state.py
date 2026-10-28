from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.rag.hybrid_search import HybridSearchResult


class RAGGraphState(TypedDict, total=False):
    """
    Shared state for the LangGraph RAG Agent.
    """

    question: str

    top_k: int
    candidate_k: int

    service: str | None
    environment: str | None
    document_type: str | None

    retrieved_documents: list[HybridSearchResult]

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    answer: str

    relevance_passed: bool

    relevance_score: float | None

    relevance_reason: str | None

    retry_count: int

    rewritten_query: str | None

    rewrite_failed: bool

    active_query: str

    grounding_passed: bool
    grounding_score: float | None
    grounding_reason: str | None

    # True when the Agent replied conversationally without retrieval.
    direct_answer: bool

    # scope_check outcome: retrieve, conversational, refused, or unavailable.
    scope_decision: str

    agent_failed: bool
