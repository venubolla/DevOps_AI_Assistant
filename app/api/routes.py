import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.schemas import (
    AgentAskRequest,
    RAGAskRequest,
    RAGAskResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    SearchResultResponse,
    SourceResponse,
)
from app.database.connection import get_db
from app.graph.graph import graph
from app.rag.generation import generate_answer
from app.rag.hybrid_search import hybrid_reranked_search
from app.rag.search import (
    search_documents,
)

logger = logging.getLogger(__name__)

router = APIRouter()

AGENT_FALLBACK_ANSWER = (
    "I could not generate a reliable response from the knowledge base. "
    "Please try again later or refine your question."
)

AGENT_UNAVAILABLE_MESSAGE = "The agent is temporarily unable to answer this request."

# User-facing labels for the graph nodes streamed to the chat UI.
NODE_STATUS = {
    "validate_state": "Validating the request",
    "agent": "Thinking",
    "tool": "Searching the knowledge base",
    "scope_check": "Checking the knowledge base for relevant documents",
    "relevance": "Checking retrieval relevance",
    "rewrite": "Rewriting the query",
    "retry_search": "Retrying the search",
    "grounding": "Verifying the answer against the sources",
    "retrieval_failed": "Preparing a safe response",
}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.get("/info")
async def application_info():
    return {
        "application": "DevOps AI Assistant",
        "description": "AI-powered DevOps Operations Assistant",
    }


@router.get("/health/database")
def database_health(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1"))
        result.scalar()

        return {"database": "healthy"}

    except Exception:
        # Connection errors carry the host, database and user, so log only.
        logger.exception("Database health check failed.")

        return {"database": "unhealthy"}


@router.post(
    "/api/v1/rag/search",
    response_model=RAGSearchResponse,
)
def rag_search(
    request: RAGSearchRequest,
):

    results = search_documents(
        query=request.query,
        top_k=request.top_k,
        service=request.service,
        environment=request.environment,
        document_type=request.document_type,
    )

    return RAGSearchResponse(
        query=request.query,
        results=[
            SearchResultResponse(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                source=result.source,
                service=result.service,
                environment=result.environment,
                document_type=result.document_type,
                score=result.score,
            )
            for result in results
        ],
    )


@router.post(
    "/api/v1/rag/ask",
    response_model=RAGAskResponse,
)
def rag_ask(request: RAGAskRequest):

    results = hybrid_reranked_search(
        query=request.question,
        top_k=request.top_k,
        candidate_k=max(request.top_k * 4, 10),
        service=request.service,
        environment=request.environment,
        document_type=request.document_type,
    )

    if not results:
        return RAGAskResponse(
            question=request.question,
            answer=(
                "I don't have enough information in the knowledge base to answer this confidently."
            ),
            sources=[],
        )

    answer = generate_answer(
        question=request.question,
        results=results,
    )

    sources = [
        SourceResponse(
            citation=f"[{index}]",
            title=result.title,
            source=result.source,
            score=result.reranker_score
            if result.reranker_score is not None
            else result.hybrid_score,
        )
        for index, result in enumerate(results, start=1)
    ]

    return RAGAskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
    )


def build_agent_state(request: AgentAskRequest) -> dict:
    """Seed the graph with prior turns so follow-up questions keep their context."""

    messages = [
        HumanMessage(content=turn.content)
        if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in request.history
    ]

    messages.append(HumanMessage(content=request.question))

    return {
        "question": request.question,
        "top_k": request.top_k,
        "candidate_k": max(request.top_k * 4, 10),
        "service": request.service,
        "environment": request.environment,
        "document_type": request.document_type,
        "messages": messages,
    }


def build_agent_response(question: str, result: dict) -> RAGAskResponse:
    answer = result.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        answer = AGENT_FALLBACK_ANSWER

    retrieved_documents = result.get("retrieved_documents", []) or []

    sources = [
        SourceResponse(
            citation=f"[{index}]",
            title=document.title,
            source=document.source,
            score=(
                document.reranker_score
                if document.reranker_score is not None
                else document.hybrid_score
            ),
        )
        for index, document in enumerate(retrieved_documents, start=1)
    ]

    return RAGAskResponse(
        question=question,
        answer=answer.strip(),
        sources=sources,
    )


@router.post(
    "/api/v1/agent/ask",
    response_model=RAGAskResponse,
)
def agent_ask(request: AgentAskRequest):
    """Answer a question using the existing LangGraph RAG agent."""

    try:
        result = graph.invoke(build_agent_state(request))
    except Exception as exc:
        # Do not expose internal model, database, or provider exceptions.
        raise HTTPException(
            status_code=503,
            detail=AGENT_UNAVAILABLE_MESSAGE,
        ) from exc

    return build_agent_response(request.question, result)


@router.post("/api/v1/agent/ask/stream")
def agent_ask_stream(request: AgentAskRequest):
    """Stream graph progress as NDJSON, then the validated answer.

    The answer is emitted only after grounding validation, so partial text is
    never shown for a response the graph later rejects.
    """

    def event_stream():
        final_state: dict = {}

        try:
            for update in graph.stream(build_agent_state(request), stream_mode="updates"):
                for node_name, node_state in update.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)

                    label = NODE_STATUS.get(node_name)

                    if label:
                        yield json.dumps({"type": "status", "stage": label}) + "\n"

        except Exception:
            logger.exception("Agent streaming failed.")

            yield json.dumps({"type": "error", "message": AGENT_UNAVAILABLE_MESSAGE}) + "\n"

            return

        response = build_agent_response(request.question, final_state)

        yield json.dumps({"type": "result", **response.model_dump()}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
