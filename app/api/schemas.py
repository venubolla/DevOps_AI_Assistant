from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RAGSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    service: str | None = None

    environment: str | None = None

    document_type: str | None = None


class SearchResultResponse(BaseModel):
    chunk_id: int
    document_id: str
    title: str
    content: str
    source: str
    service: str | None
    environment: str | None
    document_type: str | None
    score: float


class RAGSearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]


class RAGAskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    service: str | None = None

    environment: str | None = None

    document_type: str | None = None


class SourceResponse(BaseModel):
    citation: str
    title: str
    source: str
    score: float


class RAGAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


class ChatMessage(BaseModel):
    """A prior turn supplied by the client so the agent can resolve follow-ups."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class AgentAskRequest(BaseModel):
    """Request schema for the LangGraph-backed agent endpoint."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    service: str | None = None
    environment: str | None = None
    document_type: str | None = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value
