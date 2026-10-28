from dataclasses import dataclass

from sqlalchemy import func, select

from app.database.connection import SessionLocal
from app.database.models import DocumentChunk


@dataclass
class KeywordSearchResult:
    chunk_id: int
    document_id: str
    title: str
    content: str
    source: str
    service: str | None
    environment: str | None
    document_type: str | None
    score: float


def keyword_search(
    query: str,
    top_k: int = 10,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> list[KeywordSearchResult]:

    search_vector = func.to_tsvector(
        "simple",
        (
            func.coalesce(DocumentChunk.title, "")
            + " "
            + func.coalesce(DocumentChunk.content, "")
            + " "
            + func.coalesce(DocumentChunk.source, "")
        ),
    )

    search_query = func.plainto_tsquery(
        "simple",
        query,
    )

    rank = func.ts_rank_cd(
        search_vector,
        search_query,
    )

    statement = select(
        DocumentChunk,
        rank.label("rank"),
    ).where(search_vector.bool_op("@@")(search_query))

    if service:
        statement = statement.where(DocumentChunk.service == service)

    if environment:
        statement = statement.where(DocumentChunk.environment == environment)

    if document_type:
        statement = statement.where(DocumentChunk.document_type == document_type)

    statement = statement.order_by(rank.desc()).limit(top_k)

    with SessionLocal() as db:
        rows = db.execute(statement).all()

        results = []

        for chunk, rank_value in rows:
            results.append(
                KeywordSearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    source=chunk.source,
                    service=chunk.service,
                    environment=chunk.environment,
                    document_type=chunk.document_type,
                    score=float(rank_value),
                )
            )

        return results
