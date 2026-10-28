from dataclasses import dataclass

from sqlalchemy import select

from app.database.connection import (
    SessionLocal,
)
from app.database.models import (
    DocumentChunk,
)
from app.rag.embeddings import (
    get_embedding_model,
)


@dataclass
class SearchResult:
    chunk_id: int
    document_id: str
    title: str
    content: str
    source: str
    service: str | None
    environment: str | None
    document_type: str | None
    score: float


def search_documents(
    query: str,
    top_k: int = 5,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> list[SearchResult]:

    embedding_model = get_embedding_model()

    query_embedding = embedding_model.embed_query(query)

    with SessionLocal() as db:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        statement = select(
            DocumentChunk,
            distance.label("distance"),
        ).where(DocumentChunk.embedding.is_not(None))

        if service:
            statement = statement.where(DocumentChunk.service == service)

        if environment:
            statement = statement.where(DocumentChunk.environment == environment)

        if document_type:
            statement = statement.where(DocumentChunk.document_type == document_type)

        statement = statement.order_by(distance).limit(top_k)

        rows = db.execute(statement).all()

        results = []

        for chunk, distance_value in rows:
            similarity = 1.0 - float(distance_value)

            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    source=chunk.source,
                    service=chunk.service,
                    environment=chunk.environment,
                    document_type=chunk.document_type,
                    score=similarity,
                )
            )

        return results
