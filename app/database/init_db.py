from sqlalchemy import text

from app.database.connection import Base, engine


def init_db():
    with engine.begin() as connection:
        # Enable pgvector
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create application tables
        Base.metadata.create_all(bind=engine)

        # Full-text search index
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
                ON document_chunks
                USING GIN (
                    to_tsvector(
                        'simple',
                        coalesce(title, '') || ' ' ||
                        coalesce(content, '') || ' ' ||
                        coalesce(source, '')
                    )
                )
                """
            )
        )

        # Approximate nearest-neighbour index; vector_cosine_ops matches
        # the cosine_distance ordering used by app/rag/search.py.
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
