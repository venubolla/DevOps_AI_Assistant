from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(500),
    )

    source: Mapped[str] = mapped_column(
        String(1000),
    )

    service: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    environment: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    document_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
    )

    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
    )

    title: Mapped[str] = mapped_column(
        String(500),
    )

    content: Mapped[str] = mapped_column(
        Text,
    )

    source: Mapped[str] = mapped_column(
        String(1000),
    )

    service: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    environment: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    document_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True,
    )
