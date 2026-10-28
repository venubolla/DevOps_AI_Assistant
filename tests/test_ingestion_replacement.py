"""Integration test: requires the configured PostgreSQL database to be running."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, or_, select

from app.database.connection import SessionLocal
from app.database.models import Document, DocumentChunk
from app.rag import ingestion


class FakeSplitter:
    def split_documents(self, documents):
        return [SimpleNamespace(page_content=doc.page_content) for doc in documents]


@pytest.mark.integration
def test_ingestion_replacement() -> None:
    with TemporaryDirectory(prefix="devops-ai-ingestion-test-") as temp_dir:
        source = Path(temp_dir) / "replace_test.md"
        source_key = source.as_posix()

        def write_doc(body: str) -> None:
            source.write_text(
                "---\ntitle: Ingestion Replacement Test\nservice: test\n---\n" + body,
                encoding="utf-8",
            )

        def fake_load_document(path):
            return [SimpleNamespace(page_content=path.read_text(encoding="utf-8"))]

        try:
            with (
                patch.object(ingestion, "load_document", side_effect=fake_load_document),
                patch.object(
                    ingestion,
                    "embed_texts",
                    side_effect=lambda texts, batch_size: [[0.1] * 384 for _ in texts],
                ),
            ):
                write_doc("Version one content.")
                ingestion.ingest_file(source, FakeSplitter())
                with SessionLocal() as db:
                    first_docs = db.scalars(
                        select(Document).where(Document.source == source_key)
                    ).all()
                    first_chunks = db.scalars(
                        select(DocumentChunk).where(DocumentChunk.source == source_key)
                    ).all()
                assert len(first_docs) == 1
                assert len(first_chunks) == 1
                old_document_id = first_docs[0].document_id

                write_doc("Version two content with changed instructions.")
                ingestion.ingest_file(source, FakeSplitter())

                with SessionLocal() as db:
                    current_docs = db.scalars(
                        select(Document).where(Document.source == source_key)
                    ).all()
                    current_chunks = db.scalars(
                        select(DocumentChunk).where(DocumentChunk.source == source_key)
                    ).all()
                assert len(current_docs) == 1
                assert len(current_chunks) == 1
                assert current_docs[0].document_id != old_document_id
                assert "Version two" in current_chunks[0].content
                assert all(
                    chunk.document_id == current_docs[0].document_id for chunk in current_chunks
                )

                # Re-ingest the original contents. Legacy duplicate versions
                # must be cleaned even when the matching hash already exists.
                write_doc("Version one content.")
                ingestion.ingest_file(source, FakeSplitter())
                with SessionLocal() as db:
                    reverted_docs = db.scalars(
                        select(Document).where(Document.source == source_key)
                    ).all()
                    reverted_chunks = db.scalars(
                        select(DocumentChunk).where(DocumentChunk.source == source_key)
                    ).all()
                assert len(reverted_docs) == 1
                assert len(reverted_chunks) == 1
                assert reverted_docs[0].document_id == old_document_id
                assert "Version one" in reverted_chunks[0].content

            print("INGESTION REPLACEMENT TEST PASSED")
        finally:
            with SessionLocal() as db:
                with db.begin():
                    old_docs = db.scalars(
                        select(Document).where(Document.source == source_key)
                    ).all()
                    ids = [doc.document_id for doc in old_docs]
                    condition = DocumentChunk.source == source_key
                    if ids:
                        condition = or_(condition, DocumentChunk.document_id.in_(ids))
                    db.execute(delete(DocumentChunk).where(condition))
                    db.execute(delete(Document).where(Document.source == source_key))


if __name__ == "__main__":
    test_ingestion_replacement()
