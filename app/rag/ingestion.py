import hashlib
from pathlib import Path

from sqlalchemy import delete, or_, select

from app.database.connection import SessionLocal
from app.database.models import (
    Document,
    DocumentChunk,
)
from app.rag.chunker import (
    get_text_splitter,
)
from app.rag.embeddings import (
    embed_texts,
)
from app.rag.loaders import (
    load_document,
)
from app.rag.metadata import (
    extract_metadata,
)

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")


def calculate_hash(
    content: str,
    metadata: dict,
    source_path: str | None = None,
) -> str:
    """Hash document content, metadata, and (when supplied) source identity.

    Including the source prevents identical files at different paths from
    colliding on the database's unique content_hash constraint.
    """

    metadata_string = str(sorted(metadata.items()))

    value = content + metadata_string + (source_path or "")

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def discover_files() -> list[Path]:

    supported_extensions = {
        ".md",
        ".txt",
        ".pdf",
    }

    return [
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if (path.is_file() and path.suffix.lower() in supported_extensions)
    ]


def ingest_file(
    file_path: Path,
    splitter,
):
    """Ingest a source file and transactionally replace its prior version.

    Embedding generation happens before the database transaction. The old
    document and chunks remain available if loading or embedding fails. Once
    replacement begins, deletion and insertion commit or roll back together.
    """

    print(f"\nProcessing: {file_path}")

    source_key = file_path.as_posix()
    metadata = extract_metadata(file_path)
    raw_documents = load_document(file_path)
    full_content = "\n\n".join(document.page_content for document in raw_documents)

    if not full_content.strip():
        print("Skipping empty document.")
        return

    content_hash = calculate_hash(
        full_content,
        metadata,
        source_path=source_key,
    )

    # Avoid repeated embedding work for an unchanged source file. Also clean
    # any legacy versions left by earlier ingestion runs before returning.
    with SessionLocal() as db:
        existing_version = db.scalar(
            select(Document).where(
                Document.source == source_key,
                Document.content_hash == content_hash,
            )
        )

        if existing_version:
            stale_document_ids = db.scalars(
                select(Document.document_id).where(
                    Document.source == source_key,
                    Document.document_id != existing_version.document_id,
                )
            ).all()
            stale_chunk_filter = (DocumentChunk.source == source_key) & (
                DocumentChunk.document_id != existing_version.document_id
            )
            if stale_document_ids:
                stale_chunk_filter = or_(
                    stale_chunk_filter,
                    DocumentChunk.document_id.in_(stale_document_ids),
                )
            db.execute(delete(DocumentChunk).where(stale_chunk_filter))
            db.execute(
                delete(Document).where(
                    Document.source == source_key,
                    Document.document_id != existing_version.document_id,
                )
            )
            db.commit()
            print("Document is unchanged; stale legacy versions were cleaned.")
            return

    chunks = splitter.split_documents(raw_documents)
    chunk_texts = [chunk.page_content for chunk in chunks]
    print(f"Created {len(chunk_texts)} chunks.")

    print("Generating embeddings...")
    embeddings = embed_texts(chunk_texts, batch_size=32)
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match the number of document chunks.")
    print(f"Generated {len(embeddings)} embeddings.")

    document = Document(
        document_id=f"DOC-{content_hash[:12]}",
        title=metadata.get("title", file_path.stem),
        source=source_key,
        service=metadata.get("service"),
        environment=metadata.get("environment"),
        document_type=metadata.get("document_type"),
        version=metadata.get("version"),
        content_hash=content_hash,
        metadata_json=metadata,
    )

    with SessionLocal() as db:
        # The old version is removed only in the same transaction that adds
        # the replacement. Any exception rolls the entire change back.
        with db.begin():
            old_documents = db.scalars(select(Document).where(Document.source == source_key)).all()
            old_document_ids = [item.document_id for item in old_documents]

            stale_chunk_filter = DocumentChunk.source == source_key
            if old_document_ids:
                stale_chunk_filter = or_(
                    stale_chunk_filter,
                    DocumentChunk.document_id.in_(old_document_ids),
                )

            db.execute(delete(DocumentChunk).where(stale_chunk_filter))
            db.execute(delete(Document).where(Document.source == source_key))

            db.add(document)
            db.add_all(
                [
                    DocumentChunk(
                        document_id=document.document_id,
                        chunk_index=index,
                        title=metadata.get("title", file_path.stem),
                        content=chunk.page_content,
                        source=source_key,
                        service=metadata.get("service"),
                        environment=metadata.get("environment"),
                        document_type=metadata.get("document_type"),
                        embedding=embedding,
                    )
                    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
            )

    print("Document successfully stored (previous version replaced if present).")


def run_ingestion():

    files = discover_files()

    print(f"Found {len(files)} documents.")

    splitter = get_text_splitter()

    for file_path in files:
        ingest_file(
            file_path,
            splitter,
        )


if __name__ == "__main__":
    run_ingestion()
