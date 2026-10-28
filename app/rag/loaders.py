from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.rag.metadata import (
    clean_document_text,
)

SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
}


def load_document(
    file_path: Path,
) -> list[Document]:

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))

        return loader.load()

    if suffix in {".md", ".txt"}:
        content = clean_document_text(file_path)

        return [
            Document(
                page_content=content,
                metadata={"source": str(file_path)},
            )
        ]

    raise ValueError(f"Unsupported file type: {suffix}")
