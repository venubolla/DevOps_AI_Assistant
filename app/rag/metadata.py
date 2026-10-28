from pathlib import Path

import yaml


def parse_front_matter(
    text: str,
) -> tuple[dict, str]:
    """
    Extract YAML front matter from a Markdown document.

    Returns:
        metadata: Parsed YAML metadata.
        content: Document content without front matter.
    """

    text = text.lstrip()

    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)

    if len(parts) != 3:
        return {}, text

    front_matter = parts[1].strip()
    content = parts[2].lstrip()

    metadata = yaml.safe_load(front_matter) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata, content


def extract_metadata(
    file_path: Path,
) -> dict:
    """
    Extract metadata from a Markdown file.
    """

    if file_path.suffix.lower() != ".md":
        return {
            "title": file_path.stem.replace(
                "_",
                " ",
            ).title(),
            "source": str(file_path),
        }

    text = file_path.read_text(encoding="utf-8")

    metadata, _ = parse_front_matter(text)

    metadata.setdefault(
        "title",
        file_path.stem.replace(
            "_",
            " ",
        ).title(),
    )

    metadata.setdefault(
        "source",
        str(file_path),
    )

    return metadata


def clean_document_text(
    file_path: Path,
) -> str:
    """
    Return document text with front matter removed.
    """

    text = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() != ".md":
        return text

    _, content = parse_front_matter(text)

    return content.strip()
