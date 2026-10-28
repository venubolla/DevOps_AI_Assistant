"""Create a source ZIP without local secrets or generated files."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT_NAME = "DevOps_AI_Assistant"
OUTPUT = PROJECT_ROOT.parent / "DevOps_AI_Assistant.zip"
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "__MACOSX"}
EXCLUDED_FILES = {".env", ".DS_Store"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in EXCLUDED_FILES or path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def main() -> None:
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if should_include(path):
                archive.write(path, arcname=f"{ARCHIVE_ROOT_NAME}/{path.relative_to(PROJECT_ROOT)}")
    print(f"Created source archive: {OUTPUT}")
    print("Excluded .env, macOS metadata, bytecode, virtual environments, and caches.")


if __name__ == "__main__":
    main()
