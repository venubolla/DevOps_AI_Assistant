import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.search import search_documents


def main() -> None:
    query = "How do I troubleshoot CrashLoopBackOff?"
    results = search_documents(query, top_k=3)

    for result in results:
        print("\n--------------------")
        print(f"Title: {result.title}")
        print(f"Source: {result.source}")
        print(f"Similarity score: {result.score:.4f}")
        print(result.content[:500])


if __name__ == "__main__":
    main()
