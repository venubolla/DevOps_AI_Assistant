import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.tools import search_knowledge_base


def main() -> None:
    query = "debugging failed database replication cluster procedures"

    result = search_knowledge_base.invoke(
        {
            "query": query,
            "top_k": 5,
            "candidate_k": 20,
        }
    )

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("RAG TOOL RESULT")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    main()
