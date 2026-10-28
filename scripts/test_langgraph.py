import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.graph import graph


def main() -> None:
    question = "What causes CrashLoopBackOff in Kubernetes?"

    result = graph.invoke(
        {
            "question": question,
            "top_k": 5,
            "candidate_k": 20,
        }
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for index, document in enumerate(
        result.get("retrieved_documents", []),
        start=1,
    ):
        print(f"\n[{index}] {document.title} ({document.source})")


if __name__ == "__main__":
    main()
