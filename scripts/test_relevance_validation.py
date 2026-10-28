import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.graph import graph


def main() -> None:

    question = (
        "What are the recommended procedures for debugging a failed database replication cluster?"
    )

    result = graph.invoke(
        {
            "question": question,
        }
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("RETRIEVAL VALIDATION")
    print("=" * 70)

    print(f"Relevance Passed : {result.get('relevance_passed')}")

    print(f"Best Relevance Score : {result.get('relevance_score')}")

    print(f"Reason : {result.get('relevance_reason')}")

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for index, document in enumerate(
        result.get("retrieved_documents", []),
        start=1,
    ):
        print(f"\n[{index}] {document.title}")

        print(f"Reranker Score : {document.reranker_score}")

        print(f"Source : {document.source}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    messages = result.get("messages", [])

    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
