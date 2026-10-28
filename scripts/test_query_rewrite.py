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
            "retry_count": 0,
        }
    )

    print("\n" + "=" * 70)
    print("ORIGINAL QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("RELEVANCE")
    print("=" * 70)

    print(f"Passed : {result.get('relevance_passed')}")

    print(f"Score : {result.get('relevance_score')}")

    print(f"Reason : {result.get('relevance_reason')}")

    print("\n" + "=" * 70)
    print("QUERY REWRITE")
    print("=" * 70)

    print(f"Retry Count : {result.get('retry_count')}")

    print(f"Rewritten Query : {result.get('rewritten_query')}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        result.get(
            "answer",
            "No explicit answer state was produced.",
        )
    )


if __name__ == "__main__":
    main()
