import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.graph import graph


def run_test(name: str, state: dict) -> None:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    result = graph.invoke(state)

    print("Answer:")
    print(result.get("answer"))

    print("Question:")
    print(repr(result.get("question")))


def main() -> None:

    run_test(
        "VALID QUESTION",
        {
            "question": "What causes CrashLoopBackOff in Kubernetes?",
            "top_k": 5,
            "candidate_k": 20,
        },
    )

    run_test(
        "EMPTY QUESTION",
        {
            "question": "",
        },
    )

    run_test(
        "WHITESPACE QUESTION",
        {
            "question": "   ",
        },
    )


if __name__ == "__main__":
    main()
