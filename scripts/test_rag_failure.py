import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from unittest.mock import patch

from app.graph import nodes
from app.graph.graph import graph


def main() -> None:
    print("\n" + "=" * 70)
    print("RAG FAILURE TEST")
    print("=" * 70)

    with patch.object(
        nodes,
        "run_knowledge_base_search",
        side_effect=RuntimeError("Simulated PostgreSQL/RAG failure"),
    ):
        result = graph.invoke(
            {
                "question": "What causes CrashLoopBackOff in Kubernetes?",
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("Answer:")
    print(result.get("answer"))

    print("\nTest completed without crashing.")


if __name__ == "__main__":
    main()
