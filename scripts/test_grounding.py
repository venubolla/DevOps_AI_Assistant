import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph.graph import graph
from app.graph.nodes import grounding_node
from app.rag.hybrid_search import hybrid_reranked_search


def print_grounding_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print("GROUNDING RESULT")
    print("=" * 70)

    print(f"Passed : {result.get('grounding_passed')}")

    print(f"Score : {result.get('grounding_score')}")

    print(f"Reason : {result.get('grounding_reason')}")


def positive_grounding_test() -> None:

    question = "What causes CrashLoopBackOff in Kubernetes?"

    result = graph.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )

    print("\n" + "=" * 70)
    print("POSITIVE GROUNDING TEST")
    print("=" * 70)

    print("\nQUESTION")
    print("-" * 70)
    print(question)

    print("\nRELEVANCE")
    print("-" * 70)

    print(f"Passed : {result.get('relevance_passed')}")

    print(f"Score : {result.get('relevance_score')}")

    print(f"Reason : {result.get('relevance_reason')}")

    print("\nGROUNDING")
    print("-" * 70)

    print(f"Passed : {result.get('grounding_passed')}")

    print(f"Score : {result.get('grounding_score')}")

    print(f"Reason : {result.get('grounding_reason')}")

    print("\nFINAL ANSWER")
    print("-" * 70)

    print(
        result.get(
            "answer",
            "No explicit answer state was produced.",
        )
    )


def negative_grounding_test() -> None:

    question = "What causes CrashLoopBackOff in Kubernetes?"

    print("\n" + "=" * 70)
    print("NEGATIVE GROUNDING TEST")
    print("=" * 70)

    print("\nQUESTION")
    print("-" * 70)
    print(question)

    print("\nRETRIEVING DOCUMENTS")
    print("-" * 70)

    documents = hybrid_reranked_search(
        query=question,
        top_k=5,
        candidate_k=20,
    )

    print(f"Retrieved documents : {len(documents)}")

    unsupported_answer = (
        "Kubernetes CrashLoopBackOff is caused by "
        "a hidden Kubernetes licensing service that "
        "automatically shuts down containers after "
        "exactly 17 failed health checks. "
        "The Kubernetes scheduler also requires a "
        "special production license to restart these pods."
    )

    print("\nUNSUPPORTED ANSWER")
    print("-" * 70)
    print(unsupported_answer)

    state = {
        "question": question,
        "answer": unsupported_answer,
        "retrieved_documents": documents,
    }

    result = grounding_node(state)

    print_grounding_result(result)

    if result.get("grounding_passed") is False:
        print("\nRESULT")
        print("-" * 70)
        print("Negative grounding test PASSED.")
        print("The grounding validator rejected the unsupported answer.")
    else:
        print("\nRESULT")
        print("-" * 70)
        print("Negative grounding test FAILED.")
        print("The grounding validator accepted an unsupported answer.")


def main() -> None:

    positive_grounding_test()

    negative_grounding_test()


if __name__ == "__main__":
    main()
