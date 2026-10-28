import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.retrieval import evaluate_retrieval, evaluate_retrieval_case


def main() -> None:
    cases = load_evaluation_cases()
    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    results = [evaluate_retrieval_case(case, top_k=5) for case in cases]
    for result in results:
        status = "PASS" if result.hit else "FAIL"
        print("-" * 80)
        print(f"{result.case_id}: {status}")
        print(f"Question: {result.question}")
        print(f"Expected documents: {', '.join(result.expected_documents)}")
        print(f"Retrieved documents: {', '.join(result.retrieved_documents)}")
        print(f"Hit@5: {int(result.hit)}")
        print(f"Recall@5: {result.recall_at_k:.3f}")
        print(f"Reciprocal Rank: {result.reciprocal_rank:.3f}")

    summary = evaluate_retrieval(cases, top_k=5)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Cases: {summary.total_cases}")
    print(f"Hits: {summary.hits}")
    print(f"Hit@5: {summary.hit_rate_at_k:.3f}")
    print(f"Recall@5: {summary.recall_at_k:.3f}")
    print(f"MRR: {summary.mean_reciprocal_rank:.3f}")


if __name__ == "__main__":
    main()
