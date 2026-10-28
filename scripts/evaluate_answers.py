import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.evaluation.answer import (
    evaluate_answer,
)
from app.evaluation.dataset import (
    load_evaluation_cases,
)
from app.rag.hybrid_search import (
    hybrid_reranked_search,
)


def main():

    cases = load_evaluation_cases()

    print("=" * 80)
    print("RAG ANSWER EVALUATION")
    print("=" * 80)

    results = []

    for case in cases:
        print("\n" + "-" * 80)
        print(f"{case.id}")
        print(f"Question: {case.question}")

        retrieved = hybrid_reranked_search(
            query=case.question,
            top_k=5,
            candidate_k=20,
        )

        result = evaluate_answer(
            case=case,
            results=retrieved,
        )

        results.append(result)

        print(f"Keyword Coverage: {result.keyword_coverage:.3f}")

        print(f"Citation Validity: {result.citation_validity:.3f}")

        print(f"Citation Count: {result.citation_count}")

        print("\nAnswer:")
        print(result.answer)

        print(f"Faithfulness: {result.faithfulness:.3f}")

        print(f"Answer Relevance: {result.answer_relevance:.3f}")

        print(f"Correctness: {result.correctness:.3f}")

        print(f"Citation Correctness: {result.citation_correctness:.3f}")

        print(f"Judge Reason: {result.judge_reason}")

    average_keyword_coverage = sum(result.keyword_coverage for result in results) / len(results)

    average_citation_validity = sum(result.citation_validity for result in results) / len(results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Cases: {len(results)}")

    print(f"Average Keyword Coverage: {average_keyword_coverage:.3f}")

    print(f"Average Citation Validity: {average_citation_validity:.3f}")

    print(f"Context Coverage: {result.context_term_coverage:.3f}")

    average_context_coverage = sum(result.context_term_coverage for result in results) / len(
        results
    )

    print(f"Average Context Coverage: {average_context_coverage:.3f}")

    print(f"Reference Similarity: {result.reference_similarity:.3f}")

    average_reference_similarity = sum(result.reference_similarity for result in results) / len(
        results
    )

    print(f"Average Reference Similarity: {average_reference_similarity:.3f}")


if __name__ == "__main__":
    main()
