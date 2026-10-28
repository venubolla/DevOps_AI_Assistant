import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.evaluation.answer import (
    calculate_citation_validity,
    calculate_context_term_coverage,
    calculate_keyword_coverage,
    evaluate_answer,
)
from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.llm_judge import (
    evaluate_with_llm,
)
from app.evaluation.semantic_similarity import (
    calculate_semantic_similarity,
)
from app.graph.graph import graph
from app.rag.hybrid_search import (
    hybrid_reranked_search,
)


def evaluate_rag_v2(case):
    """
    Run one evaluation case through the existing RAG V2 pipeline.
    """

    results = hybrid_reranked_search(
        query=case.question,
        top_k=5,
        candidate_k=20,
    )

    evaluation = evaluate_answer(
        case,
        results,
    )

    return {
        "answer": evaluation.answer,
        "retrieved_documents": results,
        "keyword_coverage": evaluation.keyword_coverage,
        "citation_validity": evaluation.citation_validity,
        "context_term_coverage": evaluation.context_term_coverage,
        "reference_similarity": evaluation.reference_similarity,
        "citation_count": evaluation.citation_count,
        "faithfulness": evaluation.faithfulness,
        "answer_relevance": evaluation.answer_relevance,
        "correctness": evaluation.correctness,
        "citation_correctness": evaluation.citation_correctness,
        "judge_reason": evaluation.judge_reason,
    }


def evaluate_langgraph(case):
    """
    Run one evaluation case through the LangGraph RAG workflow.
    """

    result = graph.invoke(
        {
            "question": case.question,
            "top_k": 5,
            "candidate_k": 20,
            "retry_count": 0,
        }
    )

    retrieved_documents = result.get(
        "retrieved_documents",
        [],
    )

    answer = result.get(
        "answer",
        "",
    )

    keyword_coverage = calculate_keyword_coverage(
        answer,
        case.expected_keywords,
    )

    citation_validity, citation_count = calculate_citation_validity(
        answer,
        retrieved_documents,
    )

    context_term_coverage = calculate_context_term_coverage(
        answer,
        retrieved_documents,
    )

    reference_similarity = calculate_semantic_similarity(
        answer,
        case.reference_answer,
    )

    judge_result = evaluate_with_llm(
        question=case.question,
        reference_answer=case.reference_answer,
        generated_answer=answer,
        results=retrieved_documents,
    )

    return {
        "answer": answer,
        "retrieved_documents": retrieved_documents,
        "keyword_coverage": keyword_coverage,
        "citation_validity": citation_validity,
        "context_term_coverage": context_term_coverage,
        "reference_similarity": reference_similarity,
        "citation_count": citation_count,
        "faithfulness": float(judge_result["faithfulness"]),
        "answer_relevance": float(judge_result["answer_relevance"]),
        "correctness": float(judge_result["correctness"]),
        "citation_correctness": float(judge_result["citation_correctness"]),
        "judge_reason": judge_result["reason"],
        "grounding_passed": result.get(
            "grounding_passed",
            False,
        ),
        "grounding_score": result.get(
            "grounding_score",
        ),
        "grounding_reason": result.get(
            "grounding_reason",
        ),
        "retry_count": result.get(
            "retry_count",
            0,
        ),
    }


def print_documents(documents):
    """
    Print retrieved document sources.
    """

    for document in documents:
        print(f"  - {document.source}")


def print_metrics(label, result):
    """
    Print comparable evaluation metrics.
    """

    print(f"\n{label}")

    print(f"  Keyword Coverage      : {result['keyword_coverage']:.3f}")

    print(f"  Citation Validity     : {result['citation_validity']:.3f}")

    print(f"  Context Coverage      : {result['context_term_coverage']:.3f}")

    print(f"  Reference Similarity  : {result['reference_similarity']:.3f}")

    print(f"  Faithfulness          : {result['faithfulness']:.3f}")

    print(f"  Answer Relevance      : {result['answer_relevance']:.3f}")

    print(f"  Correctness           : {result['correctness']:.3f}")

    print(f"  Citation Correctness : {result['citation_correctness']:.3f}")


def calculate_average(results, key):
    """
    Calculate the average value of one metric.
    """

    if not results:
        return 0.0

    return sum(result[key] for result in results) / len(results)


def main():

    cases = load_evaluation_cases()

    rag_v2_results = []
    langgraph_results = []

    print("=" * 80)
    print("LANGGRAPH vs RAG V2 COMPARISON")
    print("=" * 80)

    for case in cases:
        print("\n" + "=" * 80)
        print(f"{case.id}")
        print("=" * 80)

        print(f"\nQuestion: {case.question}")

        # ------------------------------------------------------------------
        # RAG V2
        # ------------------------------------------------------------------

        print("\nRunning RAG V2...")

        rag_v2_result = evaluate_rag_v2(case)

        rag_v2_results.append(rag_v2_result)

        # ------------------------------------------------------------------
        # LangGraph
        # ------------------------------------------------------------------

        print("\nRunning LangGraph...")

        langgraph_result = evaluate_langgraph(case)

        langgraph_results.append(langgraph_result)

        # ------------------------------------------------------------------
        # Retrieved documents
        # ------------------------------------------------------------------

        print("\n" + "-" * 80)
        print("RAG V2 RETRIEVED DOCUMENTS")
        print("-" * 80)

        print_documents(rag_v2_result["retrieved_documents"])

        print("\n" + "-" * 80)
        print("LANGGRAPH RETRIEVED DOCUMENTS")
        print("-" * 80)

        print_documents(langgraph_result["retrieved_documents"])

        # ------------------------------------------------------------------
        # Answers
        # ------------------------------------------------------------------

        print("\n" + "-" * 80)
        print("RAG V2 ANSWER")
        print("-" * 80)

        print(rag_v2_result["answer"])

        print("\n" + "-" * 80)
        print("LANGGRAPH ANSWER")
        print("-" * 80)

        print(langgraph_result["answer"])

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

        print("\n" + "-" * 80)
        print("RAG V2 METRICS")
        print("-" * 80)

        print_metrics(
            "RAG V2",
            rag_v2_result,
        )

        print("\n" + "-" * 80)
        print("LANGGRAPH METRICS")
        print("-" * 80)

        print_metrics(
            "LangGraph",
            langgraph_result,
        )

        # ------------------------------------------------------------------
        # LangGraph validation
        # ------------------------------------------------------------------

        print("\n" + "-" * 80)
        print("LANGGRAPH VALIDATION")
        print("-" * 80)

        print(f"  Grounding Passed : {langgraph_result['grounding_passed']}")

        print(f"  Grounding Score  : {langgraph_result['grounding_score']}")

        print(f"  Retry Count      : {langgraph_result['retry_count']}")

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP-14 SUMMARY")
    print("=" * 80)

    print(f"\nEvaluation Cases: {len(cases)}")

    metrics = [
        (
            "keyword_coverage",
            "Keyword Coverage",
        ),
        (
            "citation_validity",
            "Citation Validity",
        ),
        (
            "context_term_coverage",
            "Context Coverage",
        ),
        (
            "reference_similarity",
            "Reference Similarity",
        ),
        (
            "faithfulness",
            "Faithfulness",
        ),
        (
            "answer_relevance",
            "Answer Relevance",
        ),
        (
            "correctness",
            "Correctness",
        ),
        (
            "citation_correctness",
            "Citation Correctness",
        ),
    ]

    print("\nMetric Comparison")
    print("-" * 80)

    print(f"{'Metric':<25}{'RAG V2':>12}{'LangGraph':>15}{'Difference':>15}")

    print("-" * 80)

    for key, label in metrics:
        rag_v2_average = calculate_average(
            rag_v2_results,
            key,
        )

        langgraph_average = calculate_average(
            langgraph_results,
            key,
        )

        difference = langgraph_average - rag_v2_average

        print(f"{label:<25}{rag_v2_average:>12.3f}{langgraph_average:>15.3f}{difference:>15.3f}")

    # ----------------------------------------------------------------------
    # LangGraph-specific metrics
    # ----------------------------------------------------------------------

    grounding_passes = sum(1 for result in langgraph_results if result["grounding_passed"])

    average_grounding_score = sum(
        result["grounding_score"] or 0.0 for result in langgraph_results
    ) / len(langgraph_results)

    average_retry_count = sum(result["retry_count"] for result in langgraph_results) / len(
        langgraph_results
    )

    print("\n" + "-" * 80)
    print("LANGGRAPH-SPECIFIC METRICS")
    print("-" * 80)

    print(f"Grounding Pass Rate     : {grounding_passes / len(cases):.3f}")

    print(f"Average Grounding Score : {average_grounding_score:.3f}")

    print(f"Average Retry Count     : {average_retry_count:.3f}")


if __name__ == "__main__":
    main()
