import re

from app.evaluation.llm_judge import (
    evaluate_with_llm,
)
from app.evaluation.models import (
    AnswerEvaluationResult,
    EvaluationCase,
)
from app.evaluation.semantic_similarity import (
    calculate_semantic_similarity,
)
from app.rag.generation import generate_answer
from app.rag.hybrid_search import (
    HybridSearchResult,
)


def calculate_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:

    if not expected_keywords:
        return 1.0

    normalized_answer = answer.lower()

    matched = 0

    for keyword in expected_keywords:
        if keyword.lower() in normalized_answer:
            matched += 1

    return matched / len(expected_keywords)


def extract_citations(
    answer: str,
) -> list[int]:

    matches = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    return [int(value) for value in matches]


def calculate_citation_validity(
    answer: str,
    results: list[HybridSearchResult],
) -> tuple[float, int]:

    citations = extract_citations(answer)

    if not citations:
        return 0.0, 0

    valid_count = 0

    for citation in citations:
        if 1 <= citation <= len(results):
            valid_count += 1

    validity = valid_count / len(citations) if citations else 0.0

    return validity, len(citations)


def evaluate_answer(
    case: EvaluationCase,
    results: list[HybridSearchResult],
) -> AnswerEvaluationResult:

    answer = generate_answer(
        question=case.question,
        results=results,
    )

    keyword_coverage = calculate_keyword_coverage(
        answer,
        case.expected_keywords,
    )

    citation_validity, citation_count = calculate_citation_validity(
        answer,
        results,
    )

    context_term_coverage = calculate_context_term_coverage(
        answer,
        results,
    )

    reference_similarity = calculate_semantic_similarity(
        answer,
        case.reference_answer,
    )

    judge_result = evaluate_with_llm(
        question=case.question,
        reference_answer=case.reference_answer,
        generated_answer=answer,
        results=results,
    )

    return AnswerEvaluationResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        reference_answer=case.reference_answer,
        keyword_coverage=keyword_coverage,
        citation_validity=citation_validity,
        context_term_coverage=context_term_coverage,
        reference_similarity=reference_similarity,
        citation_count=citation_count,
        faithfulness=float(judge_result["faithfulness"]),
        answer_relevance=float(judge_result["answer_relevance"]),
        correctness=float(judge_result["correctness"]),
        citation_correctness=float(judge_result["citation_correctness"]),
        judge_reason=judge_result["reason"],
    )


def build_context_text(
    results: list[HybridSearchResult],
) -> str:

    return "\n".join(result.content.lower() for result in results)


def calculate_context_term_coverage(
    answer: str,
    results: list[HybridSearchResult],
) -> float:

    context = build_context_text(results)

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]+",
            answer,
        )
        if sentence.strip()
    ]

    if not sentences:
        return 0.0

    supported_sentences = 0

    for sentence in sentences:
        words = [
            word
            for word in re.findall(
                r"\b[a-zA-Z0-9_-]+\b",
                sentence.lower(),
            )
            if len(word) > 3
        ]

        if not words:
            continue

        matched_words = sum(word in context for word in words)

        coverage = matched_words / len(words)

        if coverage >= 0.5:
            supported_sentences += 1

    return supported_sentences / len(sentences)
