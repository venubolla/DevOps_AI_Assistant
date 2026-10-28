from app.evaluation.models import (
    EvaluationCase,
    EvaluationSummary,
    RetrievalEvaluationResult,
)
from app.rag.hybrid_search import hybrid_reranked_search


def _source_filename(source: str) -> str:
    return source.replace("\\", "/").rsplit("/", 1)[-1]


def evaluate_retrieval_case(
    case: EvaluationCase,
    top_k: int = 5,
) -> RetrievalEvaluationResult:
    results = hybrid_reranked_search(
        query=case.question,
        top_k=top_k,
        candidate_k=max(top_k * 4, 10),
    )

    # Evaluation labels are source documents, not individual chunks. Keep only
    # each source's first occurrence so duplicate chunks cannot distort MRR.
    retrieved_documents = []
    for result in results:
        filename = _source_filename(result.source)
        if filename not in retrieved_documents:
            retrieved_documents.append(filename)

    expected_documents = list(dict.fromkeys(case.expected_documents))
    expected_set = set(expected_documents)
    retrieved_set = set(retrieved_documents)
    relevant_found = expected_set.intersection(retrieved_set)

    hit = bool(relevant_found)
    recall_at_k = len(relevant_found) / len(expected_set) if expected_set else 0.0

    reciprocal_rank = 0.0
    for rank, document in enumerate(retrieved_documents, start=1):
        if document in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalEvaluationResult(
        case_id=case.id,
        question=case.question,
        retrieved_documents=retrieved_documents,
        expected_documents=expected_documents,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        hit=hit,
    )


def evaluate_retrieval(
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> EvaluationSummary:
    results = [evaluate_retrieval_case(case, top_k=top_k) for case in cases]
    total_cases = len(results)
    hits = sum(result.hit for result in results)

    return EvaluationSummary(
        total_cases=total_cases,
        hits=hits,
        hit_rate_at_k=(hits / total_cases if total_cases else 0.0),
        recall_at_k=(
            sum(result.recall_at_k for result in results) / total_cases if total_cases else 0.0
        ),
        mean_reciprocal_rank=(
            sum(result.reciprocal_rank for result in results) / total_cases if total_cases else 0.0
        ),
    )
