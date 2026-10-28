from app.evaluation.dataset import load_evaluation_cases
from app.rag.generation import generate_answer
from app.rag.hybrid_search import hybrid_reranked_search


def build_ragas_samples(limit: int | None = None) -> list[dict]:
    """Build RAGAS-compatible sample dictionaries from the evaluation cases."""

    cases = load_evaluation_cases()
    if limit is not None:
        cases = cases[:limit]

    samples = []
    for case in cases:
        results = hybrid_reranked_search(
            query=case.question,
            top_k=5,
            candidate_k=20,
        )
        contexts = [result.content for result in results]
        answer = generate_answer(
            question=case.question,
            results=results,
        )
        samples.append(
            {
                "case_id": case.id,
                "user_input": case.question,
                "response": answer,
                "reference": case.reference_answer,
                "retrieved_contexts": contexts,
            }
        )
    return samples


def build_ragas_dataset(limit: int | None = None):
    """Return an in-memory RAGAS EvaluationDataset using the canonical builder."""

    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            reference=item["reference"],
            retrieved_contexts=item["retrieved_contexts"],
        )
        for item in build_ragas_samples(limit=limit)
    ]
    return EvaluationDataset(samples=samples)
