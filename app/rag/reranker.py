from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)

    return _reranker


def rerank_results(
    query: str,
    results,
    top_k: int = 5,
):
    if not results:
        return []

    reranker = get_reranker()

    pairs = [(query, result.content) for result in results]

    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result.reranker_score = float(score)

    results.sort(
        key=lambda result: result.reranker_score,
        reverse=True,
    )

    return results[:top_k]
