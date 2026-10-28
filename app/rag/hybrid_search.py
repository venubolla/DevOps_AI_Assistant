from dataclasses import dataclass

from app.rag.keyword_search import (
    KeywordSearchResult,
    keyword_search,
)
from app.rag.reranker import rerank_results
from app.rag.search import SearchResult, search_documents


@dataclass
class HybridSearchResult:
    chunk_id: int
    document_id: str
    title: str
    content: str
    source: str
    service: str | None
    environment: str | None
    document_type: str | None
    vector_score: float | None
    keyword_score: float | None
    hybrid_score: float
    reranker_score: float | None = None


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    keyword_results: list[KeywordSearchResult],
    rrf_k: int = 60,
) -> list[HybridSearchResult]:

    combined = {}

    for rank, result in enumerate(vector_results, start=1):
        if result.chunk_id not in combined:
            combined[result.chunk_id] = {
                "result": result,
                "vector_score": result.score,
                "keyword_score": None,
                "hybrid_score": 0.0,
            }

        combined[result.chunk_id]["hybrid_score"] += 1.0 / (rrf_k + rank)

    for rank, result in enumerate(keyword_results, start=1):
        if result.chunk_id not in combined:
            combined[result.chunk_id] = {
                "result": result,
                "vector_score": None,
                "keyword_score": result.score,
                "hybrid_score": 0.0,
            }

        combined[result.chunk_id]["hybrid_score"] += 1.0 / (rrf_k + rank)

        combined[result.chunk_id]["keyword_score"] = result.score

    results = []

    for item in combined.values():
        result = item["result"]

        results.append(
            HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                source=result.source,
                service=result.service,
                environment=result.environment,
                document_type=result.document_type,
                vector_score=item["vector_score"],
                keyword_score=item["keyword_score"],
                hybrid_score=item["hybrid_score"],
            )
        )

    results.sort(
        key=lambda result: result.hybrid_score,
        reverse=True,
    )

    return results


def hybrid_search(
    query: str,
    top_k: int = 10,
    candidate_k: int = 20,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> list[HybridSearchResult]:

    vector_results = search_documents(
        query=query,
        top_k=candidate_k,
        service=service,
        environment=environment,
        document_type=document_type,
    )

    keyword_results = keyword_search(
        query=query,
        top_k=candidate_k,
        service=service,
        environment=environment,
        document_type=document_type,
    )

    fused_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    return fused_results[:top_k]


def hybrid_reranked_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> list[HybridSearchResult]:

    candidates = hybrid_search(
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k,
        service=service,
        environment=environment,
        document_type=document_type,
    )

    reranked_results = rerank_results(
        query=query,
        results=candidates,
        top_k=top_k,
    )

    return reranked_results
