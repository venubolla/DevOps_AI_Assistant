from langchain_core.tools import tool

from app.rag.hybrid_search import (
    HybridSearchResult,
    hybrid_reranked_search,
)


def run_knowledge_base_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> list[HybridSearchResult]:
    """
    Run the existing RAG V2 hybrid search and reranking pipeline.
    """

    return hybrid_reranked_search(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        service=service,
        environment=environment,
        document_type=document_type,
    )


def format_retrieved_documents(
    results: list[HybridSearchResult],
) -> str:
    """
    Format retrieved documents using the canonical citation format.

    Citation numbering is deterministic:

        [1] -> results[0]
        [2] -> results[1]
        [3] -> results[2]

    The citation number identifies the retrieved result/chunk,
    not a unique source file.
    """

    if not results:
        return "No relevant documents were found in the knowledge base."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        formatted_results.append(
            f"[{index}]\nTitle: {result.title}\nSource: {result.source}\nContent:\n{result.content}"
        )

    return "\n\n---\n\n".join(formatted_results)


@tool
def search_knowledge_base(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    service: str | None = None,
    environment: str | None = None,
    document_type: str | None = None,
) -> str:
    """
    Search the enterprise DevOps knowledge base.

    Use this tool when the user's question requires information
    from the organization's internal documents.

    The tool performs the existing RAG V2 pipeline:
    vector search + keyword search + RRF + cross-encoder reranking.
    """

    results = run_knowledge_base_search(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        service=service,
        environment=environment,
        document_type=document_type,
    )

    return format_retrieved_documents(results)
