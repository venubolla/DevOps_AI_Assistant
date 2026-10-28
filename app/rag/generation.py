from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.hybrid_search import HybridSearchResult
from app.rag.llm import get_llm

SYSTEM_PROMPT = """
You are the DevOps AI Assistant.

Your job is to answer questions using only
the provided enterprise knowledge context.

Rules:

1. Use the supplied context as your primary
   source of truth.

2. Do not invent deployment IDs, incident IDs,
   commands, metrics, configurations, or
   operational facts.

3. If the context does not contain enough
   information to answer the question, say:

   "I don't have enough information in the
   knowledge base to answer this confidently."

4. Explain the answer clearly.

5. When using information from a context source,
   cite it using [1], [2], [3], etc.

6. Do not create citations for information that
   does not exist in the provided context.

7. Never claim that an operational action was
   performed. This RAG system is read-only.
"""


def build_context(
    results: list[HybridSearchResult],
) -> str:

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
[{index}]
Title: {result.title}
Source: {result.source}
Service: {result.service}
Environment: {result.environment}
Document Type: {result.document_type}

Content:
{result.content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    results: list[HybridSearchResult],
) -> str:

    context = build_context(results)

    user_prompt = f"""
Answer the following question using
the supplied context.

QUESTION:
{question}

CONTEXT:
{context}
"""

    llm = get_llm()

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    return response.content
