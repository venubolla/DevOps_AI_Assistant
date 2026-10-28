import json
import math
import uuid

import ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.graph.state import RAGGraphState
from app.graph.tools import (
    format_retrieved_documents,
    run_knowledge_base_search,
    search_knowledge_base,
)
from app.rag.llm import get_fast_llm, get_llm

SYSTEM_PROMPT = """
You are the DevOps AI Assistant.

You answer questions about enterprise DevOps systems, infrastructure,
Kubernetes, deployments, incidents, services, and internal documentation.

You have access to an internal enterprise knowledge-base search tool.

For questions involving:
- Kubernetes
- databases
- database replication
- incidents
- runbooks
- deployments
- services
- architecture
- infrastructure
- troubleshooting

you MUST use the search_knowledge_base tool before answering.

Use the retrieved documents as the primary source of truth.

Do not invent enterprise-specific facts.

After receiving tool results, answer the user's question using the
retrieved information.

Scope:
- You only help with DevOps, infrastructure, Kubernetes, CI/CD,
  incidents, and internal documentation.
- You may reply briefly to greetings, thanks, and questions about
  what you can do.
- For anything else, reply exactly:
  "I can only help with DevOps-related questions."
"""

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with DevOps-related questions grounded in the "
    "knowledge base. Try asking about Kubernetes, deployments, "
    "incidents, or runbooks."
)

SCOPE_CLASSIFIER_PROMPT = """
Classify the user message for a DevOps assistant.
Reply with exactly one word:

SMALLTALK - a greeting, thanks, or a question about what the assistant can do
DEVOPS - about software infrastructure, Kubernetes, CI/CD, deployments,
         incidents, cloud, monitoring, or operations
OTHER - anything else

Message: {question}
"""


TOOLS = {
    search_knowledge_base.name: search_knowledge_base,
}

MAX_RETRIES = 1

AGENT_FAILURE_MESSAGE = (
    "I could not generate a reliable response because "
    "the language model encountered an error. "
    "Please try again later."
)


def validate_graph_state(
    state: RAGGraphState,
) -> dict:
    """
    Validate and initialize the minimum state required
    by the LangGraph workflow.
    """

    question = state.get("question")

    if not isinstance(question, str):
        return {
            "answer": (
                "I could not process the request because the question is missing or invalid."
            ),
            "relevance_passed": False,
            "grounding_passed": False,
        }

    question = question.strip()

    if not question:
        return {
            "question": "",
            "answer": ("I could not process an empty question. Please provide a question."),
            "relevance_passed": False,
            "grounding_passed": False,
        }

    return {
        "question": question,
        "top_k": state.get("top_k", 5),
        "candidate_k": state.get("candidate_k", 20),
        "service": state.get("service"),
        "environment": state.get("environment"),
        "document_type": state.get("document_type"),
        "retrieved_documents": state.get(
            "retrieved_documents",
            [],
        ),
        "retry_count": state.get(
            "retry_count",
            0,
        ),
        "active_query": state.get(
            "active_query",
            question,
        ),
    }


def agent_node(state: RAGGraphState) -> dict:
    """
    Execute the Agent's retrieval or answer-generation step.

    LLM failures are converted into a controlled graph state
    instead of propagating as uncaught exceptions.
    """

    # ---------------------------------------------------------
    # Step 1: Initialize the LLM safely
    # ---------------------------------------------------------

    try:
        llm = get_llm()

    except Exception:
        return {
            "answer": AGENT_FAILURE_MESSAGE,
            "agent_failed": True,
        }

    messages = state.get("messages", [])

    if not messages:
        question = state.get("question", "")

        if not isinstance(question, str) or not question.strip():
            return {
                "answer": (
                    "I could not process the request because the question is missing or invalid."
                ),
                "agent_failed": True,
            }

        messages = [HumanMessage(content=question.strip())]

    system_message = SystemMessage(content=SYSTEM_PROMPT)

    last_message = messages[-1]

    # ---------------------------------------------------------
    # Step 2: First Agent call — request retrieval
    # ---------------------------------------------------------

    if not isinstance(last_message, ToolMessage):
        try:
            llm_with_tools = llm.bind_tools([search_knowledge_base])

            response = llm_with_tools.invoke([system_message, *messages])

        except Exception:
            return {
                "answer": AGENT_FAILURE_MESSAGE,
                "agent_failed": True,
            }

        if getattr(response, "tool_calls", None):
            return {
                "messages": [response],
                "direct_answer": False,
                "agent_failed": False,
            }

        # No tool was requested, so the model answered conversationally
        # (a greeting or a capability question). Keep that answer instead of
        # discarding it; the system prompt still forces retrieval for
        # enterprise topics, and grounding only applies to retrieved claims.
        content = getattr(response, "content", None)

        if not isinstance(content, str) or not content.strip():
            return {
                "answer": AGENT_FAILURE_MESSAGE,
                "agent_failed": True,
            }

        return {
            "messages": [response],
            "answer": content.strip(),
            "direct_answer": True,
            "agent_failed": False,
        }

    # ---------------------------------------------------------
    # Step 3: Second Agent call — generate the answer
    # ---------------------------------------------------------

    retry_context = SystemMessage(
        content=(
            "A retrieval attempt has been completed. "
            "Use the retrieved documents provided in the tool result "
            "as the primary source of truth. "
            "Answer the original user question directly. "
            "Do not perform another search. "
            "Use numeric citations such as [1], [2], [3] "
            "to cite retrieved documents. "
            "Citation numbers must correspond to the document order "
            "in the retrieved context. "
            "Do not create markdown links to files. "
            "Do not cite file paths, filenames, or document titles "
            "as citation markers. "
            "Only use citation numbers that correspond to retrieved documents."
        )
    )

    try:
        response = llm.invoke(
            [
                system_message,
                retry_context,
                *messages,
            ]
        )

        answer = response.content

        if not isinstance(answer, str) or not answer.strip():
            return {
                "answer": AGENT_FAILURE_MESSAGE,
                "agent_failed": True,
            }

    except Exception:
        return {
            "answer": AGENT_FAILURE_MESSAGE,
            "agent_failed": True,
        }

    return {
        "messages": [response],
        "answer": answer.strip(),
        "direct_answer": False,
        "agent_failed": False,
    }


def classify_scope(question: str) -> str:
    """Return SMALLTALK, DEVOPS or OTHER; any failure counts as OTHER."""

    try:
        response = get_fast_llm().invoke(
            [HumanMessage(content=SCOPE_CLASSIFIER_PROMPT.format(question=question))]
        )
        label = response.content.strip().upper()
    except Exception:
        return "OTHER"

    for candidate in ("SMALLTALK", "DEVOPS", "OTHER"):
        if label.startswith(candidate):
            return candidate

    return "OTHER"


def scope_check_node(state: RAGGraphState) -> dict:
    """
    Decide whether a tool-less Agent reply may be returned as-is.

    Qwen does not always request the search tool, so an unretrieved answer is
    only trusted when the knowledge base has nothing relevant to say. If
    relevant documents exist, they are injected as a tool result and the Agent
    is sent back to produce a cited, grounded answer instead.

    With nothing relevant, only small talk keeps the model's reply. DevOps
    questions the knowledge base cannot answer get the standard not-found
    response, and everything else is refused.
    """

    question = state.get("question", "")

    try:
        results = run_knowledge_base_search(
            query=question,
            top_k=state.get("top_k", 5),
            candidate_k=state.get("candidate_k", 20),
            service=state.get("service"),
            environment=state.get("environment"),
            document_type=state.get("document_type"),
        )
    except Exception:
        # Without retrieval the scope of the reply cannot be verified.
        return {"scope_decision": "unavailable"}

    scores = [result.reranker_score for result in results if result.reranker_score is not None]

    if not scores or max(scores) < settings.relevance_threshold:
        label = classify_scope(question)

        if label == "SMALLTALK":
            return {"scope_decision": "conversational"}

        if label == "DEVOPS":
            # Routed to retrieval_failed for the standard not-found response.
            return {"scope_decision": "unavailable"}

        return {
            "answer": OUT_OF_SCOPE_MESSAGE,
            "scope_decision": "refused",
        }

    tool_call_id = f"scope-check-{uuid.uuid4().hex}"

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {
                    "query": question,
                    "top_k": state.get("top_k", 5),
                    "candidate_k": state.get("candidate_k", 20),
                    "service": state.get("service"),
                    "environment": state.get("environment"),
                    "document_type": state.get("document_type"),
                },
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )

    tool_message = ToolMessage(
        content=format_retrieved_documents(results),
        tool_call_id=tool_call_id,
        name="search_knowledge_base",
    )

    return {
        "messages": [tool_call_message, tool_message],
        "retrieved_documents": results,
        "active_query": question,
        "direct_answer": False,
        "answer": "",
        "scope_decision": "retrieve",
    }


def tool_node(state: RAGGraphState) -> dict:
    """
    Execute the tools requested by the Agent.

    The structured RAG results are also stored in graph state
    so downstream validation nodes can inspect reranker scores.
    """

    active_query = state.get(
        "active_query",
        state["question"],
    )

    messages = state.get("messages", [])

    if not messages:
        return {
            "messages": [],
            "retrieved_documents": [],
        }

    last_message = messages[-1]

    tool_messages = []

    retrieved_documents = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # print("\n" + "=" * 70)
        # print("AGENT TOOL CALL")
        # print("=" * 70)
        # print(f"Tool : {tool_name}")
        # print(f"Args : {tool_args}")

        if tool_name != search_knowledge_base.name:
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool requested: {tool_name}",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
            continue

        # Request-level graph settings take precedence for limits and filters.
        # The model still supplies the search query, while API callers can
        # reliably constrain retrieval through graph state.
        search_args = {
            "query": tool_args.get("query") or active_query,
            "top_k": state.get("top_k", tool_args.get("top_k", 5)),
            "candidate_k": state.get("candidate_k", tool_args.get("candidate_k", 20)),
            "service": state.get("service") or tool_args.get("service"),
            "environment": state.get("environment") or tool_args.get("environment"),
            "document_type": state.get("document_type") or tool_args.get("document_type"),
        }

        try:
            results = run_knowledge_base_search(**search_args)
        except Exception:
            results = []

            tool_result = "The knowledge base search failed. No reliable documents were retrieved."

            tool_messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

            continue

        retrieved_documents.extend(results)

        tool_result = format_retrieved_documents(results)

        tool_messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        "retrieved_documents": retrieved_documents,
    }


def relevance_node(state: RAGGraphState) -> dict:
    """
    Validate whether the retrieved documents are relevant enough
    to continue.

    The strongest existing cross-encoder reranker score is used
    as the initial relevance signal.
    """

    results = state.get(
        "retrieved_documents",
        [],
    )

    if not results:
        return {
            "relevance_passed": False,
            "relevance_score": None,
            "relevance_reason": ("No documents were retrieved."),
        }

    scores = [result.reranker_score for result in results if result.reranker_score is not None]

    if not scores:
        return {
            "relevance_passed": False,
            "relevance_score": None,
            "relevance_reason": ("Retrieved documents do not have reranker scores."),
        }

    best_score = max(scores)

    # Read at call time so .env changes and tests take effect without reimport.
    passed = best_score >= settings.relevance_threshold

    if passed:
        reason = "At least one retrieved document passed the relevance threshold."
    else:
        reason = "The retrieved documents did not pass the relevance threshold."

    return {
        "relevance_passed": passed,
        "relevance_score": best_score,
        "relevance_reason": reason,
    }


def rewrite_query_node(
    state: RAGGraphState,
) -> dict:
    """
    Rewrite a failed retrieval query into a more
    retrieval-friendly enterprise DevOps query.

    If query rewriting fails, use the original question
    as a fallback and increment the retry counter.
    """

    # ---------------------------------------------------------
    # Step 1: Obtain and validate the original question
    # ---------------------------------------------------------

    question = state.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return {
            "rewritten_query": "",
            "rewrite_failed": True,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    question = question.strip()

    # ---------------------------------------------------------
    # Step 2: Prepare the query-rewriting prompt
    # ---------------------------------------------------------

    prompt = f"""
    You are a query rewriting component for an enterprise DevOps
    knowledge-base search system.

    The original user question was:

    {question}

    The previous retrieval did not return sufficiently relevant
    documents.

    Rewrite the question into a concise search query that is more
    likely to match enterprise DevOps documentation.

    Rules:

    - Preserve the user's original intent.
    - Do not answer the question.
    - Do not invent facts.
    - Keep important technical terms.
    - Prefer concrete DevOps terminology.
    - Correct obvious spacing or wording errors.
    - Return ONLY the rewritten search query.
    - Keep the rewritten query under 20 words.
    """

    # ---------------------------------------------------------
    # Step 3: Rewrite the query safely
    # ---------------------------------------------------------

    try:
        response = get_fast_llm().invoke([HumanMessage(content=prompt)])

        if isinstance(response, dict):
            message = response.get("message")
            rewritten_content = message.get("content") if isinstance(message, dict) else None
        else:
            rewritten_content = getattr(response, "content", None)

        if not isinstance(rewritten_content, str):
            raise ValueError("The query-rewriting model returned an invalid response.")

        rewritten_query = rewritten_content.strip()

        # An empty response is not a usable query.
        if not rewritten_query:
            raise ValueError("The query-rewriting model returned an empty response.")

        rewrite_failed = False

    except Exception:
        # Do not expose internal model errors to the user.
        # Fall back to the original question.
        rewritten_query = question
        rewrite_failed = True

    # ---------------------------------------------------------
    # Step 4: Increment the retry counter
    # ---------------------------------------------------------

    retry_count = state.get("retry_count", 0) + 1

    # ---------------------------------------------------------
    # Step 5: Return the updated graph state
    # ---------------------------------------------------------

    return {
        "rewritten_query": rewritten_query,
        "rewrite_failed": rewrite_failed,
        "retry_count": retry_count,
    }


def retry_search_node(
    state: RAGGraphState,
) -> dict:
    """
    Execute the existing RAG pipeline using the rewritten query.

    The retry results are stored in graph state and exposed to the
    Agent as a ToolMessage so the Agent can generate the final answer.
    """

    rewritten_query = state.get("rewritten_query")

    if not rewritten_query:
        return {
            "retrieved_documents": [],
            "active_query": state.get(
                "active_query",
                state["question"],
            ),
        }

    # print("\n" + "=" * 70)
    # print("RETRY SEARCH")
    # print("=" * 70)
    # print(f"Query : {rewritten_query!r}")

    try:
        results = run_knowledge_base_search(
            query=rewritten_query,
            top_k=state.get("top_k", 5),
            candidate_k=state.get("candidate_k", 20),
            service=state.get("service"),
            environment=state.get("environment"),
            document_type=state.get("document_type"),
        )
    except Exception:
        results = []

    # print("\n" + "=" * 70)
    # print("RETRY SEARCH RESULTS")
    # print("=" * 70)

    # for index, result in enumerate(results, start=1):
    #     print(
    #         f"[{index}] "
    #         f"{result.title} | "
    #         f"reranker_score={result.reranker_score}"
    #     )

    content = format_retrieved_documents(results)

    # Internal retries must still respect the model tool-message protocol:
    # every ToolMessage must match an AIMessage tool call.
    retry_tool_call_id = f"retry-search-{uuid.uuid4().hex}"
    retry_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {
                    "query": rewritten_query,
                    "top_k": state.get("top_k", 5),
                    "candidate_k": state.get("candidate_k", 20),
                    "service": state.get("service"),
                    "environment": state.get("environment"),
                    "document_type": state.get("document_type"),
                },
                "id": retry_tool_call_id,
                "type": "tool_call",
            }
        ],
    )
    retry_message = ToolMessage(
        content=content,
        tool_call_id=retry_tool_call_id,
        name="search_knowledge_base",
    )

    return {
        "messages": [retry_tool_call, retry_message],
        "retrieved_documents": results,
        "active_query": rewritten_query,
    }


def retrieval_failed_node(
    state: RAGGraphState,
) -> dict:
    """
    Produce a safe response when retrieval or grounding
    validation fails.
    """

    grounding_reason = state.get("grounding_reason")

    if grounding_reason:
        return {
            "answer": (
                "I could not verify that the generated answer "
                "was sufficiently supported by the retrieved "
                "documents. Please refine the question or "
                "consult the relevant documentation."
            )
        }

    return {
        "answer": (
            "I could not find sufficiently relevant "
            "information in the enterprise knowledge base "
            "to answer this question reliably."
        )
    }


def grounding_node(
    state: RAGGraphState,
) -> dict:
    """
    Validate whether the generated answer is supported
    by the retrieved documents.

    Invalid grounding responses and model failures are
    rejected safely.
    """

    # ---------------------------------------------------------
    # Step 1: Validate the answer and retrieved documents
    # ---------------------------------------------------------

    answer = state.get("answer", "")
    results = state.get("retrieved_documents", [])

    if not isinstance(answer, str) or not answer.strip():
        return {
            "grounding_passed": False,
            "grounding_score": 0.0,
            "grounding_reason": "No answer was generated.",
        }

    if not results:
        return {
            "grounding_passed": False,
            "grounding_score": 0.0,
            "grounding_reason": ("No retrieved documents are available."),
        }

    question = state.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return {
            "grounding_passed": False,
            "grounding_score": 0.0,
            "grounding_reason": ("The original question is missing or invalid."),
        }

    # ---------------------------------------------------------
    # Step 2: Build the retrieved-document context
    # ---------------------------------------------------------

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[Document {index}]\n"
            f"Title: {result.title}\n"
            f"Source: {result.source}\n"
            f"Content:\n{result.content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # ---------------------------------------------------------
    # Step 3: Prepare the grounding-validation prompt
    # ---------------------------------------------------------

    prompt = f"""
You are a grounding validation component for an enterprise
DevOps AI assistant.

Your task is to determine whether the generated answer is
supported by the retrieved documents.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}

GENERATED ANSWER:
{answer}

Rules:

1. Check whether the factual claims in the answer are supported
   by the retrieved documents.
2. Do not use outside knowledge.
3. If the answer contains unsupported enterprise-specific claims,
   mark it as not grounded.
4. If the answer is fully supported, mark it as grounded.
5. Return ONLY valid JSON.
6. The grounded field must be a JSON Boolean: true or false.
7. The score must be a JSON number between 0.0 and 1.0.
8. Use score 1.0 for grounded answers and 0.0 for ungrounded answers.
9. The reason must be a non-empty string.

Return exactly this structure:

{{
  "grounded": true,
  "score": 1.0,
  "reason": "Brief explanation"
}}

Use:
- grounded=true and score=1.0 when the answer is adequately supported.
- grounded=false and score=0.0 when the answer contains unsupported claims.
"""

    # ---------------------------------------------------------
    # Step 4: Call the grounding model and validate its response
    # ---------------------------------------------------------

    try:
        response = ollama.chat(
            # Grade with a model independent of the one that wrote the answer.
            model=settings.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
            format="json",
            options={
                "temperature": 0,
                "num_predict": 128,
            },
        )

        # Support both dictionary responses and Ollama response objects.

        if isinstance(response, dict):
            message = response.get("message")

            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = None

        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Grounding model returned missing or empty content.")

        validation = json.loads(content)

        if not isinstance(validation, dict):
            raise ValueError("Grounding response must be a JSON object.")

        grounded = validation.get("grounded")
        score = validation.get("score")
        reason = validation.get("reason")

        # The grounded field must be a real Boolean.
        # For example, the string "false" is not acceptable.
        if not isinstance(grounded, bool):
            raise ValueError("Grounding field must be a Boolean.")

        # Booleans are subclasses of integers in Python,
        # so explicitly reject Boolean values as scores.
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Grounding score must be numeric.")

        score = float(score)

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Grounding score must be between 0.0 and 1.0.")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Grounding reason must be a non-empty string.")

        # The prompt specifies binary outcomes.
        # Reject contradictory grounding decisions and scores.
        if grounded and score < 0.5:
            raise ValueError("Grounded result conflicts with its score.")

        if not grounded and score >= 0.5:
            raise ValueError("Ungrounded result conflicts with its score.")

        return {
            "grounding_passed": grounded,
            "grounding_score": score,
            "grounding_reason": reason.strip(),
        }

    except Exception:
        # Fail closed: an invalid or unavailable grounding
        # validator must never approve an answer.
        return {
            "grounding_passed": False,
            "grounding_score": 0.0,
            "grounding_reason": (
                "Grounding validation failed or returned "
                "an invalid response. The answer could not "
                "be verified."
            ),
        }
