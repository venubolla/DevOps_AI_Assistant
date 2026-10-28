import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

# ---------------------------------------------------------
# Configure project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph import nodes
from app.graph.graph import (
    graph,
    route_after_agent,
    route_after_grounding,
    route_after_relevance,
)
from app.graph.nodes import MAX_RETRIES

# ---------------------------------------------------------
# Shared test data
# ---------------------------------------------------------

QUESTION = "What causes CrashLoopBackOff in Kubernetes?"

EXPECTED_AGENT_FAILURE = nodes.AGENT_FAILURE_MESSAGE

EXPECTED_RETRIEVAL_FAILURE = (
    "I could not find sufficiently relevant "
    "information in the enterprise knowledge base "
    "to answer this question reliably."
)

EXPECTED_GROUNDING_FAILURE = (
    "I could not verify that the generated answer "
    "was sufficiently supported by the retrieved "
    "documents. Please refine the question or "
    "consult the relevant documentation."
)

SUPPORTED_ANSWER = (
    "CrashLoopBackOff can occur when a container "
    "repeatedly fails to start. Inspect the container "
    "logs to investigate the failure. [1]"
)


# ---------------------------------------------------------
# Mock LLM classes
# ---------------------------------------------------------


class FakeBoundLLM:
    """
    Simulate the Agent's initial tool-selection call.
    """

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {
                        "query": QUESTION,
                        "top_k": 5,
                        "candidate_k": 20,
                        "service": None,
                        "environment": None,
                        "document_type": None,
                    },
                    "id": "test-tool-call",
                    "type": "tool_call",
                }
            ],
        )


class FakeLLM:
    """
    Simulate an LLM that requests retrieval first
    and generates an answer after receiving tool results.
    """

    def bind_tools(self, tools):
        return FakeBoundLLM()

    def invoke(self, messages):
        return AIMessage(
            content=SUPPORTED_ANSWER,
        )


def create_document():
    """
    Create a lightweight retrieved-document object.
    """

    return SimpleNamespace(
        title="Kubernetes CrashLoopBackOff Runbook",
        source="data/knowledge_base/kubernetes/crashloopbackoff.md",
        content=(
            "Containers may repeatedly fail to start "
            "because of application errors or incorrect "
            "configuration. Inspect container logs."
        ),
        reranker_score=1.0,
    )


def create_grounding_response(
    grounded: bool,
) -> dict:
    """
    Create a mock Ollama grounding response.
    """

    if grounded:
        score = 1.0
        reason = "The answer is supported by the document."
    else:
        score = 0.0
        reason = "The answer contains unsupported claims."

    import json

    return {
        "message": {
            "content": json.dumps(
                {
                    "grounded": grounded,
                    "score": score,
                    "reason": reason,
                }
            )
        }
    }


# ---------------------------------------------------------
# Test group 1: Direct routing decisions
# ---------------------------------------------------------


def test_agent_routing() -> None:
    print("\n" + "=" * 70)
    print("AGENT ROUTING TESTS")
    print("=" * 70)

    # An agent failure must terminate the graph.
    assert (
        route_after_agent(
            {
                "agent_failed": True,
            }
        )
        == "end"
    )

    # An empty message list must terminate the graph.
    assert (
        route_after_agent(
            {
                "messages": [],
            }
        )
        == "end"
    )

    # An AI message requesting a tool must route to tool.
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {
                    "query": QUESTION,
                },
                "id": "routing-test-call",
                "type": "tool_call",
            }
        ],
    )

    assert (
        route_after_agent(
            {
                "messages": [tool_call_message],
            }
        )
        == "tool"
    )

    # An ordinary AI message must route to grounding.
    answer_message = AIMessage(
        content=SUPPORTED_ANSWER,
    )

    assert (
        route_after_agent(
            {
                "messages": [answer_message],
            }
        )
        == "grounding"
    )

    print("Agent failure route: END")
    print("Empty messages route: END")
    print("Tool call route: tool")
    print("Generated answer route: grounding")
    print("\nAGENT ROUTING TESTS PASSED.")


def test_relevance_routing() -> None:
    print("\n" + "=" * 70)
    print("RELEVANCE ROUTING TESTS")
    print("=" * 70)

    # Relevant documents proceed to answer generation.
    assert (
        route_after_relevance(
            {
                "relevance_passed": True,
                "retry_count": 0,
            }
        )
        == "answer"
    )

    # Failed relevance validation with a retry available.
    assert (
        route_after_relevance(
            {
                "relevance_passed": False,
                "retry_count": 0,
            }
        )
        == "rewrite"
    )

    # Failed relevance validation after the retry limit.
    assert (
        route_after_relevance(
            {
                "relevance_passed": False,
                "retry_count": MAX_RETRIES,
            }
        )
        == "end"
    )

    print("Relevant documents route: answer")
    print("Retry available route: rewrite")
    print("Retry limit reached route: END")
    print("\nRELEVANCE ROUTING TESTS PASSED.")


def test_grounding_routing() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING ROUTING TESTS")
    print("=" * 70)

    assert (
        route_after_grounding(
            {
                "grounding_passed": True,
            }
        )
        == "pass"
    )

    assert (
        route_after_grounding(
            {
                "grounding_passed": False,
            }
        )
        == "fail"
    )

    print("Grounded answer route: pass")
    print("Ungrounded answer route: fail")
    print("\nGROUNDING ROUTING TESTS PASSED.")


# ---------------------------------------------------------
# Test group 2: Invalid input terminates before the Agent
# ---------------------------------------------------------


def test_invalid_input_ends_graph() -> None:
    print("\n" + "=" * 70)
    print("INVALID INPUT GRAPH TEST")
    print("=" * 70)

    with patch.object(
        nodes,
        "get_llm",
        side_effect=AssertionError("The LLM should not be initialized for invalid input."),
    ):
        result = graph.invoke(
            {
                "question": "   ",
            }
        )

    print("Answer:")
    print(result.get("answer"))

    assert result["answer"] == ("I could not process an empty question. Please provide a question.")

    print("\nINVALID INPUT GRAPH TEST PASSED.")


# ---------------------------------------------------------
# Test group 3: Agent initialization failure terminates graph
# ---------------------------------------------------------


def test_agent_failure_ends_graph() -> None:
    print("\n" + "=" * 70)
    print("AGENT FAILURE GRAPH TEST")
    print("=" * 70)

    with patch.object(
        nodes,
        "get_llm",
        side_effect=RuntimeError("Simulated LLM initialization failure"),
    ):
        result = graph.invoke(
            {
                "question": QUESTION,
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("Answer:")
    print(result.get("answer"))

    print("Agent failed:")
    print(result.get("agent_failed"))

    assert result.get("agent_failed") is True

    assert result["answer"] == EXPECTED_AGENT_FAILURE

    # Because the graph ended at the Agent, retrieval
    # and grounding must not have produced results.
    assert result.get("retrieved_documents", []) == []
    assert result.get("grounding_passed") is None

    print("\nAGENT FAILURE GRAPH TEST PASSED.")


# ---------------------------------------------------------
# Test group 4: Retrieval retry is exhausted
# ---------------------------------------------------------


def test_retrieval_retry_exhausted() -> None:
    print("\n" + "=" * 70)
    print("RETRIEVAL RETRY EXHAUSTED GRAPH TEST")
    print("=" * 70)

    with (
        patch.object(
            nodes,
            "get_llm",
            return_value=FakeLLM(),
        ),
        patch.object(
            nodes,
            "run_knowledge_base_search",
            return_value=[],
        ) as mocked_search,
        patch.object(
            nodes,
            "get_fast_llm",
            return_value=SimpleNamespace(
                invoke=lambda messages: SimpleNamespace(
                    content="Kubernetes CrashLoopBackOff troubleshooting"
                )
            ),
        ) as mocked_rewrite,
    ):
        result = graph.invoke(
            {
                "question": QUESTION,
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("Answer:")
    print(result.get("answer"))

    print("Retry count:")
    print(result.get("retry_count"))

    print("Retrieved documents:")
    print(len(result.get("retrieved_documents", [])))

    assert result["answer"] == EXPECTED_RETRIEVAL_FAILURE

    assert result["retry_count"] == MAX_RETRIES

    assert result.get("relevance_passed") is False

    assert result.get("retrieved_documents", []) == []

    # Initial retrieval plus one retry search.
    assert mocked_search.call_count == 2

    # Query rewriting should occur exactly once.
    mocked_rewrite.assert_called_once_with()

    print("\nRETRIEVAL RETRY EXHAUSTED GRAPH TEST PASSED.")


# ---------------------------------------------------------
# Test group 5: Grounding failure routes to safe response
# ---------------------------------------------------------


def test_grounding_failure_routes_to_safe_response() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING FAILURE GRAPH TEST")
    print("=" * 70)

    document = create_document()

    with (
        patch.object(
            nodes,
            "get_llm",
            return_value=FakeLLM(),
        ),
        patch.object(
            nodes,
            "run_knowledge_base_search",
            return_value=[document],
        ),
        patch.object(
            nodes.ollama,
            "chat",
            return_value=create_grounding_response(grounded=False),
        ),
    ):
        result = graph.invoke(
            {
                "question": QUESTION,
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("Grounding passed:")
    print(result.get("grounding_passed"))

    print("Final answer:")
    print(result.get("answer"))

    assert result.get("relevance_passed") is True

    assert result.get("grounding_passed") is False

    assert result["answer"] == EXPECTED_GROUNDING_FAILURE

    print("\nGROUNDING FAILURE GRAPH TEST PASSED.")


# ---------------------------------------------------------
# Test group 6: Grounding success terminates normally
# ---------------------------------------------------------


def test_grounding_success_ends_graph() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING SUCCESS GRAPH TEST")
    print("=" * 70)

    document = create_document()

    with (
        patch.object(
            nodes,
            "get_llm",
            return_value=FakeLLM(),
        ),
        patch.object(
            nodes,
            "run_knowledge_base_search",
            return_value=[document],
        ),
        patch.object(
            nodes.ollama,
            "chat",
            return_value=create_grounding_response(grounded=True),
        ),
    ):
        result = graph.invoke(
            {
                "question": QUESTION,
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("Grounding passed:")
    print(result.get("grounding_passed"))

    print("Final answer:")
    print(result.get("answer"))

    assert result.get("relevance_passed") is True

    assert result.get("grounding_passed") is True

    assert result["answer"] == SUPPORTED_ANSWER

    print("\nGROUNDING SUCCESS GRAPH TEST PASSED.")


# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------


def main() -> None:
    test_agent_routing()
    test_relevance_routing()
    test_grounding_routing()

    test_invalid_input_ends_graph()
    test_agent_failure_ends_graph()
    test_retrieval_retry_exhausted()

    test_grounding_failure_routes_to_safe_response()
    test_grounding_success_ends_graph()

    print("\n" + "=" * 70)
    print("ALL GRAPH ROUTING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
