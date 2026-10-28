import sys
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------
# Configure project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph import nodes

# ---------------------------------------------------------
# Shared test data
# ---------------------------------------------------------

QUESTION = "How do I troubleshoot a failed database replication cluster?"


def create_state() -> dict:
    return {
        "question": QUESTION,
        "retry_count": 0,
    }


class FakeFastLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def invoke(self, messages):
        if self.error is not None:
            raise self.error
        return self.response


# ---------------------------------------------------------
# Test 1: Ollama raises an exception
# ---------------------------------------------------------


def test_ollama_exception() -> None:
    print("\n" + "=" * 70)
    print("QUERY REWRITE EXCEPTION TEST")
    print("=" * 70)

    with patch.object(
        nodes,
        "get_fast_llm",
        return_value=FakeFastLLM(error=RuntimeError("Simulated Ollama connection failure")),
    ):
        result = nodes.rewrite_query_node(create_state())

    print("\nOriginal question:")
    print(QUESTION)

    print("\nFallback query:")
    print(result.get("rewritten_query"))

    print("\nRewrite failed:")
    print(result.get("rewrite_failed"))

    print("\nRetry count:")
    print(result.get("retry_count"))

    assert result["rewritten_query"] == QUESTION
    assert result["rewrite_failed"] is True
    assert result["retry_count"] == 1

    print("\nQUERY REWRITE EXCEPTION TEST PASSED.")


# ---------------------------------------------------------
# Test 2: Ollama returns an invalid response
# ---------------------------------------------------------


def test_invalid_response() -> None:
    print("\n" + "=" * 70)
    print("QUERY REWRITE INVALID RESPONSE TEST")
    print("=" * 70)

    # The expected message/content structure is missing.
    invalid_response = {"unexpected_field": "invalid response"}

    with patch.object(
        nodes,
        "get_fast_llm",
        return_value=FakeFastLLM(response=invalid_response),
    ):
        result = nodes.rewrite_query_node(create_state())

    print("\nFallback query:")
    print(result.get("rewritten_query"))

    print("\nRewrite failed:")
    print(result.get("rewrite_failed"))

    print("\nRetry count:")
    print(result.get("retry_count"))

    assert result["rewritten_query"] == QUESTION
    assert result["rewrite_failed"] is True
    assert result["retry_count"] == 1

    print("\nQUERY REWRITE INVALID RESPONSE TEST PASSED.")


# ---------------------------------------------------------
# Test 3: Ollama returns an empty query
# ---------------------------------------------------------


def test_empty_response() -> None:
    print("\n" + "=" * 70)
    print("QUERY REWRITE EMPTY RESPONSE TEST")
    print("=" * 70)

    empty_response = {
        "message": {
            "content": "   ",
        }
    }

    with patch.object(
        nodes,
        "get_fast_llm",
        return_value=FakeFastLLM(response=empty_response),
    ):
        result = nodes.rewrite_query_node(create_state())

    print("\nFallback query:")
    print(result.get("rewritten_query"))

    print("\nRewrite failed:")
    print(result.get("rewrite_failed"))

    assert result["rewritten_query"] == QUESTION
    assert result["rewrite_failed"] is True
    assert result["retry_count"] == 1

    print("\nQUERY REWRITE EMPTY RESPONSE TEST PASSED.")


# ---------------------------------------------------------
# Test 4: Successful query rewriting
# ---------------------------------------------------------


def test_successful_rewrite() -> None:
    print("\n" + "=" * 70)
    print("SUCCESSFUL QUERY REWRITE TEST")
    print("=" * 70)

    rewritten_query = "database replication cluster troubleshooting"

    successful_response = {
        "message": {
            "content": rewritten_query,
        }
    }

    with patch.object(
        nodes,
        "get_fast_llm",
        return_value=FakeFastLLM(response=successful_response),
    ):
        result = nodes.rewrite_query_node(create_state())

    print("\nOriginal question:")
    print(QUESTION)

    print("\nRewritten query:")
    print(result.get("rewritten_query"))

    print("\nRewrite failed:")
    print(result.get("rewrite_failed"))

    print("\nRetry count:")
    print(result.get("retry_count"))

    assert result["rewritten_query"] == rewritten_query
    assert result["rewrite_failed"] is False
    assert result["retry_count"] == 1

    print("\nSUCCESSFUL QUERY REWRITE TEST PASSED.")


# ---------------------------------------------------------
# Test 5: The fallback query is used by retry search
# ---------------------------------------------------------


def test_fallback_query_is_used_for_search() -> None:
    print("\n" + "=" * 70)
    print("FALLBACK QUERY SEARCH TEST")
    print("=" * 70)

    state = {
        "question": QUESTION,
        "rewritten_query": QUESTION,
        "retry_count": 1,
        "top_k": 5,
        "candidate_k": 20,
    }

    # Simulate a retry search that finds no documents.
    # This test checks query forwarding, not retrieval quality.
    with patch.object(
        nodes,
        "run_knowledge_base_search",
        return_value=[],
    ) as mocked_search:
        result = nodes.retry_search_node(state)

    mocked_search.assert_called_once_with(
        query=QUESTION,
        top_k=5,
        candidate_k=20,
        service=None,
        environment=None,
        document_type=None,
    )

    print("\nSearch query:")
    print(result.get("active_query"))

    print("\nRetrieved document count:")
    print(len(result.get("retrieved_documents", [])))

    assert result["active_query"] == QUESTION
    assert result["retrieved_documents"] == []
    assert len(result["messages"]) == 2
    retry_call, retry_result = result["messages"]
    assert retry_call.tool_calls[0]["name"] == "search_knowledge_base"
    assert retry_call.tool_calls[0]["id"] == retry_result.tool_call_id

    print("\nFALLBACK QUERY SEARCH TEST PASSED.")


# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------


def main() -> None:
    test_ollama_exception()
    test_invalid_response()
    test_empty_response()
    test_successful_rewrite()
    test_fallback_query_is_used_for_search()

    print("\n" + "=" * 70)
    print("ALL QUERY REWRITE FAILURE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
