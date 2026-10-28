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

QUESTION = "What causes CrashLoopBackOff in Kubernetes?"

ANSWER = (
    "CrashLoopBackOff can occur when a container repeatedly "
    "fails to start. Checking container logs can help identify "
    "the cause."
)

DOCUMENT = {
    "title": "Kubernetes CrashLoopBackOff Runbook",
    "source": "data/knowledge_base/kubernetes/crashloopbackoff.md",
    "content": (
        "Containers may repeatedly fail to start because of "
        "application errors or incorrect configuration. "
        "Inspect container logs to investigate the failure."
    ),
}


def create_state() -> dict:
    """
    Create a minimal grounding state with one retrieved document.
    """

    from types import SimpleNamespace

    document = SimpleNamespace(
        title=DOCUMENT["title"],
        source=DOCUMENT["source"],
        content=DOCUMENT["content"],
    )

    return {
        "question": QUESTION,
        "answer": ANSWER,
        "retrieved_documents": [document],
    }


def mock_response(
    grounded,
    score,
    reason,
) -> dict:
    """
    Build a simulated Ollama JSON response.
    """

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


def assert_failed_closed(result: dict) -> None:
    """
    Confirm that invalid responses cannot approve an answer.
    """

    assert result["grounding_passed"] is False
    assert result["grounding_score"] == 0.0
    assert isinstance(result["grounding_reason"], str)
    assert result["grounding_reason"].strip()


# ---------------------------------------------------------
# Test 1: Valid grounded response
# ---------------------------------------------------------


def test_valid_grounded_response() -> None:
    print("\n" + "=" * 70)
    print("VALID GROUNDED RESPONSE TEST")
    print("=" * 70)

    response = mock_response(
        grounded=True,
        score=1.0,
        reason="The answer is supported by the retrieved document.",
    )

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert result["grounding_passed"] is True
    assert result["grounding_score"] == 1.0
    assert result["grounding_reason"]

    print("\nVALID GROUNDED RESPONSE TEST PASSED.")


# ---------------------------------------------------------
# Test 2: Valid ungrounded response
# ---------------------------------------------------------


def test_valid_ungrounded_response() -> None:
    print("\n" + "=" * 70)
    print("VALID UNGROUNDED RESPONSE TEST")
    print("=" * 70)

    response = mock_response(
        grounded=False,
        score=0.0,
        reason="The answer contains unsupported claims.",
    )

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert result["grounding_passed"] is False
    assert result["grounding_score"] == 0.0

    print("\nVALID UNGROUNDED RESPONSE TEST PASSED.")


# ---------------------------------------------------------
# Test 3: Ollama raises an exception
# ---------------------------------------------------------


def test_ollama_exception() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING OLLAMA EXCEPTION TEST")
    print("=" * 70)

    with patch.object(
        nodes.ollama,
        "chat",
        side_effect=RuntimeError("Simulated Ollama connection failure"),
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING OLLAMA EXCEPTION TEST PASSED.")


# ---------------------------------------------------------
# Test 4: Invalid response structure
# ---------------------------------------------------------


def test_invalid_response_structure() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING INVALID RESPONSE STRUCTURE TEST")
    print("=" * 70)

    response = {"unexpected_field": "invalid response"}

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING INVALID RESPONSE STRUCTURE TEST PASSED.")


# ---------------------------------------------------------
# Test 5: Invalid JSON
# ---------------------------------------------------------


def test_invalid_json() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING INVALID JSON TEST")
    print("=" * 70)

    response = {"message": {"content": "{this is not valid JSON"}}

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING INVALID JSON TEST PASSED.")


# ---------------------------------------------------------
# Test 6: The grounded field is a string
# ---------------------------------------------------------


def test_grounded_string_rejected() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING STRING BOOLEAN TEST")
    print("=" * 70)

    response = mock_response(
        grounded="false",
        score=0.0,
        reason="The answer is unsupported.",
    )

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING STRING BOOLEAN TEST PASSED.")


# ---------------------------------------------------------
# Test 7: Invalid score
# ---------------------------------------------------------


def test_invalid_score() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING INVALID SCORE TEST")
    print("=" * 70)

    response = mock_response(
        grounded=True,
        score=5.0,
        reason="The answer is supported.",
    )

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING INVALID SCORE TEST PASSED.")


# ---------------------------------------------------------
# Test 8: Missing reason
# ---------------------------------------------------------


def test_missing_reason() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING MISSING REASON TEST")
    print("=" * 70)

    response = {"message": {"content": ('{"grounded": true, "score": 1.0}')}}

    with patch.object(
        nodes.ollama,
        "chat",
        return_value=response,
    ):
        result = nodes.grounding_node(create_state())

    print(result)

    assert_failed_closed(result)

    print("\nGROUNDING MISSING REASON TEST PASSED.")


# ---------------------------------------------------------
# Test 9: No retrieved documents
# ---------------------------------------------------------


def test_no_retrieved_documents() -> None:
    print("\n" + "=" * 70)
    print("GROUNDING WITHOUT DOCUMENTS TEST")
    print("=" * 70)

    state = {
        "question": QUESTION,
        "answer": ANSWER,
        "retrieved_documents": [],
    }

    # The model must not be called when there are no documents.
    with patch.object(
        nodes.ollama,
        "chat",
    ) as mocked_chat:
        result = nodes.grounding_node(state)

    print(result)

    assert_failed_closed(result)
    mocked_chat.assert_not_called()

    print("\nGROUNDING WITHOUT DOCUMENTS TEST PASSED.")


# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------


def main() -> None:
    test_valid_grounded_response()
    test_valid_ungrounded_response()
    test_ollama_exception()
    test_invalid_response_structure()
    test_invalid_json()
    test_grounded_string_rejected()
    test_invalid_score()
    test_missing_reason()
    test_no_retrieved_documents()

    print("\n" + "=" * 70)
    print("ALL GROUNDING RESPONSE VALIDATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
