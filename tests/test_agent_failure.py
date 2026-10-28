import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage

# ---------------------------------------------------------
# Configure project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.graph import nodes
from app.graph.graph import graph

# ---------------------------------------------------------
# Test 1: LLM initialization failure
# ---------------------------------------------------------


def test_initialization_failure() -> None:
    print("\n" + "=" * 70)
    print("AGENT INITIALIZATION FAILURE TEST")
    print("=" * 70)

    # Simulate an LLM initialization failure.
    # No actual model initialization is required for this test.

    with patch.object(
        nodes,
        "get_llm",
        side_effect=RuntimeError("Simulated LLM initialization failure"),
    ):
        result = graph.invoke(
            {
                "question": ("What causes CrashLoopBackOff in Kubernetes?"),
                "top_k": 5,
                "candidate_k": 20,
            }
        )

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nAgent failed:")
    print(result.get("agent_failed"))

    # Verify the graph handled the failure safely.
    assert result.get("agent_failed") is True

    assert result.get("answer") == nodes.AGENT_FAILURE_MESSAGE

    print("\nAGENT INITIALIZATION FAILURE TEST PASSED.")


# ---------------------------------------------------------
# Test 2: LLM answer-generation failure
# ---------------------------------------------------------


def test_answer_generation_failure() -> None:
    print("\n" + "=" * 70)
    print("AGENT ANSWER GENERATION FAILURE TEST")
    print("=" * 70)

    # Simulate an LLM that initializes successfully
    # but fails when generating the answer.

    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("Simulated answer generation failure")

    state = {
        "question": ("What causes CrashLoopBackOff in Kubernetes?"),
        "messages": [
            HumanMessage(content=("What causes CrashLoopBackOff in Kubernetes?")),
            ToolMessage(
                content=(
                    "[1]\n"
                    "Title: Kubernetes CrashLoopBackOff Runbook\n"
                    "Content: Containers may repeatedly fail to start."
                ),
                tool_call_id="test-tool-call",
            ),
        ],
    }

    # Return the simulated LLM instead of initializing
    # the real model.
    with patch.object(
        nodes,
        "get_llm",
        return_value=BrokenLLM(),
    ):
        result = nodes.agent_node(state)

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nAgent failed:")
    print(result.get("agent_failed"))

    # Verify the failure was caught and handled.
    assert result.get("agent_failed") is True

    assert result.get("answer") == nodes.AGENT_FAILURE_MESSAGE

    print("\nAGENT ANSWER GENERATION FAILURE TEST PASSED.")


# ---------------------------------------------------------
# Run both tests
# ---------------------------------------------------------


def main() -> None:
    test_initialization_failure()
    test_answer_generation_failure()

    print("\n" + "=" * 70)
    print("ALL AGENT FAILURE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
