import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph import nodes
from app.graph.graph import route_after_scope_check

STATE = {"question": "What is a Transformer?", "top_k": 5, "candidate_k": 20}


def fake_classifier(label: str):
    return SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=label))


def run_scope_check(label: str, results: list) -> dict:
    with (
        patch.object(nodes, "run_knowledge_base_search", return_value=results),
        patch.object(nodes, "get_fast_llm", return_value=fake_classifier(label)),
    ):
        return nodes.scope_check_node(STATE)


def test_off_topic_is_refused() -> None:
    result = run_scope_check("OTHER", [])

    assert result["scope_decision"] == "refused"
    assert result["answer"] == nodes.OUT_OF_SCOPE_MESSAGE
    assert route_after_scope_check(result) == "end"


def test_smalltalk_keeps_the_model_reply() -> None:
    result = run_scope_check("SMALLTALK", [])

    assert result == {"scope_decision": "conversational"}
    assert route_after_scope_check(result) == "end"


def test_devops_without_knowledge_base_coverage_fails_safe() -> None:
    result = run_scope_check("DEVOPS", [])

    assert result["scope_decision"] == "unavailable"
    assert route_after_scope_check(result) == "failed"


def test_low_relevance_documents_are_treated_as_no_coverage() -> None:
    weak = SimpleNamespace(reranker_score=-11.0)

    result = run_scope_check("OTHER", [weak])

    assert result["scope_decision"] == "refused"


def test_relevant_documents_force_retrieval_regardless_of_label() -> None:
    strong = SimpleNamespace(
        title="Runbook",
        source="kb/x.md",
        content="Inspect pod logs.",
        reranker_score=3.0,
    )

    result = run_scope_check("OTHER", [strong])

    assert result["scope_decision"] == "retrieve"
    assert result["retrieved_documents"] == [strong]
    assert route_after_scope_check(result) == "agent"


def test_classifier_failure_fails_closed() -> None:
    def broken():
        raise RuntimeError("model unavailable")

    with patch.object(nodes, "get_fast_llm", side_effect=broken):
        assert nodes.classify_scope("hello") == "OTHER"


def test_unparseable_classifier_output_fails_closed() -> None:
    with patch.object(nodes, "get_fast_llm", return_value=fake_classifier("maybe?")):
        assert nodes.classify_scope("hello") == "OTHER"
