import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


def test_agent_ask_endpoint() -> None:
    client = TestClient(app)
    fake_document = SimpleNamespace(
        title="Kubernetes Runbook",
        source="data/knowledge_base/kubernetes/crashloopbackoff.md",
        reranker_score=0.9,
        hybrid_score=0.03,
    )

    with patch.object(
        routes.graph,
        "invoke",
        return_value={
            "answer": "Inspect pod logs and events. [1]",
            "retrieved_documents": [fake_document],
        },
    ) as mocked_invoke:
        response = client.post(
            "/api/v1/agent/ask",
            json={"question": "How do I troubleshoot CrashLoopBackOff?", "top_k": 3},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "Inspect pod logs and events. [1]"
    assert payload["sources"][0]["citation"] == "[1]"
    called_state = mocked_invoke.call_args.args[0]
    assert called_state["top_k"] == 3
    assert called_state["candidate_k"] == 12

    invalid = client.post("/api/v1/agent/ask", json={"question": "   "})
    assert invalid.status_code == 422

    with patch.object(routes.graph, "invoke", side_effect=RuntimeError("internal secret")):
        failed = client.post(
            "/api/v1/agent/ask",
            json={"question": "What happened?"},
        )
    assert failed.status_code == 503
    assert "internal secret" not in failed.text

    print("AGENT API TESTS PASSED")


if __name__ == "__main__":
    test_agent_ask_endpoint()
