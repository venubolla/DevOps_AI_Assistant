import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import retrieval
from app.evaluation.models import EvaluationCase


def result(source: str):
    return SimpleNamespace(source=source)


def test_retrieval_metrics() -> None:
    case = EvaluationCase(
        id="metric_test",
        question="test question",
        expected_documents=["a.md", "b.md", "c.md"],
        expected_keywords=[],
        reference_answer="reference",
    )
    with patch.object(
        retrieval,
        "hybrid_reranked_search",
        return_value=[result("/kb/a.md"), result("/kb/a.md"), result("/kb/c.md")],
    ):
        item = retrieval.evaluate_retrieval_case(case, top_k=3)
    assert item.hit is True
    assert item.recall_at_k == 2 / 3
    assert item.reciprocal_rank == 1.0

    with patch.object(
        retrieval,
        "hybrid_reranked_search",
        return_value=[result("/kb/unrelated.md")],
    ):
        item = retrieval.evaluate_retrieval_case(case, top_k=3)
    assert item.hit is False
    assert item.recall_at_k == 0.0

    print("RETRIEVAL METRIC TESTS PASSED")


if __name__ == "__main__":
    test_retrieval_metrics()
