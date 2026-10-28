import json
from pathlib import Path

from app.evaluation.models import EvaluationCase

EVALUATION_FILE = Path("data/evaluation/rag_eval.json")


def load_evaluation_cases() -> list[EvaluationCase]:

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return [
        EvaluationCase(
            id=item["id"],
            question=item["question"],
            expected_documents=item["expected_documents"],
            expected_keywords=item["expected_keywords"],
            reference_answer=item["reference_answer"],
        )
        for item in data
    ]
