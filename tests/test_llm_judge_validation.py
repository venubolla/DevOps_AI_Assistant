import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.llm_judge import parse_judge_response

VALID = {
    "faithfulness": 0.9,
    "answer_relevance": 0.8,
    "correctness": 1,
    "citation_correctness": 0.75,
    "reason": "Claims are supported by the provided context.",
}


def test_judge_response_validation() -> None:
    parsed = parse_judge_response(json.dumps(VALID))
    assert parsed["faithfulness"] == 0.9
    assert isinstance(parsed["correctness"], float)

    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_judge_response(fenced)["reason"] == VALID["reason"]

    invalid_cases = [
        "not json",
        "[]",
        json.dumps({"faithfulness": 0.5}),
        json.dumps({**VALID, "correctness": True}),
        json.dumps({**VALID, "correctness": 1.5}),
        json.dumps({**VALID, "reason": "  "}),
    ]
    for raw in invalid_cases:
        try:
            parse_judge_response(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid judge response to fail: {raw!r}")

    print("LLM JUDGE VALIDATION TESTS PASSED")


if __name__ == "__main__":
    test_judge_response_validation()
