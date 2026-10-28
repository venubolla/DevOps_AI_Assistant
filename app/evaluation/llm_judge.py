import json
import math

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.rag.hybrid_search import HybridSearchResult
from app.rag.llm import get_judge_llm

JUDGE_SYSTEM_PROMPT = """
You are an evaluator for an enterprise DevOps RAG system.

Evaluate the generated answer using ONLY the supplied
question, reference answer, and retrieved context.

Do not use outside knowledge.

Evaluate four dimensions from 0.0 to 1.0:

1. faithfulness:
   Are the claims in the answer supported by the
   retrieved context?

2. answer_relevance:
   Does the answer directly and appropriately
   address the user's question?

3. correctness:
   Does the answer agree with the reference answer
   on the important factual points?

4. citation_correctness:
   Do the citations in the answer point to context
   that actually supports the cited claims?

Return ONLY valid JSON:

{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "correctness": 0.0,
  "citation_correctness": 0.0,
  "reason": "short explanation"
}

Do not include markdown.
Do not include additional fields.
"""


SCORE_FIELDS = (
    "faithfulness",
    "answer_relevance",
    "correctness",
    "citation_correctness",
)


def parse_judge_response(raw_content: str) -> dict:
    """Parse and validate the custom judge's JSON response."""

    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("LLM judge returned an empty response.")

    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        evaluation = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM judge returned invalid JSON.") from exc

    if not isinstance(evaluation, dict):
        raise ValueError("LLM judge response must be a JSON object.")

    required_fields = set(SCORE_FIELDS) | {"reason"}
    missing = sorted(required_fields - set(evaluation))
    unexpected = sorted(set(evaluation) - required_fields)
    if missing:
        raise ValueError("LLM judge response is missing required fields: " + ", ".join(missing))
    if unexpected:
        raise ValueError("LLM judge response contains unexpected fields: " + ", ".join(unexpected))

    for field in SCORE_FIELDS:
        score = evaluation[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"LLM judge field '{field}' must be numeric.")
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"LLM judge field '{field}' must be between 0.0 and 1.0.")
        evaluation[field] = score

    if not isinstance(evaluation["reason"], str) or not evaluation["reason"].strip():
        raise ValueError("LLM judge field 'reason' must be a non-empty string.")

    return evaluation


def build_judge_context(
    results: list[HybridSearchResult],
) -> str:

    parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        parts.append(
            f"""
[{index}]
Title: {result.title}
Source: {result.source}

Content:
{result.content}
"""
        )

    return "\n".join(parts)


def evaluate_with_llm(
    question: str,
    reference_answer: str,
    generated_answer: str,
    results: list[HybridSearchResult],
) -> dict:

    context = build_judge_context(results)

    prompt = f"""
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{context}

Evaluate the generated answer.
"""

    llm = get_judge_llm()

    # Request JSON mode where supported, then validate the schema ourselves.
    response = llm.bind(format="json").invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return parse_judge_response(response.content)
