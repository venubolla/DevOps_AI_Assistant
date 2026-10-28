from dataclasses import dataclass


@dataclass
class EvaluationCase:
    id: str
    question: str
    expected_documents: list[str]
    expected_keywords: list[str]
    reference_answer: str


@dataclass
class RetrievalEvaluationResult:
    case_id: str
    question: str
    retrieved_documents: list[str]
    expected_documents: list[str]
    recall_at_k: float
    reciprocal_rank: float
    hit: bool


@dataclass
class AnswerEvaluationResult:
    case_id: str
    question: str
    answer: str
    reference_answer: str
    keyword_coverage: float
    citation_validity: float
    context_term_coverage: float
    reference_similarity: float
    citation_count: int
    faithfulness: float
    answer_relevance: float
    correctness: float
    citation_correctness: float
    judge_reason: str


@dataclass
class EvaluationSummary:
    total_cases: int
    hits: int
    hit_rate_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
