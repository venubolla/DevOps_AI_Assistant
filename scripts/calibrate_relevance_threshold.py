"""Report candidate relevance thresholds against labeled retrieval cases.

This script reports evidence only; it does not modify .env or choose a threshold
automatically. Run it against the local PostgreSQL knowledge base.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.evaluation.dataset import load_evaluation_cases
from app.rag.hybrid_search import hybrid_reranked_search


def metrics(
    labels: list[bool], predictions: list[bool]
) -> tuple[float, float, float, int, int, int]:
    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum((not y) and p for y, p in zip(labels, predictions))
    fn = sum(y and (not p) for y, p in zip(labels, predictions))
    tn = sum((not y) and (not p) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def main() -> None:
    cases = load_evaluation_cases()
    rows = []
    for case in cases:
        results = hybrid_reranked_search(
            query=case.question,
            top_k=5,
            candidate_k=20,
        )
        scores = [float(item.reranker_score) for item in results if item.reranker_score is not None]
        expected = set(case.expected_documents)
        found = {Path(item.source).name for item in results}
        label = bool(expected.intersection(found))
        rows.append((max(scores) if scores else None, label, case.id))

    scored = [row for row in rows if row[0] is not None]
    if not scored:
        print("No reranker scores were returned. Check PostgreSQL and the retrieval pipeline.")
        return

    print(f"Evaluation cases: {len(cases)}")
    print(f"Cases with reranker scores: {len(scored)}")
    print(f"Current RELEVANCE_THRESHOLD: {settings.relevance_threshold}")
    print("Positive label: at least one expected source document appeared in top 5.")
    if len(cases) < 30:
        print("WARNING: fewer than 30 labeled questions; treat threshold results as exploratory.")
    print("\nthreshold\tprecision\trecall\tF1\tTP\tFP\tFN")

    scores = sorted({row[0] for row in scored})
    candidates = {settings.relevance_threshold, scores[0] - 1e-6, scores[-1] + 1e-6}
    candidates.update((left + right) / 2 for left, right in zip(scores, scores[1:]))
    reports = []
    labels = [row[1] for row in scored]
    values = [row[0] for row in scored]
    for threshold in sorted(candidates):
        predictions = [score >= threshold for score in values]
        precision, recall, f1, tp, fp, fn = metrics(labels, predictions)
        reports.append((f1, threshold, precision, recall, tp, fp, fn))
        print(f"{threshold:.6f}\t{precision:.3f}\t{recall:.3f}\t{f1:.3f}\t{tp}\t{fp}\t{fn}")

    best = max(reports, key=lambda item: (item[0], item[1]))
    print(f"\nHighest observed F1 threshold: {best[1]:.6f} (F1={best[0]:.3f})")
    print(
        "Do not copy this threshold into .env without reviewing false positives/negatives and testing on a separate holdout set."
    )


if __name__ == "__main__":
    main()
