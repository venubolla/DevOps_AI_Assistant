import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragas import evaluate
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from app.evaluation.dataset import load_evaluation_cases
from app.evaluation.ragas_config import get_ragas_llm
from app.evaluation.ragas_dataset import load_ragas_dataset

RESULTS_DIR = Path("data/ragas_results")


def main():

    print("=" * 80)
    print("RAGAS EVALUATION")
    print("=" * 80)

    print("\nLoading persistent dataset...")

    dataset = load_ragas_dataset()
    current_case_count = len(load_evaluation_cases())

    print(f"Loaded {len(dataset.samples)} samples.")
    if len(dataset.samples) != current_case_count:
        raise SystemExit(
            "The saved RAGAS dataset does not match the current evaluation "
            f"set ({len(dataset.samples)} saved samples vs "
            f"{current_case_count} questions). Rebuild it with "
            "`python scripts/build_ragas_dataset.py` before evaluating."
        )

    print("\nInitializing evaluator...")

    evaluator_llm = get_ragas_llm()

    metrics = [
        Faithfulness(),
        LLMContextRecall(),
        FactualCorrectness(mode="precision"),
    ]

    print("\nRunning Ragas evaluation...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        # Local Ollama serves ~1 request at a time; keep concurrency low so
        # queued requests don't hit the timeout and trigger wasteful retries.
        run_config=RunConfig(
            max_workers=1,
            timeout=600,
            max_retries=2,
        ),
    )

    # Convert Ragas result into a pandas DataFrame
    result_df = result.to_pandas()

    metric_columns = [
        "faithfulness",
        "context_recall",
        "factual_correctness(mode=precision)",
    ]
    failed = [column for column in metric_columns if result_df[column].isna().all()]
    if failed:
        raise SystemExit(
            "All samples failed for metric(s): "
            f"{', '.join(failed)}. Check for timeouts or model errors."
        )

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save detailed per-sample results
    detailed_file = RESULTS_DIR / f"evaluation_{timestamp}.json"

    result_df.to_json(
        detailed_file,
        orient="records",
        indent=2,
    )

    # Extract aggregate metrics
    summary = {
        "timestamp": timestamp,
        "dataset": "data/ragas_dataset.json",
        "dataset_version": "rag_v2",
        "samples": len(dataset.samples),
        "metrics": {
            "faithfulness": float(result_df["faithfulness"].mean()),
            "context_recall": float(result_df["context_recall"].mean()),
            "factual_correctness": float(result_df["factual_correctness(mode=precision)"].mean()),
        },
    }

    summary_file = RESULTS_DIR / f"summary_{timestamp}.json"

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n")
    print("=" * 80)
    print("RAGAS RESULTS")
    print("=" * 80)

    print(f"Faithfulness: {summary['metrics']['faithfulness']:.4f}")

    print(f"Context Recall: {summary['metrics']['context_recall']:.4f}")

    print(f"Factual Correctness: {summary['metrics']['factual_correctness']:.4f}")

    print("\nSaved files:")

    print(detailed_file)
    print(summary_file)


if __name__ == "__main__":
    main()
