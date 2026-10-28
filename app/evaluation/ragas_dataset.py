import json
from pathlib import Path

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

DATASET_FILE = Path("data/ragas_dataset.json")


def load_ragas_dataset() -> EvaluationDataset:

    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    samples = []

    for item in payload["samples"]:
        sample = SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            reference=item["reference"],
            retrieved_contexts=item["retrieved_contexts"],
        )

        samples.append(sample)

    return EvaluationDataset(samples=samples)
