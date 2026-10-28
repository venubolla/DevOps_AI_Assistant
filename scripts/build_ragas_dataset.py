import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.ragas_eval import build_ragas_samples

OUTPUT_FILE = Path("data/ragas_dataset.json")


def build_dataset() -> None:
    samples = build_ragas_samples()
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "dataset_version": "rag_v2",
            "top_k": 5,
            "candidate_k": 20,
            "sample_count": len(samples),
        },
        "samples": samples,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Total samples: {len(samples)}")


if __name__ == "__main__":
    build_dataset()
