from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_evolution_agent.schemas import VisionResult


def normalized_names(value: dict[str, Any]) -> set[str]:
    return {
        str(item.get("normalized_name") or item.get("name", "")).strip().lower()
        for item in value.get("items", [])
    }


def evaluate(path: Path) -> dict[str, float | int]:
    total = strict_json = exact_samples = true_positive = false_positive = false_negative = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        target = row["target"]
        prediction = row.get("output", row.get("prediction", {}))
        try:
            parsed = VisionResult.model_validate(prediction).model_dump(mode="json")
            strict_json += 1
        except Exception:
            continue
        target_names = normalized_names(target)
        predicted_names = normalized_names(parsed)
        exact_samples += int(target_names == predicted_names)
        true_positive += len(target_names & predicted_names)
        false_positive += len(predicted_names - target_names)
        false_negative += len(target_names - predicted_names)
    denominator = max(total, 1)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 1.0
    recall = true_positive / recall_denominator if recall_denominator else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": total,
        "strict_json_rate": strict_json / denominator,
        "ingredient_precision": precision,
        "ingredient_recall": recall,
        "ingredient_f1": f1,
        "exact_sample_accuracy": exact_samples / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
