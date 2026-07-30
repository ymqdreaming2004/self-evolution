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


def date_map(value: dict[str, Any]) -> dict[str, str | None]:
    return {
        str(item.get("normalized_name") or item.get("name", "")).strip().lower(): item.get(
            "expiry_date"
        )
        for item in value.get("items", [])
    }


def evaluate(path: Path) -> dict[str, float | int]:
    total = strict_json = name_correct = date_correct = full_correct = 0
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
        names_match = target_names == predicted_names
        dates_match = date_map(target) == date_map(parsed)
        name_correct += int(names_match)
        date_correct += int(dates_match)
        full_correct += int(names_match and dates_match)
    denominator = max(total, 1)
    return {
        "samples": total,
        "strict_json_rate": strict_json / denominator,
        "ingredient_name_accuracy": name_correct / denominator,
        "expiry_date_accuracy": date_correct / denominator,
        "full_sample_accuracy": full_correct / denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
