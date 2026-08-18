import json

from scripts.evaluate_vision import evaluate


def test_evaluation_metrics(tmp_path) -> None:
    result = {
        "items": [
            {
                "name": "牛奶",
                "normalized_name": "牛奶",
                "confidence": 0.9,
            }
        ],
        "model_version": "v1",
        "raw_text": "",
    }
    dataset = tmp_path / "predictions.jsonl"
    dataset.write_text(
        json.dumps({"target": result, "output": result}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = evaluate(dataset)
    assert metrics["strict_json_rate"] == 1
    assert metrics["ingredient_precision"] == 1
    assert metrics["ingredient_recall"] == 1
    assert metrics["ingredient_f1"] == 1
    assert metrics["exact_sample_accuracy"] == 1
