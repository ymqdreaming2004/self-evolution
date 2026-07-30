import json

from scripts.evaluate_vision import evaluate


def test_evaluation_metrics(tmp_path) -> None:
    result = {
        "items": [
            {
                "name": "牛奶",
                "normalized_name": "牛奶",
                "quantity": 1,
                "unit": "盒",
                "production_date": None,
                "shelf_life_days": None,
                "expiry_date": "2026-08-01",
                "date_source": "printed",
                "confidence": 0.9,
                "evidence_text": "2026-08-01",
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
    assert metrics["ingredient_name_accuracy"] == 1
    assert metrics["full_sample_accuracy"] == 1
