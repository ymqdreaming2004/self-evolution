"""Convert GroceryStoreDataset splits to LLaMAFactory multimodal ShareGPT JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COARSE_NAME_ZH = {
    "Apple": "苹果",
    "Avocado": "牛油果",
    "Banana": "香蕉",
    "Kiwi": "猕猴桃",
    "Lemon": "柠檬",
    "Lime": "青柠",
    "Mango": "芒果",
    "Melon": "甜瓜",
    "Nectarine": "油桃",
    "Orange": "橙子",
    "Papaya": "木瓜",
    "Passion-Fruit": "百香果",
    "Peach": "桃子",
    "Pear": "梨",
    "Pineapple": "菠萝",
    "Plum": "李子",
    "Pomegranate": "石榴",
    "Red-Grapefruit": "红西柚",
    "Satsumas": "蜜橘",
    "Juice": "果汁",
    "Milk": "牛奶",
    "Oatghurt": "燕麦酸奶",
    "Oat-Milk": "燕麦奶",
    "Sour-Cream": "酸奶油",
    "Sour-Milk": "酸牛奶",
    "Soyghurt": "豆乳酸奶",
    "Soy-Milk": "豆奶",
    "Yoghurt": "酸奶",
    "Asparagus": "芦笋",
    "Aubergine": "茄子",
    "Cabbage": "卷心菜",
    "Carrots": "胡萝卜",
    "Cucumber": "黄瓜",
    "Garlic": "大蒜",
    "Ginger": "生姜",
    "Leek": "韭葱",
    "Mushroom": "蘑菇",
    "Onion": "洋葱",
    "Pepper": "彩椒",
    "Potato": "土豆",
    "Red-Beet": "甜菜根",
    "Tomato": "番茄",
    "Zucchini": "西葫芦",
}

SYSTEM_PROMPT = (
    "你是冰箱食材识别助手。识别图片中可以可靠确认的食材，只输出符合要求的JSON，不要解释。"
)
USER_PROMPT = "<image>识别图中的食材，只输出JSON。"


def load_classes(dataset_root: Path) -> tuple[dict[int, str], list[dict[str, object]]]:
    coarse_by_id: dict[int, str] = {}
    with (dataset_root / "classes.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            coarse_id = int(row["Coarse Class ID (int)"])
            coarse_name = row["Coarse Class Name (str)"]
            previous = coarse_by_id.setdefault(coarse_id, coarse_name)
            if previous != coarse_name:
                raise ValueError(f"Coarse class ID {coarse_id} has conflicting names")

    missing = sorted(set(coarse_by_id.values()) - COARSE_NAME_ZH.keys())
    if missing:
        raise ValueError(f"Missing Chinese mappings: {missing}")

    mappings = [
        {
            "coarse_class_id": class_id,
            "coarse_class_name": name,
            "normalized_name_zh": COARSE_NAME_ZH[name],
        }
        for class_id, name in sorted(coarse_by_id.items())
    ]
    return coarse_by_id, mappings


def convert_split(dataset_root: Path, split: str, coarse_by_id: dict[int, str]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    with (dataset_root / f"{split}.txt").open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 3:
                raise ValueError(f"Invalid {split}.txt line {line_number}: {raw_line!r}")

            image_path, _fine_id, coarse_id_text = fields
            coarse_id = int(coarse_id_text)
            if coarse_id not in coarse_by_id:
                raise ValueError(f"Unknown coarse ID {coarse_id} at {split}.txt:{line_number}")
            absolute_image = dataset_root / Path(image_path)
            if not absolute_image.is_file():
                raise FileNotFoundError(absolute_image)

            label = COARSE_NAME_ZH[coarse_by_id[coarse_id]]
            answer = {
                "items": [
                    {"name": label, "normalized_name": label, "confidence": 1.0}
                ]
            }
            samples.append(
                {
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": USER_PROMPT},
                        {
                            "role": "assistant",
                            "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
                        },
                    ],
                    "images": [Path(image_path).as_posix()],
                }
            )
    return samples


def dataset_entry(file_name: str) -> dict[str, object]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images", "system": "system"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_by_id, mappings = load_classes(dataset_root)
    counts: dict[str, int] = {}
    dataset_info: dict[str, object] = {}
    for split in ("train", "val", "test"):
        samples = convert_split(dataset_root, split, coarse_by_id)
        file_name = f"grocery_{split}.json"
        (output_dir / file_name).write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dataset_info[f"fridge_grocery_{split}"] = dataset_entry(file_name)
        counts[split] = len(samples)

    (output_dir / "class_mapping_zh.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "classes": len(mappings), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
