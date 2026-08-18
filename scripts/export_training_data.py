from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from self_evolution_agent.config import get_settings
from self_evolution_agent.db import Database, RecognitionDraft
from self_evolution_agent.repositories import load_json
from self_evolution_agent.schemas import VisionResult


async def export(output: Path) -> int:
    settings = get_settings()
    database = Database(settings)
    await database.initialize()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    async with database.sessions() as session, output.open("w", encoding="utf-8") as handle:
        result = await session.execute(
            select(RecognitionDraft).where(
                RecognitionDraft.status == "confirmed",
                RecognitionDraft.corrected_json.is_not(None),
                RecognitionDraft.image_path.is_not(None),
            )
        )
        for draft in result.scalars():
            corrected = VisionResult.model_validate(load_json(draft.corrected_json or "{}"))
            row = {
                "sample_id": draft.id,
                "image_path": draft.image_path,
                "model_version": draft.model_version,
                "prediction": load_json(draft.prediction_json),
                "target": {
                    "items": [item.model_dump(mode="json") for item in corrected.items]
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    await database.dispose()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/training/confirmed.jsonl"))
    args = parser.parse_args()
    count = asyncio.run(export(args.output))
    print(f"exported {count} samples to {args.output}")


if __name__ == "__main__":
    main()
