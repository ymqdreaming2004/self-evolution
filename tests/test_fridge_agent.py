from pathlib import Path

import pytest

from self_evolution_agent.agents import FridgeAgent
from self_evolution_agent.config import Settings
from self_evolution_agent.db import Database
from self_evolution_agent.repositories import InventoryRepository
from self_evolution_agent.schemas import (
    ImageAttachment,
    IncomingMessage,
    IngredientPrediction,
    Intent,
    PlannedTask,
    RecipeResult,
    TaskKind,
    VisionResult,
)


class FakeChat:
    async def structured(self, *, schema, system, user):
        assert schema is RecipeResult
        return RecipeResult.model_validate(
            {
                "recipes": [
                    {
                        "title": "番茄炒蛋",
                        "inventory_ingredients": ["番茄", "鸡蛋"],
                        "extra_ingredients": ["盐"],
                        "steps": ["炒鸡蛋", "加入番茄"],
                    },
                    {
                        "title": "番茄蛋汤",
                        "inventory_ingredients": ["番茄", "鸡蛋"],
                        "extra_ingredients": ["水", "盐"],
                        "steps": ["煮番茄", "加入蛋液"],
                    },
                ]
            }
        )


class FakeVision:
    def __init__(self, result: VisionResult):
        self.result = result

    async def recognize(self, image_path):
        return self.result


@pytest.fixture
async def database(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        chroma_path=tmp_path / "chroma",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fridge.db'}",
    )
    database = Database(settings)
    await database.initialize()
    yield database
    await database.dispose()


def message(text: str = "") -> IncomingMessage:
    return IncomingMessage(
        event_id="e1",
        message_id="m1",
        chat_id="c1",
        open_id="u1",
        text=text,
    )


async def test_recipe_uses_existing_ingredients_and_renders_multiple_suggestions(
    database: Database,
) -> None:
    async with database.sessions() as session:
        repository = InventoryRepository(session)
        for name in ("番茄", "鸡蛋"):
            await repository.create_from_prediction(
                owner_id="u1",
                prediction=IngredientPrediction(name=name, confidence=1),
                image_key=None,
                model_version="manual",
            )
    agent = FridgeAgent(
        chat=FakeChat(),
        vision=FakeVision(VisionResult(items=[], model_version="v1")),
        sessions=database.sessions,
    )
    task = PlannedTask(
        id="recipe",
        kind=TaskKind.FRIDGE,
        intent=Intent.RECIPE,
        instruction="推荐几个菜",
    )
    result = await agent.run(task, message("推荐几个菜"), "thread-1")
    assert "番茄炒蛋" in result.reply
    assert "番茄蛋汤" in result.reply
    assert "需要补充" in result.reply


async def test_empty_recognition_does_not_create_confirmation(database: Database) -> None:
    agent = FridgeAgent(
        chat=FakeChat(),
        vision=FakeVision(VisionResult(items=[], model_version="v1")),
        sessions=database.sessions,
    )
    task = PlannedTask(
        id="ingest",
        kind=TaskKind.FRIDGE,
        intent=Intent.FRIDGE_INGEST,
        instruction="识别食材",
        requires_confirmation=True,
    )
    incoming = message()
    incoming.images = [ImageAttachment(image_key="img", local_path=str(Path("image.jpg")))]
    result = await agent.run(task, incoming, "thread-1")
    assert "没有识别到" in result.reply
    assert result.effects == []
