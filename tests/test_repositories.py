import pytest

from self_evolution_agent.config import Settings
from self_evolution_agent.db import Database
from self_evolution_agent.repositories import InventoryRepository, JobRepository
from self_evolution_agent.schemas import IngredientPrediction


@pytest.fixture
async def database(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        chroma_path=tmp_path / "chroma",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
    )
    db = Database(settings)
    await db.initialize()
    yield db
    await db.dispose()


async def test_job_enqueue_is_idempotent(database: Database) -> None:
    async with database.sessions() as session:
        repository = JobRepository(session)
        first, created_first = await repository.enqueue(
            kind="message", payload={"x": 1}, idempotency_key="same"
        )
        second, created_second = await repository.enqueue(
            kind="message", payload={"x": 2}, idempotency_key="same"
        )
    assert created_first is True
    assert created_second is False
    assert first.id == second.id


async def test_inventory_lifecycle(database: Database) -> None:
    prediction = IngredientPrediction(
        name="牛奶",
        confidence=0.98,
    )
    async with database.sessions() as session:
        repository = InventoryRepository(session)
        item = await repository.create_from_prediction(
            owner_id="user", prediction=prediction, image_key="img", model_version="v1"
        )
        assert (await repository.list_active("user"))[0].id == item.id
        await repository.consume(item)
        assert await repository.list_active("user") == []


async def test_duplicate_active_ingredient_is_merged(database: Database) -> None:
    prediction = IngredientPrediction(name="西红柿", normalized_name="番茄", confidence=0.9)
    async with database.sessions() as session:
        repository = InventoryRepository(session)
        first = await repository.create_from_prediction(
            owner_id="user", prediction=prediction, image_key="img-1", model_version="v1"
        )
        second = await repository.create_from_prediction(
            owner_id="user", prediction=prediction, image_key="img-2", model_version="v2"
        )
        items = await repository.list_active("user")
    assert first.id == second.id
    assert len(items) == 1
    assert items[0].image_key == "img-2"
    assert items[0].model_version == "v2"
