from datetime import date

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
        quantity=1,
        unit="盒",
        expiry_date=date.today(),
        date_source="printed",
        confidence=0.98,
    )
    async with database.sessions() as session:
        repository = InventoryRepository(session)
        item = await repository.create_from_prediction(
            owner_id="user", prediction=prediction, image_key="img", model_version="v1"
        )
        assert (await repository.list_expiring("user"))[0].id == item.id
        await repository.consume(item)
        assert await repository.list_active("user") == []
