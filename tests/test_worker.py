from self_evolution_agent.config import Settings
from self_evolution_agent.worker import Worker


async def test_worker_initializes_with_sqlite_checkpoint(tmp_path, monkeypatch) -> None:
    settings = Settings(
        data_dir=tmp_path,
        chroma_path=tmp_path / "chroma",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'worker.db'}",
        feishu_bitable_app_token="",
        feishu_bitable_table_id="",
    )
    monkeypatch.setattr("self_evolution_agent.worker.get_settings", lambda: settings)
    worker = Worker()
    await worker.initialize()
    try:
        assert worker.workflow is not None
        assert (tmp_path / "langgraph_checkpoints.db").exists()
    finally:
        await worker.close()
