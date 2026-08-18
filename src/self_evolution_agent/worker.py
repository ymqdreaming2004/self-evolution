from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .agents import FridgeAgent, GeneralChatAgent, InspirationAgent
from .config import get_settings
from .db import Database, recover_stale_jobs
from .effects import EffectExecutor
from .long_connection import FeishuLongConnection
from .planner import Planner
from .providers.chat import ChatProvider
from .providers.feishu import FeishuClient
from .providers.obsidian import ObsidianVault
from .providers.vision import VisionProvider
from .providers.web import WebContentFetcher
from .rag import KnowledgeStore
from .repositories import JobRepository, load_json
from .schemas import IncomingMessage
from .workflow import AgentWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Worker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.database = Database(self.settings)
        self.feishu = FeishuClient(self.settings)
        self.vision = VisionProvider(self.settings)
        self.web = WebContentFetcher(self.settings)
        self.checkpoint_connection: aiosqlite.Connection | None = None
        self.workflow: AgentWorkflow | None = None
        self.listener_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        await self.database.initialize()
        async with self.database.sessions() as session:
            await recover_stale_jobs(session)
        if self.settings.feishu_bitable_app_token and self.settings.feishu_bitable_table_id:
            valid, missing = await self.feishu.validate_bitable_schema()
            if not valid:
                raise RuntimeError(f"Bitable schema is missing fields: {', '.join(missing)}")
        checkpoint_path = self.settings.data_dir / "langgraph_checkpoints.db"
        self.checkpoint_connection = await aiosqlite.connect(str(checkpoint_path))
        checkpointer = AsyncSqliteSaver(self.checkpoint_connection)
        await checkpointer.setup()
        chat = ChatProvider(self.settings)
        store = KnowledgeStore(self.settings)
        self.workflow = AgentWorkflow(
            planner=Planner(chat),
            inspiration=InspirationAgent(
                chat=chat,
                store=store,
                web=self.web,
                vault=ObsidianVault(self.settings.obsidian_vault_path),
            ),
            fridge=FridgeAgent(chat=chat, vision=self.vision, sessions=self.database.sessions),
            general=GeneralChatAgent(chat),
            effects=EffectExecutor(feishu=self.feishu, sessions=self.database.sessions),
            checkpointer=checkpointer,
        )

    async def run_forever(self) -> None:
        await self.initialize()
        assert self.workflow is not None
        logger.info("worker started")
        if self.settings.feishu_event_transport == "long_connection":
            listener = FeishuLongConnection(self.settings, self.database)
            self.listener_task = asyncio.create_task(listener.run())
        try:
            while True:
                handled = await self.run_once()
                if not handled:
                    await asyncio.sleep(self.settings.worker_poll_seconds)
        finally:
            if self.listener_task:
                self.listener_task.cancel()
                await asyncio.gather(self.listener_task, return_exceptions=True)
            await self.close()

    async def run_once(self) -> bool:
        assert self.workflow is not None
        async with self.database.sessions() as session:
            repository = JobRepository(session)
            job = await repository.claim_next()
            if job is None:
                return False
            try:
                payload = load_json(job.payload_json)
                if job.kind == "message":
                    message = IncomingMessage.model_validate(payload)
                    await self._download_images(message)
                    await self.workflow.invoke_message(message, thread_id=message.message_id)
                elif job.kind == "card_action":
                    await self.workflow.resume(payload["thread_id"], payload)
                else:
                    raise ValueError(f"unknown job kind: {job.kind}")
                await repository.complete(job)
            except Exception as exc:
                logger.exception("job %s failed", job.id)
                await repository.fail(job, exc)
            return True

    async def _download_images(self, message: IncomingMessage) -> None:
        for image in message.images:
            safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", image.image_key)[:100]
            destination = (
                Path(self.settings.data_dir) / "images" / f"{message.message_id}_{safe_key}.img"
            )
            await self.feishu.download_image(message.message_id, image.image_key, destination)
            image.local_path = str(destination)

    async def close(self) -> None:
        await self.feishu.close()
        await self.vision.close()
        await self.web.close()
        if self.checkpoint_connection:
            await self.checkpoint_connection.close()
        await self.database.dispose()


async def main() -> None:
    await Worker().run_forever()


if __name__ == "__main__":
    asyncio.run(main())
