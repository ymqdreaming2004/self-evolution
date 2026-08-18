from __future__ import annotations

import asyncio
import re
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker

from .providers.chat import ChatProvider, json_for_prompt
from .providers.obsidian import ObsidianVault
from .providers.vision import VisionProvider
from .providers.web import WebContentFetcher
from .rag import KnowledgeStore
from .repositories import DraftRepository, InventoryRepository
from .responses import (
    inspiration_recorded,
    knowledge_not_found,
    knowledge_search_results,
    knowledge_stored,
)
from .schemas import (
    AgentEffect,
    AgentResult,
    BitableIdea,
    ContentMetadata,
    IncomingMessage,
    InventoryMutation,
    PlannedTask,
    RecipeResult,
)


class InspirationAgent:
    """ReAct-style specialist with a fixed knowledge/idea tool whitelist."""

    def __init__(
        self,
        *,
        chat: ChatProvider,
        store: KnowledgeStore,
        web: WebContentFetcher,
        vault: ObsidianVault,
    ):
        self.chat = chat
        self.store = store
        self.web = web
        self.vault = vault

    async def run(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        try:
            if task.intent.value == "knowledge_store":
                return await self._store(task, message)
            if task.intent.value == "knowledge_query":
                return await self._query(task, message)
            return await self._idea(task, message)
        except Exception as exc:
            return AgentResult(task_id=task.id, intent=task.intent, error=str(exc))

    async def _metadata(self, content: str, default_kind: str = "知识") -> ContentMetadata:
        try:
            return await self.chat.structured(
                schema=ContentMetadata,
                system="提取简洁标题、最多 8 个中文标签和内容类型。不要改写原文。",
                user=content[:12000],
            )
        except Exception:
            title = re.sub(r"\s+", " ", content).strip()[:40] or "未命名内容"
            return ContentMetadata(title=title, tags=[], kind=default_kind)

    async def _store(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        sources: list[tuple[str, str, str]] = []
        for url in message.urls:
            title, content = await self.web.fetch(str(url))
            sources.append((title, content, str(url)))
        if message.text.strip():
            sources.append(("飞书消息", message.text, f"feishu:{message.message_id}"))
        counts = 0
        for fallback_title, content, source in sources:
            metadata = await self._metadata(content)
            title = metadata.title or fallback_title
            document_id = str(uuid4())
            note = await asyncio.to_thread(
                self.vault.write_knowledge,
                document_id=document_id,
                title=title,
                content=content,
                tags=metadata.tags,
                source=source,
                created_at=message.received_at,
            )
            chunks = await asyncio.to_thread(
                self.store.add_document,
                content=content,
                title=title,
                tags=metadata.tags,
                source=source,
                note_link=note.link,
                created_at=message.received_at,
                document_id=document_id,
            )
            counts += len(chunks)
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=knowledge_stored(document_count=len(sources), chunk_count=counts),
        )

    async def _query(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        hits = await asyncio.to_thread(self.store.search, message.text)
        if not hits:
            return AgentResult(
                task_id=task.id, intent=task.intent, reply=knowledge_not_found()
            )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=knowledge_search_results(hits),
        )

    async def _idea(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        metadata = await self._metadata(message.text, "灵感")
        idea = BitableIdea(
            title=metadata.title,
            content=message.text,
            type="TODO" if metadata.kind == "TODO" else "灵感",
            tags=metadata.tags,
            source_message=message.message_id,
            created_at=message.received_at,
        )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=inspiration_recorded(idea_type=idea.type, title=idea.title),
            effects=[
                AgentEffect(
                    type="bitable_append",
                    payload=idea.model_dump(mode="json"),
                    idempotency_key=f"bitable:{message.message_id}:{task.id}",
                )
            ],
        )


class FridgeAgent:
    """ReAct-style specialist restricted to vision, inventory and recipe tools."""

    def __init__(self, *, chat: ChatProvider, vision: VisionProvider, sessions: async_sessionmaker):
        self.chat = chat
        self.vision = vision
        self.sessions = sessions

    async def run(self, task: PlannedTask, message: IncomingMessage, thread_id: str) -> AgentResult:
        try:
            if task.intent.value == "fridge_ingest":
                return await self._ingest(task, message, thread_id)
            if task.intent.value == "fridge_query":
                return await self._query(task, message)
            if task.intent.value == "recipe":
                return await self._recipe(task, message)
            return await self._mutation(task, message)
        except Exception as exc:
            return AgentResult(task_id=task.id, intent=task.intent, error=str(exc))

    async def _ingest(
        self, task: PlannedTask, message: IncomingMessage, thread_id: str
    ) -> AgentResult:
        if not message.images or not message.images[0].local_path:
            raise ValueError("食材图片尚未下载")
        image = message.images[0]
        result = await self.vision.recognize(image.local_path)
        if not result.items:
            return AgentResult(
                task_id=task.id,
                intent=task.intent,
                reply="没有识别到可以确认的食材，请换一张更清晰的照片重试。",
            )
        async with self.sessions() as session:
            draft = await DraftRepository(session).create(
                owner_id=message.open_id,
                message_id=message.message_id,
                thread_id=thread_id,
                image_key=image.image_key,
                image_path=image.local_path,
                result=result,
            )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply="图片识别完成，请确认后入库。",
            data={"draft_id": draft.id},
            effects=[
                AgentEffect(
                    type="fridge_confirmation",
                    payload={
                        "draft_id": draft.id,
                        "thread_id": thread_id,
                        "result": result.model_dump(mode="json"),
                    },
                    idempotency_key=f"fridge-confirm:{draft.id}",
                    requires_confirmation=True,
                )
            ],
        )

    async def _query(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        async with self.sessions() as session:
            items = await InventoryRepository(session).list_active(message.open_id)
        data = {
            "title": "现有食材",
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                }
                for item in items
            ],
        }
        return AgentResult(task_id=task.id, intent=task.intent, data=data)

    async def _recipe(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        async with self.sessions() as session:
            items = await InventoryRepository(session).list_active(message.open_id)
        if not items:
            return AgentResult(
                task_id=task.id, intent=task.intent, reply="现有食材清单为空，暂时无法推荐菜。"
            )
        inventory = [
            {"name": item.name}
            for item in items
        ]
        recipe = await self.chat.structured(
            schema=RecipeResult,
            system=(
                "根据现有食材推荐 3 道简单可做的菜。只能把清单中的食材列为现有食材；"
                "允许少量外购辅料，并且必须明确区分现有食材和需要补充的食材。"
            ),
            user=json_for_prompt({"request": message.text, "inventory": inventory}),
        )
        lines = []
        for recipe_index, suggestion in enumerate(recipe.recipes, start=1):
            if lines:
                lines.append("\n---\n")
            lines.extend([f"### {recipe_index}. {suggestion.title}", "", "现有食材："])
            lines.extend(f"- {item}" for item in suggestion.inventory_ingredients)
            if suggestion.extra_ingredients:
                lines.append("\n需要补充：")
                lines.extend(f"- {item}" for item in suggestion.extra_ingredients)
            lines.append("\n步骤：")
            lines.extend(
                f"{index}. {step}" for index, step in enumerate(suggestion.steps, start=1)
            )
        return AgentResult(task_id=task.id, intent=task.intent, reply="\n".join(lines))

    async def _mutation(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        mutation = await self.chat.structured(
            schema=InventoryMutation,
            system="从用户命令中提取已经用完的食材名称。不要添加原文中没有出现的食材。",
            user=message.text,
        )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=f"需要确认是否将“{mutation.item_name}”标记为已用完。",
            effects=[
                AgentEffect(
                    type="inventory_mutation",
                    payload=mutation.model_dump(mode="json"),
                    idempotency_key=f"inventory-mutation:{message.message_id}:{task.id}",
                    requires_confirmation=True,
                )
            ],
        )


class GeneralChatAgent:
    """Fallback conversation agent with no data-writing dependencies."""

    def __init__(self, chat: ChatProvider):
        self.chat = chat

    async def run(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        reply = await self.chat.text(
            system=(
                "你是个人助手的通用对话 Agent。仅回答普通对话；不要声称已写入知识库、"
                "灵感表或现有食材清单，也不要执行或承诺执行外部操作。"
            ),
            user=message.text,
        )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=reply,
        )
