from __future__ import annotations

import asyncio
import re

from sqlalchemy.ext.asyncio import async_sessionmaker

from .providers.chat import ChatProvider, json_for_prompt
from .providers.vision import VisionProvider
from .providers.web import WebContentFetcher
from .rag import KnowledgeStore
from .repositories import DraftRepository, InventoryRepository, record_unhandled
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

    def __init__(self, *, chat: ChatProvider, store: KnowledgeStore, web: WebContentFetcher):
        self.chat = chat
        self.store = store
        self.web = web

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
        titles = []
        for fallback_title, content, source in sources:
            metadata = await self._metadata(content)
            title = metadata.title or fallback_title
            chunks = await asyncio.to_thread(
                self.store.add_document,
                content=content,
                title=title,
                tags=metadata.tags,
                source=source,
                created_at=message.received_at,
            )
            counts += len(chunks)
            titles.append(title)
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply=f"已沉淀 {len(sources)} 条知识，共 {counts} 个片段：" + "、".join(titles),
        )

    async def _query(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        hits = await asyncio.to_thread(self.store.search, message.text)
        if not hits:
            return AgentResult(
                task_id=task.id, intent=task.intent, reply="知识库中没有找到相关内容。"
            )
        context = "\n\n".join(
            f"[{index}] {hit.title}\n{hit.content}\n来源：{hit.source}"
            for index, hit in enumerate(hits, start=1)
        )
        try:
            answer = await self.chat.text(
                system="只依据给定资料回答。每个关键结论使用 [序号] 引用；资料不足时明确说明。",
                user=f"问题：{message.text}\n\n资料：\n{context}",
            )
        except Exception:
            answer = "找到以下相关内容：\n" + "\n".join(
                f"[{index}] {hit.title} - {hit.source}" for index, hit in enumerate(hits, start=1)
            )
        return AgentResult(task_id=task.id, intent=task.intent, reply=answer)

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
            reply=f"已整理为{idea.type}：{idea.title}",
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
            repository = InventoryRepository(session)
            if any(word in message.text for word in ("临期", "过期")):
                items = await repository.list_expiring(message.open_id)
                title = "临期与过期食材"
            else:
                items = await repository.list_active(message.open_id)
                title = "冰箱库存"
        data = {
            "title": title,
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "expiry_date": item.expiry_date.isoformat() if item.expiry_date else None,
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
                task_id=task.id, intent=task.intent, reply="冰箱库存为空，暂时无法推荐菜谱。"
            )
        inventory = [
            {
                "name": item.name,
                "quantity": item.quantity,
                "unit": item.unit,
                "expiry_date": item.expiry_date,
            }
            for item in items
        ]
        recipe = await self.chat.structured(
            schema=RecipeResult,
            system="生成优先消耗临期食材的菜谱。允许少量外购辅料，必须区分两类食材。",
            user=json_for_prompt({"request": message.text, "inventory": inventory}),
        )
        lines = [f"**{recipe.title}**（{recipe.servings} 人份）", "", "库存食材："]
        lines.extend(f"- {item}" for item in recipe.inventory_ingredients)
        if recipe.extra_ingredients:
            lines.append("\n需要补充：")
            lines.extend(f"- {item}" for item in recipe.extra_ingredients)
        lines.append("\n步骤：")
        lines.extend(f"{index}. {step}" for index, step in enumerate(recipe.steps, start=1))
        if recipe.notes:
            lines.append(f"\n提示：{recipe.notes}")
        return AgentResult(task_id=task.id, intent=task.intent, reply="\n".join(lines))

    async def _mutation(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        mutation = await self.chat.structured(
            schema=InventoryMutation,
            system="从命令提取库存操作。item_id 必须来自用户原文，不得猜测。",
            user=message.text,
        )
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply="该库存操作需要确认。",
            effects=[
                AgentEffect(
                    type="inventory_mutation",
                    payload=mutation.model_dump(mode="json"),
                    idempotency_key=f"inventory-mutation:{message.message_id}:{task.id}",
                    requires_confirmation=True,
                )
            ],
        )


class PlaceholderAgent:
    def __init__(self, sessions: async_sessionmaker):
        self.sessions = sessions

    async def run(self, task: PlannedTask, message: IncomingMessage) -> AgentResult:
        try:
            async with self.sessions() as session:
                await record_unhandled(
                    session, message.open_id, message.message_id, message.text, task.intent.value
                )
        except Exception:
            pass
        return AgentResult(
            task_id=task.id,
            intent=task.intent,
            reply="目前我只支持知识/灵感沉淀、知识检索，以及冰箱库存和菜谱管理。这个需求已记录。",
        )
