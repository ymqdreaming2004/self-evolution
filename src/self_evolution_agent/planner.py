from __future__ import annotations

import logging
import re
from uuid import uuid4

from .providers.chat import ChatProvider, json_for_prompt
from .schemas import ExecutionPlan, IncomingMessage, Intent, PlannedTask, TaskKind

PLANNER_SYSTEM = """你是个人智能中枢的 Planner。你只能输出一个主任务，先识别意图再路由。
intent 只允许 knowledge_store、knowledge_query、inspiration、fridge_ingest、fridge_query、
fridge_mutate、recipe、general_chat。kind 只允许 inspiration、fridge、general。
knowledge_store、knowledge_query、inspiration 使用 inspiration；fridge_ingest、fridge_query、
fridge_mutate、recipe 使用 fridge；其他普通对话使用 general。图片消息必须路由为
fridge_ingest；fridge_ingest 和 fridge_mutate 必须 requires_confirmation=true。冰箱 Agent
只负责识别食材、维护现有食材清单和基于清单推荐菜，不处理日期、临期、数量或采购。任务
dependencies 必须为空。general_chat 只能对话，不得声称可以写入知识、灵感或库存。
输出必须严格符合 JSON schema。"""

logger = logging.getLogger(__name__)

_INTENT_KINDS = {
    Intent.KNOWLEDGE_STORE: TaskKind.INSPIRATION,
    Intent.KNOWLEDGE_QUERY: TaskKind.INSPIRATION,
    Intent.INSPIRATION: TaskKind.INSPIRATION,
    Intent.FRIDGE_INGEST: TaskKind.FRIDGE,
    Intent.FRIDGE_QUERY: TaskKind.FRIDGE,
    Intent.FRIDGE_MUTATE: TaskKind.FRIDGE,
    Intent.RECIPE: TaskKind.FRIDGE,
    Intent.GENERAL_CHAT: TaskKind.GENERAL,
}
_CONFIRMATION_INTENTS = {Intent.FRIDGE_INGEST, Intent.FRIDGE_MUTATE}


class Planner:
    def __init__(self, chat: ChatProvider):
        self.chat = chat

    async def plan(self, message: IncomingMessage) -> ExecutionPlan:
        prompt = json_for_prompt(
            {
                "text": message.text,
                "image_count": len(message.images),
                "urls": [str(url) for url in message.urls],
            }
        )
        try:
            plan = await self.chat.structured(
                schema=ExecutionPlan,
                system=PLANNER_SYSTEM,
                user=prompt,
                model=self.chat.settings.planner_model,
            )
            self.validate_plan(plan, message)
            return plan
        except Exception as exc:
            logger.warning("planner model output rejected; using heuristic fallback: %s", exc)
        return self.heuristic_plan(message)

    @staticmethod
    def validate_plan(plan: ExecutionPlan, message: IncomingMessage) -> None:
        """Reject unsafe plans before dispatching the single routed agent."""
        if len(plan.tasks) != 1:
            raise ValueError("planner must return exactly one primary task")
        if plan.intent != plan.tasks[0].intent:
            raise ValueError("primary intent must match the task")
        if plan.requires_confirmation != plan.tasks[0].requires_confirmation:
            raise ValueError("plan confirmation flag does not match task flags")

        task = plan.tasks[0]
        if not task.instruction.strip():
            raise ValueError("planner task instruction must not be empty")
        if task.kind != _INTENT_KINDS[task.intent]:
            raise ValueError(f"task kind does not match intent: {task.intent.value}")
        if task.dependencies:
            raise ValueError("task dependencies are not supported")
        if task.intent in _CONFIRMATION_INTENTS and not task.requires_confirmation:
            raise ValueError(f"confirmation is required for {task.intent.value}")
        if message.images and task.intent != Intent.FRIDGE_INGEST:
            raise ValueError("image messages must route to fridge_ingest")
        if not message.images and task.intent == Intent.FRIDGE_INGEST:
            raise ValueError("fridge_ingest requires an image")

    @staticmethod
    def heuristic_plan(message: IncomingMessage) -> ExecutionPlan:
        text = message.text.strip().lower()
        if message.images:
            intent = Intent.FRIDGE_INGEST
        elif any(word in text for word in ("菜谱", "做什么菜", "吃什么")):
            intent = Intent.RECIPE
        elif any(word in text for word in ("冰箱里", "现有食材", "有什么食材", "食材列表")):
            intent = Intent.FRIDGE_QUERY
        elif re.search(r"(用完|吃完|没有了|移除|删除)", text):
            intent = Intent.FRIDGE_MUTATE
        elif any(word in text for word in ("知识库", "我之前", "查找笔记", "搜索知识")):
            intent = Intent.KNOWLEDGE_QUERY
        elif any(word in text for word in ("灵感", "点子", "todo", "待办")):
            intent = Intent.INSPIRATION
        elif (
            message.urls
            or len(text) >= 180
            or any(word in text for word in ("收藏", "保存知识", "记录文章", "沉淀"))
        ):
            intent = Intent.KNOWLEDGE_STORE
        else:
            intent = Intent.GENERAL_CHAT

        task = PlannedTask(
            id=str(uuid4()),
            kind=_INTENT_KINDS[intent],
            intent=intent,
            instruction=message.text or intent.value,
            requires_confirmation=intent in _CONFIRMATION_INTENTS,
        )
        return ExecutionPlan(
            intent=intent,
            tasks=[task],
            requires_confirmation=task.requires_confirmation,
            rationale="本地规则路由",
        )
