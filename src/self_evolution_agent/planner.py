from __future__ import annotations

import logging
import re
from uuid import uuid4

from .providers.chat import ChatProvider, json_for_prompt
from .schemas import ExecutionPlan, IncomingMessage, Intent, PlannedTask, TaskKind

PLANNER_SYSTEM = """你是个人智能中枢的 Planner。必须使用 Plan-and-Execute 模式拆分输入。
intent 只允许 knowledge_store、knowledge_query、inspiration、fridge_ingest、fridge_query、
fridge_mutate、recipe、placeholder。kind 只允许 inspiration、fridge、placeholder。
为每个任务指定与 intent 对应的 kind：knowledge_store、knowledge_query、inspiration 使用
inspiration；fridge_ingest、fridge_query、fridge_mutate、recipe 使用 fridge；placeholder 使用
placeholder。图片消息必须包含 fridge_ingest，且 fridge_ingest 和 fridge_mutate 必须
requires_confirmation=true。任务必须彼此独立，dependencies 保持为空。无法支持的请求使用
placeholder；不要虚构能力。输出必须严格符合 JSON schema。"""

logger = logging.getLogger(__name__)

_INTENT_KINDS = {
    Intent.KNOWLEDGE_STORE: TaskKind.INSPIRATION,
    Intent.KNOWLEDGE_QUERY: TaskKind.INSPIRATION,
    Intent.INSPIRATION: TaskKind.INSPIRATION,
    Intent.FRIDGE_INGEST: TaskKind.FRIDGE,
    Intent.FRIDGE_QUERY: TaskKind.FRIDGE,
    Intent.FRIDGE_MUTATE: TaskKind.FRIDGE,
    Intent.RECIPE: TaskKind.FRIDGE,
    Intent.PLACEHOLDER: TaskKind.PLACEHOLDER,
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
        """Reject unsafe or unsupported model plans before dispatching agents in parallel."""
        if not plan.tasks:
            raise ValueError("planner returned no tasks")
        if plan.intent != plan.tasks[0].intent:
            raise ValueError("primary intent must match the first task")

        task_ids = [task.id for task in plan.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("planner task IDs must be unique")
        if plan.requires_confirmation != any(task.requires_confirmation for task in plan.tasks):
            raise ValueError("plan confirmation flag does not match task flags")

        for task in plan.tasks:
            if not task.instruction.strip():
                raise ValueError("planner task instruction must not be empty")
            if task.kind != _INTENT_KINDS[task.intent]:
                raise ValueError(f"task kind does not match intent: {task.intent.value}")
            if task.dependencies:
                raise ValueError("task dependencies are not supported by the parallel workflow")
            if task.intent in _CONFIRMATION_INTENTS and not task.requires_confirmation:
                raise ValueError(f"confirmation is required for {task.intent.value}")

        task_intents = {task.intent for task in plan.tasks}
        if message.images and Intent.FRIDGE_INGEST not in task_intents:
            raise ValueError("image messages must include fridge_ingest")
        if not message.images and Intent.FRIDGE_INGEST in task_intents:
            raise ValueError("fridge_ingest requires an image")

    @staticmethod
    def heuristic_plan(message: IncomingMessage) -> ExecutionPlan:
        text = message.text.strip().lower()
        intents: list[Intent] = []
        if message.images:
            intents.append(Intent.FRIDGE_INGEST)
        if any(word in text for word in ("菜谱", "做什么菜", "吃什么")):
            intents.append(Intent.RECIPE)
        elif any(word in text for word in ("临期", "过期", "冰箱里", "库存", "食材列表")):
            intents.append(Intent.FRIDGE_QUERY)
        elif re.search(r"(消耗|吃掉|删除|移除|修改).{0,40}[0-9a-f]{8}", text):
            intents.append(Intent.FRIDGE_MUTATE)
        if any(word in text for word in ("知识库", "我之前", "查找笔记", "搜索知识")):
            intents.append(Intent.KNOWLEDGE_QUERY)
        elif any(word in text for word in ("灵感", "点子", "todo", "待办")):
            intents.append(Intent.INSPIRATION)
        elif (
            message.urls
            or len(text) >= 180
            or any(word in text for word in ("收藏", "保存知识", "记录文章", "沉淀"))
        ):
            intents.append(Intent.KNOWLEDGE_STORE)
        if not intents:
            intents.append(Intent.PLACEHOLDER)
        intents = list(dict.fromkeys(intents))
        tasks: list[PlannedTask] = []
        for intent in intents:
            kind = (
                TaskKind.FRIDGE
                if intent.value.startswith("fridge") or intent == Intent.RECIPE
                else TaskKind.INSPIRATION
                if intent.value.startswith("knowledge") or intent == Intent.INSPIRATION
                else TaskKind.PLACEHOLDER
            )
            tasks.append(
                PlannedTask(
                    id=str(uuid4()),
                    kind=kind,
                    intent=intent,
                    instruction=message.text or intent.value,
                    requires_confirmation=intent in _CONFIRMATION_INTENTS,
                )
            )
        return ExecutionPlan(
            intent=intents[0],
            tasks=tasks,
            requires_confirmation=any(task.requires_confirmation for task in tasks),
            rationale="本地规则路由",
        )
