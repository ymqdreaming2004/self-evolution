from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from .providers.feishu import FeishuClient, fridge_confirmation_card, inventory_card
from .repositories import (
    DraftRepository,
    InventoryRepository,
    PendingActionRepository,
    SideEffectRepository,
    load_json,
)
from .schemas import AgentEffect, BitableIdea, VisionResult


class EffectExecutor:
    def __init__(self, *, feishu: FeishuClient, sessions: async_sessionmaker):
        self.feishu = feishu
        self.sessions = sessions

    async def execute(
        self, effects: list[AgentEffect], *, open_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        pending: list[dict[str, Any]] = []
        for effect in effects:
            if effect.requires_confirmation:
                action = await self._prepare_confirmation(
                    effect, open_id=open_id, thread_id=thread_id
                )
                pending.append(action)
            else:
                await self._execute_once(effect, open_id=open_id)
        return {"actions": pending, "open_id": open_id} if pending else None

    async def _execute_once(self, effect: AgentEffect, *, open_id: str) -> Any:
        async with self.sessions() as session:
            repository = SideEffectRepository(session)
            record, should_run = await repository.begin(effect.idempotency_key, effect.type)
            if not should_run:
                return load_json(record.response_json) if record.response_json else None
            try:
                if effect.type == "bitable_append":
                    result = await self.feishu.append_idea(
                        BitableIdea.model_validate(effect.payload)
                    )
                elif effect.type == "send_card":
                    result = await self.feishu.send_card(
                        open_id, effect.payload["card"], effect.idempotency_key
                    )
                else:
                    result = await self.feishu.send_text(
                        open_id, effect.payload["text"], effect.idempotency_key
                    )
                await repository.succeed(record, result)
                return result
            except Exception as exc:
                await repository.fail(record, exc)
                raise

    async def _prepare_confirmation(
        self, effect: AgentEffect, *, open_id: str, thread_id: str
    ) -> dict[str, Any]:
        async with self.sessions() as session:
            effects = SideEffectRepository(session)
            record, should_run = await effects.begin(effect.idempotency_key, effect.type)
            if not should_run and record.response_json:
                return load_json(record.response_json)
            pending_repository = PendingActionRepository(session)
            action = await pending_repository.create(
                thread_id=thread_id,
                owner_id=open_id,
                kind=effect.type,
                payload=effect.payload,
            )
            if effect.type == "fridge_confirmation":
                result = VisionResult.model_validate(effect.payload["result"])
                card = fridge_confirmation_card(
                    action_id=action.action_id,
                    thread_id=thread_id,
                    draft_id=effect.payload["draft_id"],
                    result=result,
                )
            else:
                card = mutation_confirmation_card(action.action_id, thread_id, effect.payload)
            try:
                await self.feishu.send_card(open_id, card, effect.idempotency_key)
                response = {"action_id": action.action_id, "kind": effect.type}
                await effects.succeed(record, response)
                return response
            except Exception as exc:
                await effects.fail(record, exc)
                raise

    async def apply_confirmation(self, command: dict[str, Any]) -> str:
        open_id = command["open_id"]
        action_id = command["action_id"]
        decision = command.get("action", "cancel")
        async with self.sessions() as session:
            pending_repository = PendingActionRepository(session)
            pending = await pending_repository.consume(action_id, open_id)
            if pending is None:
                return "该操作已处理、已过期或无权执行。"
            if decision == "cancel":
                await pending_repository.finish(pending, "cancelled")
                return "操作已取消。"
            payload = load_json(pending.payload_json)
            if pending.kind == "fridge_confirmation":
                try:
                    message = await self._commit_fridge(
                        session, payload, command.get("values", {}), open_id
                    )
                except ValueError as exc:
                    await pending_repository.finish(pending, "pending")
                    return f"无法入库：{exc}。请补全后重新确认。"
            else:
                message = await self._commit_mutation(session, payload, open_id)
            await pending_repository.finish(pending)
            return message

    async def _commit_fridge(
        self, session: Any, payload: dict[str, Any], values: dict[str, Any], open_id: str
    ) -> str:
        result = VisionResult.model_validate(payload["result"])
        confirmed_items = []
        for index, item in enumerate(result.items):
            raw_name = values.get(f"name_{index}", item.name)
            name_value = "" if raw_name is None else str(raw_name).strip()
            if not name_value:
                continue
            original_name = item.name
            item.name = name_value
            if name_value != original_name:
                item.normalized_name = name_value.lower()
            confirmed_items.append(item)
        if not confirmed_items:
            raise ValueError("至少保留一种食材；不需要录入时请选择取消")
        result.items = confirmed_items
        drafts = DraftRepository(session)
        draft = await drafts.get(payload["draft_id"], open_id)
        if draft is None or draft.status != "pending":
            raise ValueError("识别草稿不存在或已经提交")
        inventory = InventoryRepository(session)
        for item in result.items:
            await inventory.create_from_prediction(
                owner_id=open_id,
                prediction=item,
                image_key=draft.image_key,
                model_version=result.model_version,
                commit=False,
            )
        await drafts.confirm(draft, result)
        names = "、".join(item.name for item in result.items)
        return f"已更新现有食材：{names}。"

    async def _commit_mutation(self, session: Any, payload: dict[str, Any], open_id: str) -> str:
        repository = InventoryRepository(session)
        item_name = str(payload.get("item_name", "")).strip()
        item = await repository.find_active_by_name(open_id, item_name)
        if item is None:
            return f"现有食材中没有找到“{item_name}”。"
        await repository.consume(item)
        return f"已标记用完：{item.name}。"


def mutation_confirmation_card(
    action_id: str, thread_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    value = {"action_id": action_id, "thread_id": thread_id}
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "red",
            "title": {"tag": "plain_text", "content": "确认食材已用完"},
        },
        "elements": [
            {
                "tag": "markdown",
                "content": f"确认将 **{payload.get('item_name', '')}** 标记为已用完？",
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "type": "danger",
                        "text": {"tag": "plain_text", "content": "确认"},
                        "value": {**value, "action": "confirm"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "取消"},
                        "value": {**value, "action": "cancel"},
                    },
                ],
            },
        ],
    }


def query_card_effect(data: dict[str, Any], *, message_id: str, task_id: str) -> AgentEffect:
    class Item:
        def __init__(self, value: dict[str, Any]):
            self.__dict__.update(value)

    card = inventory_card([Item(item) for item in data["items"]], data["title"])
    return AgentEffect(
        type="send_card",
        payload={"card": card},
        idempotency_key=f"inventory-card:{message_id}:{task_id}",
    )
