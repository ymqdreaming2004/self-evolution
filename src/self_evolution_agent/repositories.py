from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .db import (
    InventoryItem,
    Job,
    PendingAction,
    ProcessedEvent,
    RecognitionDraft,
    SideEffect,
    UnhandledIntent,
    utcnow,
)
from .schemas import IngredientPrediction, VisionResult


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def load_json(value: str) -> Any:
    return json.loads(value)


class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str,
        max_attempts: int = 5,
    ) -> tuple[Job | None, bool]:
        job = Job(
            kind=kind,
            payload_json=dump_json(payload),
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        self.session.add(job)
        try:
            await self.session.commit()
            await self.session.refresh(job)
            return job, True
        except IntegrityError:
            await self.session.rollback()
            existing = await self.session.scalar(
                select(Job).where(Job.idempotency_key == idempotency_key)
            )
            return existing, False

    async def claim_next(self) -> Job | None:
        job = await self.session.scalar(
            select(Job)
            .where(Job.status == "queued", Job.available_at <= utcnow())
            .order_by(Job.created_at)
            .limit(1)
        )
        if job is None:
            return None
        job.status = "running"
        job.locked_at = utcnow()
        job.attempts += 1
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def complete(self, job: Job) -> None:
        job.status = "completed"
        job.locked_at = None
        await self.session.commit()

    async def fail(self, job: Job, error: Exception) -> None:
        job.last_error = str(error)[:4000]
        job.locked_at = None
        if job.attempts >= job.max_attempts:
            job.status = "failed"
        else:
            job.status = "queued"
            delay = min(300, 2 ** max(job.attempts - 1, 0))
            job.available_at = utcnow() + timedelta(seconds=delay)
        await self.session.commit()


class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def mark_once(self, event_id: str) -> bool:
        self.session.add(ProcessedEvent(event_id=event_id))
        try:
            await self.session.commit()
            return True
        except IntegrityError:
            await self.session.rollback()
            return False


class InventoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_from_prediction(
        self,
        *,
        owner_id: str,
        prediction: IngredientPrediction,
        image_key: str | None,
        model_version: str | None,
        commit: bool = True,
    ) -> InventoryItem:
        existing = await self.session.scalar(
            select(InventoryItem).where(
                InventoryItem.owner_id == owner_id,
                InventoryItem.normalized_name == (
                    prediction.normalized_name or prediction.name.lower()
                ),
                InventoryItem.status == "active",
            )
        )
        if existing is not None:
            existing.name = prediction.name
            existing.image_key = image_key or existing.image_key
            existing.model_version = model_version or existing.model_version
            if commit:
                await self.session.commit()
                await self.session.refresh(existing)
            else:
                await self.session.flush()
            return existing
        item = InventoryItem(
            owner_id=owner_id,
            name=prediction.name,
            normalized_name=prediction.normalized_name or prediction.name.lower(),
            image_key=image_key,
            model_version=model_version,
        )
        self.session.add(item)
        if commit:
            await self.session.commit()
            await self.session.refresh(item)
        else:
            await self.session.flush()
        return item

    async def list_active(self, owner_id: str) -> list[InventoryItem]:
        result = await self.session.execute(
            select(InventoryItem)
            .where(InventoryItem.owner_id == owner_id, InventoryItem.status == "active")
            .order_by(InventoryItem.name, InventoryItem.created_at)
        )
        return list(result.scalars())

    async def consume(self, item: InventoryItem) -> InventoryItem:
        item.status = "consumed"
        await self.session.commit()
        return item

    async def find_active_by_name(self, owner_id: str, item_name: str) -> InventoryItem | None:
        normalized = item_name.strip().lower()
        return await self.session.scalar(
            select(InventoryItem)
            .where(
                InventoryItem.owner_id == owner_id,
                InventoryItem.status == "active",
                (
                    (InventoryItem.normalized_name == normalized)
                    | (InventoryItem.name == item_name.strip())
                ),
            )
            .order_by(InventoryItem.created_at.desc())
        )


class DraftRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        owner_id: str,
        message_id: str,
        thread_id: str,
        image_key: str,
        image_path: str | None,
        result: VisionResult,
    ) -> RecognitionDraft:
        draft = RecognitionDraft(
            owner_id=owner_id,
            message_id=message_id,
            thread_id=thread_id,
            image_key=image_key,
            image_path=image_path,
            prediction_json=result.model_dump_json(),
            model_version=result.model_version,
        )
        self.session.add(draft)
        await self.session.commit()
        await self.session.refresh(draft)
        return draft

    async def get(self, draft_id: str, owner_id: str) -> RecognitionDraft | None:
        return await self.session.scalar(
            select(RecognitionDraft).where(
                RecognitionDraft.id == draft_id, RecognitionDraft.owner_id == owner_id
            )
        )

    async def confirm(self, draft: RecognitionDraft, corrected: VisionResult) -> None:
        draft.corrected_json = corrected.model_dump_json()
        draft.status = "confirmed"
        await self.session.commit()


class PendingActionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        thread_id: str,
        owner_id: str,
        kind: str,
        payload: dict[str, Any],
        ttl_hours: int = 24,
    ) -> PendingAction:
        action = PendingAction(
            thread_id=thread_id,
            owner_id=owner_id,
            kind=kind,
            payload_json=dump_json(payload),
            expires_at=utcnow() + timedelta(hours=ttl_hours),
        )
        self.session.add(action)
        await self.session.commit()
        await self.session.refresh(action)
        return action

    async def consume(self, action_id: str, owner_id: str) -> PendingAction | None:
        action = await self.session.scalar(
            select(PendingAction).where(
                PendingAction.action_id == action_id,
                PendingAction.owner_id == owner_id,
                PendingAction.status == "pending",
                PendingAction.expires_at > utcnow(),
            )
        )
        if action:
            action.status = "processing"
            await self.session.commit()
        return action

    async def finish(self, action: PendingAction, status: str = "completed") -> None:
        action.status = status
        await self.session.commit()


class SideEffectRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def begin(self, key: str, effect_type: str) -> tuple[SideEffect, bool]:
        effect = SideEffect(idempotency_key=key, effect_type=effect_type)
        self.session.add(effect)
        try:
            await self.session.commit()
            await self.session.refresh(effect)
            return effect, True
        except IntegrityError:
            await self.session.rollback()
            existing = await self.session.scalar(
                select(SideEffect).where(SideEffect.idempotency_key == key)
            )
            if existing is None:
                raise RuntimeError("side effect idempotency lookup failed") from None
            return existing, existing.status == "failed"

    async def succeed(self, effect: SideEffect, response: Any) -> None:
        effect.status = "completed"
        effect.response_json = dump_json(response)
        await self.session.commit()

    async def fail(self, effect: SideEffect, error: Exception) -> None:
        effect.status = "failed"
        effect.error = str(error)[:4000]
        await self.session.commit()


async def record_unhandled(
    session: AsyncSession, owner_id: str, message_id: str, request: str, intent: str
) -> None:
    session.add(
        UnhandledIntent(
            owner_id=owner_id,
            message_id=message_id,
            raw_request=request,
            planner_intent=intent,
        )
    )
    await session.commit()
