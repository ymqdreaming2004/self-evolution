from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, UniqueConstraint, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import Settings


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    kind: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=5)
    available_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)


class ProcessedEvent(Base):
    __tablename__ = "processed_events"

    event_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class InventoryItem(Base, TimestampMixin):
    __tablename__ = "inventory_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(255))
    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    # Legacy storage columns remain mapped so existing SQLite databases can accept new rows.
    # The simplified Fridge Agent does not read, expose or update them.
    quantity: Mapped[float] = mapped_column(Float, default=1)
    unit: Mapped[str] = mapped_column(String(32), default="件")
    production_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[Any | None] = mapped_column(Date, nullable=True, index=True)
    date_source: Mapped[str] = mapped_column(String(32), default="unknown")
    image_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)


class RecognitionDraft(Base, TimestampMixin):
    __tablename__ = "recognition_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    message_id: Mapped[str] = mapped_column(String(255))
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    image_key: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    prediction_json: Mapped[str] = mapped_column(Text)
    corrected_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="pending")


class PendingAction(Base, TimestampMixin):
    __tablename__ = "pending_actions"

    action_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class SideEffect(Base, TimestampMixin):
    __tablename__ = "side_effects"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_side_effect_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    idempotency_key: Mapped[str] = mapped_column(String(255))
    effect_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class UnhandledIntent(Base):
    __tablename__ = "unhandled_intents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    message_id: Mapped[str] = mapped_column(String(255))
    raw_request: Mapped[str] = mapped_column(Text)
    planner_intent: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Database:
    def __init__(self, settings: Settings):
        self.engine: AsyncEngine = create_async_engine(settings.database_url, future=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            if self.engine.url.get_backend_name() == "sqlite":
                await connection.execute(text("PRAGMA journal_mode=WAL"))

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session

    async def healthy(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()


async def recover_stale_jobs(session: AsyncSession, stale_after_minutes: int = 15) -> int:
    cutoff = utcnow() - timedelta(minutes=stale_after_minutes)
    result = await session.execute(
        select(Job).where(Job.status == "running", Job.locked_at < cutoff)
    )
    jobs = list(result.scalars())
    for job in jobs:
        job.status = "queued"
        job.locked_at = None
        job.available_at = utcnow()
    await session.commit()
    return len(jobs)
