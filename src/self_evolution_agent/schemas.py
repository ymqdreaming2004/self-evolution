from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Intent(StrEnum):
    KNOWLEDGE_STORE = "knowledge_store"
    KNOWLEDGE_QUERY = "knowledge_query"
    INSPIRATION = "inspiration"
    FRIDGE_INGEST = "fridge_ingest"
    FRIDGE_QUERY = "fridge_query"
    FRIDGE_MUTATE = "fridge_mutate"
    RECIPE = "recipe"
    PLACEHOLDER = "placeholder"


class TaskKind(StrEnum):
    INSPIRATION = "inspiration"
    FRIDGE = "fridge"
    PLACEHOLDER = "placeholder"


class PlannedTask(BaseModel):
    id: str
    kind: TaskKind
    intent: Intent
    instruction: str
    dependencies: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False


class ExecutionPlan(BaseModel):
    intent: Intent
    tasks: list[PlannedTask]
    requires_confirmation: bool = False
    rationale: str = ""


class ImageAttachment(BaseModel):
    image_key: str
    local_path: str | None = None
    mime_type: str = "image/jpeg"


class IncomingMessage(BaseModel):
    event_id: str
    message_id: str
    chat_id: str
    open_id: str
    chat_type: str = "p2p"
    text: str = ""
    images: list[ImageAttachment] = Field(default_factory=list)
    urls: list[HttpUrl] = Field(default_factory=list)
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IngredientPrediction(BaseModel):
    name: str
    normalized_name: str | None = None
    quantity: float = Field(default=1, gt=0)
    unit: str = "件"
    production_date: date | None = None
    shelf_life_days: int | None = Field(default=None, gt=0)
    expiry_date: date | None = None
    date_source: Literal["printed", "calculated", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1)
    evidence_text: str = ""

    @model_validator(mode="after")
    def derive_expiry_date(self) -> IngredientPrediction:
        if self.expiry_date is None and self.production_date and self.shelf_life_days:
            from datetime import timedelta

            self.expiry_date = self.production_date + timedelta(days=self.shelf_life_days)
            self.date_source = "calculated"
        if not self.normalized_name:
            self.normalized_name = self.name.strip().lower()
        return self


class VisionResult(BaseModel):
    items: list[IngredientPrediction]
    model_version: str
    raw_text: str = ""


class KnowledgeChunk(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    title: str
    tags: list[str] = Field(default_factory=list)
    source: str
    created_at: datetime


class KnowledgeHit(BaseModel):
    content: str
    title: str
    source: str
    created_at: datetime
    score: float | None = None


class BitableIdea(BaseModel):
    title: str
    content: str
    type: Literal["灵感", "TODO"] = "灵感"
    tags: list[str] = Field(default_factory=list)
    status: str = "待处理"
    source_message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CardAction(BaseModel):
    action_id: str
    thread_id: str
    action: Literal["confirm", "edit", "consume", "delete", "cancel"]
    values: dict[str, Any] = Field(default_factory=dict)


class JobPayload(BaseModel):
    type: Literal["message", "card_action"]
    data: dict[str, Any]


class RecipeResult(BaseModel):
    title: str
    servings: int = 1
    inventory_ingredients: list[str]
    extra_ingredients: list[str] = Field(default_factory=list)
    steps: list[str]
    notes: str = ""


class AgentEffect(BaseModel):
    type: Literal[
        "bitable_append",
        "send_text",
        "send_card",
        "inventory_mutation",
        "fridge_confirmation",
    ]
    payload: dict[str, Any]
    idempotency_key: str
    requires_confirmation: bool = False


class AgentResult(BaseModel):
    task_id: str
    intent: Intent
    reply: str = ""
    effects: list[AgentEffect] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ContentMetadata(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list, max_length=8)
    kind: Literal["知识", "灵感", "TODO"] = "知识"


class InventoryMutation(BaseModel):
    action: Literal["update", "consume", "delete"]
    item_id: str
    values: dict[str, Any] = Field(default_factory=dict)
