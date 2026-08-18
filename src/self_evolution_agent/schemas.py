from __future__ import annotations

from datetime import UTC, datetime
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
    GENERAL_CHAT = "general_chat"


class TaskKind(StrEnum):
    INSPIRATION = "inspiration"
    FRIDGE = "fridge"
    GENERAL = "general"


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
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def normalize_name(self) -> IngredientPrediction:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("ingredient name must not be empty")
        if not self.normalized_name:
            self.normalized_name = self.name.strip().lower()
        else:
            self.normalized_name = self.normalized_name.strip().lower()
        return self


class VisionResult(BaseModel):
    items: list[IngredientPrediction]
    model_version: str = ""
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
    note_link: str = ""
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


class RecipeSuggestion(BaseModel):
    title: str
    inventory_ingredients: list[str]
    extra_ingredients: list[str] = Field(default_factory=list)
    steps: list[str]


class RecipeResult(BaseModel):
    recipes: list[RecipeSuggestion] = Field(min_length=1, max_length=5)


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
    action: Literal["consume"] = "consume"
    item_name: str
