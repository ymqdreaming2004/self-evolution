from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import ORJSONResponse

from .config import get_settings
from .db import Database
from .providers.feishu import FeishuClient
from .repositories import JobRepository
from .schemas import CardAction, ImageAttachment, IncomingMessage

URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
settings = get_settings()
database = Database(settings)
feishu = FeishuClient(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await database.initialize()
    yield
    await feishu.close()
    await database.dispose()


app = FastAPI(
    title="self-evolution-Agent",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> ORJSONResponse:
    db_ok = await database.healthy()
    missing = settings.missing_runtime_configuration()
    payload = {
        "status": "ok" if db_ok and not missing else "not_ready",
        "database": db_ok,
        "missing": missing,
    }
    return ORJSONResponse(payload, status_code=200 if db_ok and not missing else 503)


@app.post("/webhooks/feishu/events")
async def feishu_events(request: Request) -> dict[str, Any]:
    body = await request.body()
    _verify_request(request, body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if "encrypt" in payload:
        raise HTTPException(status_code=400, detail="encrypted event payloads are not enabled")
    if payload.get("type") == "url_verification":
        if not feishu.verify_token(payload):
            raise HTTPException(status_code=403, detail="invalid verification token")
        return {"challenge": payload.get("challenge", "")}
    if not feishu.verify_token(payload):
        raise HTTPException(status_code=403, detail="invalid verification token")
    message = parse_message_event(payload)
    if message is None:
        return {"ok": True, "ignored": True}
    _authorize(message.open_id, message.chat_type)
    async with database.sessions() as session:
        job, created = await JobRepository(session).enqueue(
            kind="message",
            payload=message.model_dump(mode="json"),
            idempotency_key=f"feishu-event:{message.event_id}",
            max_attempts=settings.worker_max_attempts,
        )
    return {"ok": True, "queued": created, "job_id": job.id if job else None}


@app.post("/webhooks/feishu/actions")
async def feishu_actions(request: Request) -> dict[str, Any]:
    body = await request.body()
    _verify_request(request, body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not feishu.verify_token(payload):
        raise HTTPException(status_code=403, detail="invalid verification token")
    action, open_id = parse_card_action(payload)
    _authorize(open_id, "p2p")
    job_payload = action.model_dump(mode="json") | {"open_id": open_id}
    async with database.sessions() as session:
        job, created = await JobRepository(session).enqueue(
            kind="card_action",
            payload=job_payload,
            idempotency_key=f"card:{action.action_id}:{action.action}",
            max_attempts=settings.worker_max_attempts,
        )
    return {
        "toast": {"type": "info", "content": "正在处理"},
        "queued": created,
        "job_id": job.id if job else None,
    }


def _verify_request(request: Request, body: bytes) -> None:
    if not settings.feishu_encrypt_key:
        return
    timestamp = request.headers.get("X-Lark-Request-Timestamp", "")
    nonce = request.headers.get("X-Lark-Request-Nonce", "")
    signature = request.headers.get("X-Lark-Signature", "")
    if not feishu.verify_signature(body, timestamp, nonce, signature):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid signature")


def _authorize(open_id: str, chat_type: str) -> None:
    if chat_type != "p2p":
        raise HTTPException(status_code=403, detail="only private chats are allowed")
    if not settings.feishu_allowed_open_id or open_id != settings.feishu_allowed_open_id:
        raise HTTPException(status_code=403, detail="user is not authorized")


def parse_message_event(payload: dict[str, Any]) -> IncomingMessage | None:
    header = payload.get("header", {})
    if header.get("event_type") != "im.message.receive_v1":
        return None
    event = payload.get("event", {})
    raw_message = event.get("message", {})
    sender = event.get("sender", {}).get("sender_id", {})
    message_type = raw_message.get("message_type")
    if message_type not in {"text", "image"}:
        return None
    try:
        content = json.loads(raw_message.get("content") or "{}")
    except json.JSONDecodeError:
        content = {}
    text = content.get("text", "") if message_type == "text" else ""
    images = (
        [ImageAttachment(image_key=content["image_key"])]
        if message_type == "image" and content.get("image_key")
        else []
    )
    return IncomingMessage(
        event_id=header.get("event_id") or raw_message.get("message_id", ""),
        message_id=raw_message.get("message_id", ""),
        chat_id=raw_message.get("chat_id", ""),
        open_id=sender.get("open_id", ""),
        chat_type=raw_message.get("chat_type", ""),
        text=text,
        images=images,
        urls=URL_PATTERN.findall(text),
    )


def parse_card_action(payload: dict[str, Any]) -> tuple[CardAction, str]:
    event = payload.get("event", payload)
    action_data = event.get("action", {})
    value = action_data.get("value", {})
    if isinstance(value, str):
        value = json.loads(value)
    form_values = action_data.get("form_value") or action_data.get("form_values") or {}
    user = event.get("operator", {}).get("operator_id", {}) or event.get("user_id", {})
    open_id = user.get("open_id") or event.get("open_id", "")
    action = CardAction(
        action_id=value.get("action_id", ""),
        thread_id=value.get("thread_id", ""),
        action=value.get("action", "cancel"),
        values=form_values,
    )
    if not action.action_id or not action.thread_id or not open_id:
        raise HTTPException(status_code=400, detail="invalid card action payload")
    return action, open_id
