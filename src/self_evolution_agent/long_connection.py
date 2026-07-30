from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from .config import Settings
from .db import Database
from .repositories import JobRepository
from .schemas import ImageAttachment, IncomingMessage

logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")


def parse_long_connection_message(event: Any) -> IncomingMessage | None:
    """Convert the official SDK's typed message event into the app's stable schema."""
    data = getattr(event, "event", None)
    raw_message = getattr(data, "message", None)
    sender = getattr(getattr(data, "sender", None), "sender_id", None)
    message_type = getattr(raw_message, "message_type", None)
    if message_type not in {"text", "image"}:
        return None
    try:
        content = json.loads(getattr(raw_message, "content", "") or "{}")
    except json.JSONDecodeError:
        content = {}
    text = content.get("text", "") if message_type == "text" else ""
    images = (
        [ImageAttachment(image_key=content["image_key"])]
        if message_type == "image" and content.get("image_key")
        else []
    )
    message_id = getattr(raw_message, "message_id", "")
    header = getattr(event, "header", None)
    return IncomingMessage(
        event_id=getattr(header, "event_id", "") or message_id,
        message_id=message_id,
        chat_id=getattr(raw_message, "chat_id", ""),
        open_id=getattr(sender, "open_id", ""),
        chat_type=getattr(raw_message, "chat_type", ""),
        text=text,
        images=images,
        urls=URL_PATTERN.findall(text),
    )


class FeishuLongConnection:
    """Receives events over Feishu WebSocket and enqueues them without blocking ACKs."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._run_client)

    def _run_client(self) -> None:
        """Run the SDK on its own loop; the SDK binds a loop at import time."""
        try:
            sdk_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(sdk_loop)
            import lark_oapi as lark
            from lark_oapi.ws import Client as WsClient
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError("lark-oapi is required for Feishu long connections") from exc

        dispatcher = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )
        client = WsClient(
            self.settings.feishu_app_id,
            self.settings.feishu_app_secret,
            event_handler=dispatcher,
        )
        logger.info("Feishu long connection starting")
        client.start()

    def _on_message(self, event: Any) -> None:
        if self._loop is None:
            logger.warning("dropping Feishu event before listener loop is ready")
            return
        future = asyncio.run_coroutine_threadsafe(self._enqueue(event), self._loop)
        future.add_done_callback(self._log_failure)

    @staticmethod
    def _log_failure(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("failed to enqueue Feishu long-connection event")

    async def _enqueue(self, event: Any) -> None:
        message = parse_long_connection_message(event)
        if message is None:
            return
        if message.chat_type != "p2p" or message.open_id != self.settings.feishu_allowed_open_id:
            logger.warning("rejected Feishu event from unauthorized chat or user")
            return
        async with self.database.sessions() as session:
            _, created = await JobRepository(session).enqueue(
                kind="message",
                payload=message.model_dump(mode="json"),
                idempotency_key=f"feishu-event:{message.event_id}",
                max_attempts=self.settings.worker_max_attempts,
            )
        logger.info(
            "Feishu long-connection event queued=%s message_id=%s", created, message.message_id
        )
