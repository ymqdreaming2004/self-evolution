from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings
from ..schemas import BitableIdea, VisionResult


class FeishuClient:
    API_BASE = "https://open.feishu.cn/open-apis"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=30)
        self._owns_client = client is None
        self._token = ""
        self._token_expires_at = 0.0

    def verify_signature(self, body: bytes, timestamp: str, nonce: str, signature: str) -> bool:
        if not self.settings.feishu_encrypt_key:
            return True
        expected = hashlib.sha256(
            timestamp.encode() + nonce.encode() + self.settings.feishu_encrypt_key.encode() + body
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def verify_token(self, payload: dict[str, Any]) -> bool:
        expected = self.settings.feishu_verification_token
        if not expected:
            return True
        supplied = payload.get("token") or payload.get("header", {}).get("token")
        return bool(supplied and hmac.compare_digest(str(supplied), expected))

    async def _tenant_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        response = await self.client.post(
            f"{self.API_BASE}/auth/v3/tenant_access_token/internal",
            json={
                "app_id": self.settings.feishu_app_id,
                "app_secret": self.settings.feishu_app_secret,
            },
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"Feishu token request failed: {data}")
        self._token = data["tenant_access_token"]
        self._token_expires_at = time.monotonic() + max(60, int(data.get("expire", 7200)) - 120)
        return self._token

    async def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self._tenant_token()}"}

    async def download_image(self, message_id: str, image_key: str, destination: Path) -> Path:
        response = await self.client.get(
            f"{self.API_BASE}/im/v1/messages/{message_id}/resources/{image_key}",
            params={"type": "image"},
            headers=await self._headers(),
        )
        response.raise_for_status()
        if len(response.content) > self.settings.vision_max_image_bytes:
            raise ValueError("Feishu image exceeds configured size limit")
        await asyncio.to_thread(self._write_image, destination, response.content)
        return destination

    @staticmethod
    def _write_image(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    async def send_text(
        self, open_id: str, text: str, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return await self._send_message(open_id, "text", {"text": text}, idempotency_key)

    async def send_card(
        self, open_id: str, card: dict[str, Any], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return await self._send_message(open_id, "interactive", card, idempotency_key)

    async def _send_message(
        self,
        open_id: str,
        message_type: str,
        content: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        params = {"receive_id_type": "open_id"}
        if idempotency_key:
            params["uuid"] = hashlib.sha256(idempotency_key.encode()).hexdigest()[:40]
        response = await self.client.post(
            f"{self.API_BASE}/im/v1/messages",
            params=params,
            headers=await self._headers(),
            json={
                "receive_id": open_id,
                "msg_type": message_type,
                "content": json.dumps(content, ensure_ascii=False),
            },
        )
        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text[:1000]}
        if response.is_error or data.get("code", 0) != 0:
            raise RuntimeError(
                "Feishu message send failed: "
                f"http_status={response.status_code}, code={data.get('code')}, "
                f"message={data.get('msg') or data.get('message') or data.get('raw')}"
            )
        return data.get("data", {})

    async def append_idea(self, idea: BitableIdea) -> dict[str, Any]:
        if not self.settings.feishu_bitable_app_token or not self.settings.feishu_bitable_table_id:
            raise RuntimeError("Feishu Bitable is not configured")
        fields = {
            "标题": idea.title,
            "内容": idea.content,
            "类型": idea.type,
            "标签": ", ".join(idea.tags),
            "状态": idea.status,
            "来源消息": idea.source_message,
            "创建时间": int(idea.created_at.timestamp() * 1000),
        }
        response = await self.client.post(
            f"{self.API_BASE}/bitable/v1/apps/{self.settings.feishu_bitable_app_token}"
            f"/tables/{self.settings.feishu_bitable_table_id}/records",
            headers=await self._headers(),
            json={"fields": fields},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"Bitable append failed: {data}")
        return data.get("data", {})

    async def validate_bitable_schema(self) -> tuple[bool, list[str]]:
        if not self.settings.feishu_bitable_app_token or not self.settings.feishu_bitable_table_id:
            return False, ["Bitable app token/table id are not configured"]
        response = await self.client.get(
            f"{self.API_BASE}/bitable/v1/apps/{self.settings.feishu_bitable_app_token}"
            f"/tables/{self.settings.feishu_bitable_table_id}/fields",
            headers=await self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        names = {item["field_name"] for item in data.get("data", {}).get("items", [])}
        required = {"标题", "内容", "类型", "标签", "状态", "来源消息", "创建时间"}
        missing = sorted(required - names)
        return not missing, missing

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def fridge_confirmation_card(
    *, action_id: str, thread_id: str, draft_id: str, result: VisionResult
) -> dict[str, Any]:
    lines = []
    form_elements: list[dict[str, Any]] = []
    for index, item in enumerate(result.items, start=1):
        lines.append(f"{index}. **{item.name}** · 置信度 {item.confidence:.0%}")
        form_elements.append(
            {
                "tag": "input",
                "name": f"name_{index - 1}",
                "default_value": item.name,
                "placeholder": {
                    "tag": "plain_text",
                    "content": "食材名称；留空表示不加入",
                },
            }
        )
    value = {"action_id": action_id, "thread_id": thread_id, "draft_id": draft_id}
    return {
        "schema": "2.0",
        "config": {"update_multi": True, "width_mode": "default"},
        "header": {
            "template": "orange",
            "title": {"tag": "plain_text", "content": "确认食材录入"},
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 20px 12px",
            "vertical_spacing": "12px",
            "elements": [
                {
                    "tag": "markdown",
                    "content": "\n".join(lines) + "\n\n可修改名称；识别错误的条目请清空名称。",
                },
                {
                    "tag": "form",
                    "name": "fridge_confirmation",
                    "vertical_spacing": "8px",
                    "elements": [
                        *form_elements,
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "确认入库"},
                            "type": "primary_filled",
                            "width": "fill",
                            "form_action_type": "submit",
                            "name": "confirm_inventory",
                            "value": {**value, "action": "confirm"},
                        },
                    ],
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "取消"},
                    "type": "default",
                    "width": "fill",
                    "behaviors": [
                        {"type": "callback", "value": {**value, "action": "cancel"}}
                    ],
                },
            ],
        },
    }


def inventory_card(items: list[Any], title: str = "现有食材") -> dict[str, Any]:
    rows = [f"- **{item.name}**" for item in items]
    return {
        "config": {"wide_screen_mode": True},
        "header": {"template": "blue", "title": {"tag": "plain_text", "content": title}},
        "elements": [{"tag": "markdown", "content": "\n".join(rows) or "暂无食材"}],
    }
