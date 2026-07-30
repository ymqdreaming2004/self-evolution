from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..config import Settings
from ..schemas import VisionResult


class VisionProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=90)
        self._owns_client = client is None

    async def recognize(self, image_path: str | Path) -> VisionResult:
        path = Path(image_path)
        content = await asyncio.to_thread(path.read_bytes)
        if len(content) > self.settings.vision_max_image_bytes:
            raise ValueError("image exceeds configured size limit")
        response = await self.client.post(
            f"{self.settings.vision_base_url.rstrip('/')}/v1/recognize",
            files={"image": (path.name, content, "application/octet-stream")},
        )
        response.raise_for_status()
        return VisionResult.model_validate(response.json())

    async def healthy(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.settings.vision_base_url.rstrip('/')}/health/ready", timeout=3
            )
            return response.is_success
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
