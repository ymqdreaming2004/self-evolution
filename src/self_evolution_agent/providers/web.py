from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import Settings


class UnsafeUrlError(ValueError):
    pass


class WebContentFetcher:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(follow_redirects=False)
        self._owns_client = client is None

    async def fetch(self, url: str) -> tuple[str, str]:
        current = url
        for _ in range(4):
            await self._validate_public_url(current)
            async with self.client.stream(
                "GET",
                current,
                timeout=self.settings.web_fetch_timeout_seconds,
                headers={"User-Agent": "self-evolution-agent/0.1"},
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise httpx.HTTPError("redirect without location")
                    current = str(response.url.join(location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise ValueError(f"unsupported content type: {content_type}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.settings.web_fetch_max_bytes:
                        raise ValueError("web content exceeds configured size limit")
                text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
                if "text/html" in content_type:
                    soup = BeautifulSoup(text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer"]):
                        tag.decompose()
                    title = soup.title.get_text(" ", strip=True) if soup.title else current
                    content = soup.get_text("\n", strip=True)
                else:
                    title, content = current, text.strip()
                return title[:500], content
        raise ValueError("too many redirects")

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise UnsafeUrlError("only public HTTP(S) URLs are allowed")
        loopback_names = {"localhost", "localhost.localdomain"}
        if parsed.hostname.lower() in loopback_names:
            raise UnsafeUrlError("local addresses are not allowed")
        try:
            addresses = (
                await __import__("asyncio")
                .get_running_loop()
                .run_in_executor(
                    None, lambda: socket.getaddrinfo(parsed.hostname, parsed.port or 443)
                )
            )
        except socket.gaierror as exc:
            raise UnsafeUrlError("URL hostname cannot be resolved") from exc
        for item in addresses:
            ip = ipaddress.ip_address(item[4][0])
            if not ip.is_global:
                raise UnsafeUrlError("private or reserved addresses are not allowed")

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
