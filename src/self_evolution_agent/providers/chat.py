from __future__ import annotations

import json
from typing import TypeVar

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from ..config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ChatProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.chat_api_key or "not-configured",
            base_url=settings.chat_base_url,
        )
        self.planner_client = AsyncOpenAI(
            api_key=settings.planner_api_key or "not-configured",
            base_url=settings.planner_base_url,
        )

    def _client_for(self, model: str) -> AsyncOpenAI:
        if model == self.settings.planner_model:
            return self.planner_client
        return self.client

    async def structured(
        self,
        *,
        schema: type[SchemaT],
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0,
    ) -> SchemaT:
        selected_model = model or self.settings.chat_model
        api_key = (
            self.settings.planner_api_key
            if selected_model == self.settings.planner_model
            else self.settings.chat_api_key
        )
        if not selected_model or not api_key:
            raise RuntimeError("chat model is not configured")
        client = self._client_for(selected_model)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = await client.chat.completions.create(
                model=selected_model,
                temperature=temperature,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            messages[0]["content"] += (
                "\n输出纯 JSON，不要 Markdown。JSON Schema："
                + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            )
            response = await client.chat.completions.create(
                model=selected_model,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("chat model returned empty structured output")
        return schema.model_validate_json(content)

    async def text(self, *, system: str, user: str, model: str | None = None) -> str:
        selected_model = model or self.settings.chat_model
        if not selected_model or not self.settings.chat_api_key:
            raise RuntimeError("chat model is not configured")
        response = await self._client_for(selected_model).chat.completions.create(
            model=selected_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


def json_for_prompt(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)
