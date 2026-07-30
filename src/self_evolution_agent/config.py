from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    database_url: str = "sqlite+aiosqlite:///./data/self_evolution.db"
    chroma_path: Path = Path("./data/chroma")

    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    feishu_allowed_open_id: str = ""
    feishu_bitable_app_token: str = ""
    feishu_bitable_table_id: str = ""
    feishu_event_transport: Literal["webhook", "long_connection"] = "long_connection"

    chat_base_url: str = "https://api.openai.com/v1"
    chat_api_key: str = ""
    chat_model: str = ""
    planner_base_url: str = ""
    planner_api_key: str = ""
    planner_model: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    vision_base_url: str = "http://vision:8001"
    vision_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vision_model_version: str = "qwen2.5-vl-3b-4bit-v1"
    vision_max_image_bytes: int = 10 * 1024 * 1024

    worker_poll_seconds: float = 1.0
    worker_max_attempts: int = 5
    web_fetch_timeout_seconds: float = 10.0
    web_fetch_max_bytes: int = 2 * 1024 * 1024
    knowledge_top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def fill_model_defaults(self) -> Settings:
        if not self.planner_model:
            self.planner_model = self.chat_model
        if not self.planner_base_url:
            self.planner_base_url = self.chat_base_url
        if not self.planner_api_key:
            self.planner_api_key = self.chat_api_key
        return self

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "training").mkdir(parents=True, exist_ok=True)

    def missing_runtime_configuration(self) -> list[str]:
        required = {
            "FEISHU_APP_ID": self.feishu_app_id,
            "FEISHU_APP_SECRET": self.feishu_app_secret,
            "FEISHU_VERIFICATION_TOKEN": self.feishu_verification_token,
            "FEISHU_ALLOWED_OPEN_ID": self.feishu_allowed_open_id,
            "FEISHU_BITABLE_APP_TOKEN": self.feishu_bitable_app_token,
            "FEISHU_BITABLE_TABLE_ID": self.feishu_bitable_table_id,
            "CHAT_API_KEY": self.chat_api_key,
            "CHAT_MODEL": self.chat_model,
        }
        return [name for name, value in required.items() if not value]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
