from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="ai_service/.env.production", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NutriSnap AI Service"
    app_env: str = "production"
    api_prefix: str = "/api/v1"
    model_path: Path
    labels_path: Path
    class_names_path: Path
    model_version: str = "mobilenetv2-v1"
    confidence_threshold: float = 0.50
    default_top_k: int = 3
    max_top_k: int = 15
    max_image_size_bytes: int = 10_485_760
    allowed_image_types: str = "image/jpeg,image/png,image/webp"
    max_image_width: int = 6000
    max_image_height: int = 6000
    inference_concurrency: int = 1
    enable_docs: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    auth_mode: str = "api_key"
    api_key: str | None = Field(default=None, repr=False)

    @property
    def allowed_types(self) -> set[str]:
        return {item.strip() for item in self.allowed_image_types.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
