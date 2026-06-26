from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    data_dir: Path = Field(default=Path(".local-data"))
    database_path: Path | None = None
    feishu_enabled: bool = False
    feishu_app_id: str | None = None
    feishu_app_secret: str | None = None
    feishu_verification_token: str | None = None
    feishu_encrypt_key: str | None = None
    model_provider: str = "fake"
    search_provider: str = "fake"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source: Mapping[str, str]
        if env is None:
            import os

            source = os.environ
        else:
            source = env

        data_dir = Path(source.get("SEARCH_ASSISTANT_DATA_DIR", ".local-data"))
        raw_database_path = source.get("SEARCH_ASSISTANT_DATABASE_PATH")
        database_path = Path(raw_database_path) if raw_database_path else data_dir / "assistant.sqlite3"

        values: dict[str, Any] = {
            "data_dir": data_dir,
            "database_path": database_path,
            "feishu_enabled": _to_bool(source.get("SEARCH_ASSISTANT_FEISHU_ENABLED"), default=False),
            "feishu_app_id": source.get("FEISHU_APP_ID") or None,
            "feishu_app_secret": source.get("FEISHU_APP_SECRET") or None,
            "feishu_verification_token": source.get("FEISHU_VERIFICATION_TOKEN") or None,
            "feishu_encrypt_key": source.get("FEISHU_ENCRYPT_KEY") or None,
            "model_provider": source.get("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake"),
            "search_provider": source.get("SEARCH_ASSISTANT_SEARCH_PROVIDER", "fake"),
        }
        return cls(**values)
