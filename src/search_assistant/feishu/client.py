from __future__ import annotations

from typing import Protocol


class FeishuClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        ...


class FakeFeishuClient:
    def __init__(self) -> None:
        self.replies: list[dict[str, object]] = []

    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        reply = {"message_id": message_id, "text": text}
        self.replies.append(reply)
        return reply


class FeishuHttpClient:
    def __init__(self, app_id: str | None, app_secret: str | None):
        self.app_id = app_id
        self.app_secret = app_secret

    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FeishuHttpClient requires FEISHU_APP_ID and FEISHU_APP_SECRET")
        raise RuntimeError("Live Feishu HTTP delivery is reserved for production configuration")
