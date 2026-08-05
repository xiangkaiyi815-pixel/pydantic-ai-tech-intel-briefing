from __future__ import annotations

import json
import time
import urllib.parse
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol


class FeishuClient(Protocol):
    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        ...

    def reply_post(self, message_id: str, post: dict[str, object]) -> dict[str, object]:
        ...


Transport = Callable[[str, str, dict[str, str], str], dict[str, object]]


class FeishuApiError(RuntimeError):
    pass


class FakeFeishuClient:
    def __init__(self) -> None:
        self.replies: list[dict[str, object]] = []

    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        reply = {"message_id": message_id, "text": text}
        self.replies.append(reply)
        return reply

    def reply_post(self, message_id: str, post: dict[str, object]) -> dict[str, object]:
        reply = {"message_id": message_id, "post": post}
        self.replies.append(reply)
        return reply


class FeishuHttpClient:
    def __init__(
        self,
        app_id: str | None,
        app_secret: str | None,
        base_url: str = "https://open.feishu.cn",
        transport: Transport | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _urllib_transport
        self.clock = clock or time.time
        self._tenant_access_token: str | None = None
        self._tenant_access_token_expires_at = 0.0

    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FeishuHttpClient requires FEISHU_APP_ID and FEISHU_APP_SECRET")
        token = self._get_tenant_access_token()
        response = self.transport(
            "POST",
            f"{self.base_url}/open-apis/im/v1/messages/{message_id}/reply",
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json.dumps(
                {
                    "msg_type": "text",
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                },
                ensure_ascii=False,
            ),
        )
        self._ensure_success(response, "reply message")
        return {"message_id": message_id, "response": response.get("data", {})}

    def reply_post(self, message_id: str, post: dict[str, object]) -> dict[str, object]:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FeishuHttpClient requires FEISHU_APP_ID and FEISHU_APP_SECRET")
        token = self._get_tenant_access_token()
        response = self.transport(
            "POST",
            f"{self.base_url}/open-apis/im/v1/messages/{message_id}/reply",
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json.dumps(
                {
                    "msg_type": "post",
                    "content": json.dumps(post, ensure_ascii=False),
                },
                ensure_ascii=False,
            ),
        )
        self._ensure_success(response, "reply message")
        return {"message_id": message_id, "response": response.get("data", {})}

    def get_bot_info(self) -> dict[str, object]:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FeishuHttpClient requires FEISHU_APP_ID and FEISHU_APP_SECRET")
        token = self._get_tenant_access_token()
        response = self.transport(
            "GET",
            f"{self.base_url}/open-apis/bot/v3/info",
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            "",
        )
        self._ensure_success(response, "get bot info")
        data = response.get("data", {})
        if not isinstance(data, dict):
            raise FeishuApiError("Feishu bot info response did not include data")
        return data

    def list_chat_messages(
        self,
        chat_id: str,
        start_time: int,
        end_time: int,
        page_size: int = 20,
    ) -> list[dict[str, object]]:
        if not self.app_id or not self.app_secret:
            raise RuntimeError("FeishuHttpClient requires FEISHU_APP_ID and FEISHU_APP_SECRET")
        token = self._get_tenant_access_token()
        query = urllib.parse.urlencode(
            {
                "container_id_type": "chat",
                "container_id": chat_id,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "page_size": str(page_size),
                "sort_type": "ByCreateTimeDesc",
            }
        )
        response = self.transport(
            "GET",
            f"{self.base_url}/open-apis/im/v1/messages?{query}",
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            "",
        )
        self._ensure_success(response, "list chat messages")
        data = response.get("data", {})
        if not isinstance(data, dict):
            raise FeishuApiError("Feishu message list response did not include data")
        items = data.get("items", [])
        if not isinstance(items, list):
            raise FeishuApiError("Feishu message list response did not include items")
        return [item for item in items if isinstance(item, dict)]

    def _get_tenant_access_token(self) -> str:
        if self._tenant_access_token and self.clock() < self._tenant_access_token_expires_at:
            return self._tenant_access_token
        response = self.transport(
            "POST",
            f"{self.base_url}/open-apis/auth/v3/tenant_access_token/internal",
            {"Content-Type": "application/json; charset=utf-8"},
            json.dumps({"app_id": self.app_id, "app_secret": self.app_secret}, ensure_ascii=False),
        )
        self._ensure_success(response, "get tenant access token")
        token = response.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuApiError("Feishu token response did not include tenant_access_token")
        expires_in = response.get("expire", 7000)
        try:
            expires_in_seconds = float(expires_in)
        except (TypeError, ValueError):
            expires_in_seconds = 7000.0
        self._tenant_access_token = token
        self._tenant_access_token_expires_at = self.clock() + max(60.0, expires_in_seconds - 60.0)
        return token

    def _ensure_success(self, response: dict[str, object], action: str) -> None:
        if response.get("code") != 0:
            raise FeishuApiError(f"Feishu {action} failed: {response}")


def _urllib_transport(method: str, url: str, headers: dict[str, str], body: str) -> dict[str, object]:
    request = urllib.request.Request(
        url=url,
        data=body.encode("utf-8") if body else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise FeishuApiError(f"Feishu HTTP {exc.code}: {payload}") from exc
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise FeishuApiError(f"Feishu response was not a JSON object: {payload}")
    return parsed
