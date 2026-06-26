from __future__ import annotations

import json
from typing import Any

from search_assistant.contracts import IncomingMessage


def handle_challenge(payload: dict[str, Any]) -> dict[str, str] | None:
    if payload.get("type") == "url_verification" and "challenge" in payload:
        return {"challenge": str(payload["challenge"])}
    if "challenge" in payload and payload.get("schema") in {None, "2.0"}:
        return {"challenge": str(payload["challenge"])}
    return None


def parse_feishu_event(payload: dict[str, Any]) -> IncomingMessage:
    header = _dict(payload.get("header"))
    event_type = header.get("event_type")
    if event_type != "im.message.receive_v1":
        raise ValueError(f"unsupported Feishu event type: {event_type}")

    event = _dict(payload.get("event"))
    message = _dict(event.get("message"))
    sender = _dict(event.get("sender"))
    sender_id = _dict(sender.get("sender_id"))
    content = _parse_content(message.get("content"))
    text = str(content.get("text") or "").strip()
    if not text:
        raise ValueError("Feishu message event does not contain text")

    return IncomingMessage(
        message_id=str(message["message_id"]),
        event_id=str(header.get("event_id") or message["message_id"]),
        user_id=str(sender_id.get("open_id") or sender_id.get("user_id") or "unknown"),
        chat_id=str(message["chat_id"]),
        text=text,
        source="feishu",
        raw_event=payload,
    )


def _parse_content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Feishu message content must be a JSON object")


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
