from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lark_oapi as lark
from lark_oapi.core.enum import LogLevel

from search_assistant.config import Settings
from search_assistant.feishu.client import FeishuHttpClient


EndpointChecker = Callable[[str, str], dict[str, object]]


def run_feishu_diagnostics(
    settings: Settings | None = None,
    endpoint_checker: EndpointChecker | None = None,
    client: FeishuHttpClient | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    checks: list[dict[str, Any]] = []
    hints: list[str] = []

    app_id_set = bool(settings.feishu_app_id)
    app_secret_set = bool(settings.feishu_app_secret)
    checks.append(
        {
            "name": "environment",
            "ok": app_id_set and app_secret_set,
            "detail": "FEISHU_APP_ID and FEISHU_APP_SECRET are set"
            if app_id_set and app_secret_set
            else "FEISHU_APP_ID or FEISHU_APP_SECRET is missing",
        }
    )
    if not app_id_set or not app_secret_set:
        hints.append("Set FEISHU_APP_ID and FEISHU_APP_SECRET in the shell that starts the assistant.")
        return _report(checks, hints)

    http_client = client or FeishuHttpClient(settings.feishu_app_id, settings.feishu_app_secret)
    try:
        bot_info = http_client.get_bot_info()
        checks.append({"name": "credentials", "ok": True, "detail": "tenant token and bot info request succeeded"})
        checks.append(_bot_info_check(bot_info))
    except Exception as exc:
        checks.append({"name": "credentials", "ok": False, "detail": _safe_error(exc)})
        hints.append("Check whether the Feishu app id and app secret belong to the bot app.")
        return _report(checks, hints)

    checker = endpoint_checker or check_long_connection_endpoint
    try:
        endpoint_result = checker(settings.feishu_app_id or "", settings.feishu_app_secret or "")
        checks.append(
            {
                "name": "long_connection_endpoint",
                "ok": bool(endpoint_result.get("ok")),
                "detail": str(endpoint_result.get("detail") or "endpoint check completed"),
            }
        )
        if not endpoint_result.get("ok"):
            hints.append("Enable Feishu long-connection event subscription for this app.")
    except Exception as exc:
        checks.append({"name": "long_connection_endpoint", "ok": False, "detail": _safe_error(exc)})
        hints.append("Enable long-connection event subscription and verify the app has event access.")

    hints.extend(_event_delivery_hints(checks))
    return _report(checks, hints)


def check_long_connection_endpoint(app_id: str, app_secret: str) -> dict[str, object]:
    event_handler = lark.EventDispatcherHandler.builder("", "").build()
    client = lark.ws.Client(
        app_id=app_id,
        app_secret=app_secret,
        log_level=LogLevel.ERROR,
        event_handler=event_handler,
        auto_reconnect=False,
    )
    conn_url = client._get_conn_url()
    return {"ok": conn_url.startswith("wss://"), "detail": "long-connection endpoint generated"}


def _bot_info_check(bot_info: dict[str, object]) -> dict[str, Any]:
    activate_status = bot_info.get("activate_status")
    ok = activate_status in {None, 2, "2"}
    detail = "bot info request succeeded"
    if activate_status is not None:
        detail = f"bot activate_status={activate_status}"
    data: dict[str, object] = {}
    if "app_name" in bot_info:
        data["app_name"] = bot_info["app_name"]
    if activate_status is not None:
        data["activate_status"] = activate_status
    return {"name": "bot_info", "ok": ok, "detail": detail, "data": data}


def _event_delivery_hints(checks: list[dict[str, Any]]) -> list[str]:
    if not all(check["ok"] for check in checks):
        return []
    return [
        "If no message arrives, verify the app subscribed to im.message.receive_v1 in long-connection mode.",
        "Send a direct message to this bot, or add and mention the bot in a chat where it has permission.",
    ]


def _report(checks: list[dict[str, Any]], hints: list[str]) -> dict[str, Any]:
    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "hints": hints,
    }


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("tenant_access_token", "Authorization", "app_secret"):
        text = text.replace(marker, "[redacted]")
    return text
