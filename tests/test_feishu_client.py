import json

from search_assistant.feishu.client import FeishuHttpClient


def test_feishu_http_client_fetches_token_and_replies_text():
    calls = []

    def transport(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": headers, "body": json.loads(body)})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return {"code": 0, "tenant_access_token": "tenant-token"}
        if url.endswith("/im/v1/messages/om_1/reply"):
            return {"code": 0, "data": {"message_id": "reply_1"}}
        raise AssertionError(f"unexpected url: {url}")

    client = FeishuHttpClient(
        app_id="cli_x",
        app_secret="secret",
        base_url="https://open.feishu.cn",
        transport=transport,
    )

    result = client.reply_text("om_1", "hello")

    assert result["message_id"] == "om_1"
    assert calls[0]["body"] == {"app_id": "cli_x", "app_secret": "secret"}
    assert calls[1]["headers"]["Authorization"] == "Bearer tenant-token"
    assert calls[1]["body"]["msg_type"] == "text"
    assert json.loads(calls[1]["body"]["content"]) == {"text": "hello"}
