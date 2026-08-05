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


def test_feishu_http_client_replies_post_content():
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
    post = {
        "zh_cn": {
            "title": "Search answer",
            "content": [
                [{"tag": "text", "text": "Clean paragraph"}],
                [{"tag": "a", "text": "Source", "href": "https://example.com"}],
            ],
        }
    }

    result = client.reply_post("om_1", post)

    assert result["message_id"] == "om_1"
    assert calls[1]["headers"]["Authorization"] == "Bearer tenant-token"
    assert calls[1]["body"]["msg_type"] == "post"
    assert json.loads(calls[1]["body"]["content"]) == post


def test_feishu_http_client_refreshes_tenant_token_before_expiry():
    calls = []
    now = [1000.0]
    token_index = [0]

    def transport(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            token_index[0] += 1
            return {"code": 0, "tenant_access_token": f"tenant-token-{token_index[0]}", "expire": 7200}
        if url.endswith("/im/v1/messages/om_1/reply"):
            return {"code": 0, "data": {"message_id": "reply_1"}}
        raise AssertionError(f"unexpected url: {url}")

    client = FeishuHttpClient(
        app_id="cli_x",
        app_secret="secret",
        base_url="https://open.feishu.cn",
        transport=transport,
        clock=lambda: now[0],
    )

    client.reply_text("om_1", "first")
    now[0] += 100
    client.reply_text("om_1", "second")
    now[0] = 1000 + 7200 - 30
    client.reply_text("om_1", "third")

    auth_calls = [call for call in calls if call["url"].endswith("/auth/v3/tenant_access_token/internal")]
    reply_calls = [call for call in calls if call["url"].endswith("/im/v1/messages/om_1/reply")]
    assert len(auth_calls) == 2
    assert reply_calls[0]["headers"]["Authorization"] == "Bearer tenant-token-1"
    assert reply_calls[1]["headers"]["Authorization"] == "Bearer tenant-token-1"
    assert reply_calls[2]["headers"]["Authorization"] == "Bearer tenant-token-2"


def test_feishu_http_client_get_bot_info_uses_tenant_token():
    calls = []

    def transport(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return {"code": 0, "tenant_access_token": "tenant-token"}
        if url.endswith("/bot/v3/info"):
            return {"code": 0, "data": {"app_name": "智能搜索助手", "activate_status": 2}}
        raise AssertionError(f"unexpected url: {url}")

    client = FeishuHttpClient(
        app_id="cli_x",
        app_secret="secret",
        base_url="https://open.feishu.cn",
        transport=transport,
    )

    result = client.get_bot_info()

    assert result == {"app_name": "智能搜索助手", "activate_status": 2}
    assert calls[1]["method"] == "GET"
    assert calls[1]["headers"]["Authorization"] == "Bearer tenant-token"
    assert calls[1]["body"] == ""


def test_feishu_http_client_lists_chat_messages():
    calls = []

    def transport(method, url, headers, body):
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return {"code": 0, "tenant_access_token": "tenant-token"}
        if "/im/v1/messages?" in url:
            return {
                "code": 0,
                "data": {
                    "items": [
                        {
                            "message_id": "om_1",
                            "chat_id": "oc_1",
                            "msg_type": "text",
                            "body": {"content": "{\"text\":\"hello\"}"},
                            "sender": {"sender_type": "user", "id": "ou_1"},
                        }
                    ]
                },
            }
        raise AssertionError(f"unexpected url: {url}")

    client = FeishuHttpClient(app_id="cli_x", app_secret="secret", transport=transport)

    messages = client.list_chat_messages(
        chat_id="oc_1",
        start_time=1700000000,
        end_time=1700000100,
        page_size=10,
    )

    assert messages[0]["message_id"] == "om_1"
    assert calls[1]["method"] == "GET"
    assert "container_id_type=chat" in calls[1]["url"]
    assert "container_id=oc_1" in calls[1]["url"]
    assert "start_time=1700000000" in calls[1]["url"]
    assert "end_time=1700000100" in calls[1]["url"]
    assert calls[1]["headers"]["Authorization"] == "Bearer tenant-token"
