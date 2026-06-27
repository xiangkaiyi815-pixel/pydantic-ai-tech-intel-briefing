from fastapi.testclient import TestClient

from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.server import create_app


def test_feishu_webhook_replies_with_answer(tmp_path):
    fake_client = FakeFeishuClient()
    app = create_app(data_dir=tmp_path, feishu_client=fake_client)
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert fake_client.replies[0]["message_id"] == "om_1"


def test_server_uses_feishu_http_client_when_enabled(monkeypatch, tmp_path):
    selected_client = FakeFeishuClient()

    def fake_http_client(app_id, app_secret):
        selected_client.app_id = app_id
        selected_client.app_secret = app_secret
        return selected_client

    monkeypatch.setenv("SEARCH_ASSISTANT_FEISHU_ENABLED", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "cli_x")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setattr("search_assistant.server.FeishuHttpClient", fake_http_client)

    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_enabled", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_enabled", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert selected_client.app_id == "cli_x"
    assert selected_client.app_secret == "secret"
    assert selected_client.replies[0]["message_id"] == "om_enabled"
