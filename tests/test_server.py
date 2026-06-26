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
