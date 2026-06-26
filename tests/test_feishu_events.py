from search_assistant.feishu.events import handle_challenge, parse_feishu_event


def test_feishu_challenge_response():
    assert handle_challenge({"type": "url_verification", "challenge": "abc"}) == {"challenge": "abc"}


def test_parse_receive_message_event():
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    message = parse_feishu_event(payload)

    assert message.text == "hello"
    assert message.event_id == "evt_1"
