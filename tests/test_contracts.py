from search_assistant.contracts import AnswerPackage, IncomingMessage


def test_answer_package_requires_verification_fields():
    package = AnswerPackage(
        question_id="q-1",
        answer_text="Use verified sources.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["current pricing"],
        calibration={"ran": True, "critique": "Needs source", "revision": "Marked unverified"},
        memory_updates=[],
    )

    assert package.classification == "research"
    assert package.unverified_claims == ["current pricing"]


def test_incoming_message_keeps_feishu_metadata():
    message = IncomingMessage(
        message_id="om_1",
        event_id="evt_1",
        user_id="ou_1",
        chat_id="oc_1",
        text="What changed in Agent Framework?",
        source="feishu",
    )

    assert message.dedupe_key == "evt_1"
