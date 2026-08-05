import pytest
from pydantic import ValidationError

from search_assistant.contracts import (
    AnswerPackage,
    BriefingSynthesis,
    IncomingMessage,
    SearchRecord,
    SourceEvidence,
)


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


def test_answer_package_serializes_structured_search_record():
    source = SourceEvidence(
        title="Official source",
        url="https://example.com/official",
        snippet="Source text used by the verifier.",
        provider="unit",
        checked_at="2026-07-03T00:00:00Z",
    )
    package = AnswerPackage(
        question_id="q-search-record",
        answer_text="Answer with visible search record.",
        classification="research",
        confidence="medium",
        sources=[source],
        search_record=SearchRecord(
            executed=True,
            queries=["official source query"],
            engines=["bing", "google"],
            sources=[source],
        ),
    )

    payload = package.model_dump(mode="json")
    restored = AnswerPackage.model_validate(payload)

    assert payload["search_record"]["executed"] is True
    assert payload["search_record"]["queries"] == ["official source query"]
    assert payload["search_record"]["engines"] == ["bing", "google"]
    assert payload["search_record"]["sources"][0]["url"] == "https://example.com/official"
    assert restored.search_record.sources[0].title == "Official source"


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


def test_briefing_synthesis_requires_nonempty_themes_and_action_lists():
    with pytest.raises(ValidationError):
        BriefingSynthesis(
            search_content_summary="summary",
            short_summary="short summary",
            themes=[],
            key_signal_interpretation="signals",
            analysis_judgment="analysis",
            next_search_directions=[],
            landing_suggestions=[],
        )
