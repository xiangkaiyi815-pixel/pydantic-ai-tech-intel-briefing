from datetime import date
import sqlite3

import pytest

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
)
from search_assistant.evolution.service import DomainKnowledgeCandidateService
from search_assistant.memory.store import MemoryStore
from search_assistant.reports.service import ReportService


def test_briefing_evidence_creates_deduplicated_reviewable_domain_candidates(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    store = MemoryStore(database_path)
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    briefing = _briefing(source_count=2)

    first_ids = service.capture_briefing(briefing)
    second_ids = service.capture_briefing(briefing)

    assert first_ids == second_ids
    assert len(first_ids) == 1
    candidates = store.list_domain_knowledge_candidates()
    assert len(candidates) == 1
    assert candidates[0]["topic"] == "industrial AI"
    assert candidates[0]["status"] == "candidate"
    assert candidates[0]["confidence"] == "medium"
    assert len(candidates[0]["evidence"]) == 2
    assert candidates[0]["source_ids"][0] == briefing.id

    validation = service.validate(first_ids[0])
    service.deprecate(first_ids[0], "superseded by a later evidence review")

    assert validation == {"validated": True, "candidate_id": first_ids[0], "failures": []}
    assert store.get_domain_knowledge_candidate(first_ids[0])["status"] == "deprecated"
    events = store.list_domain_knowledge_candidate_events(first_ids[0])
    assert [event["to_status"] for event in events] == ["candidate", "validated", "deprecated"]
    assert events[-1]["reason"] == "superseded by a later evidence review"
    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="candidate events are immutable"):
            connection.execute(
                "UPDATE domain_knowledge_candidate_events SET reason = 'changed' WHERE candidate_id = ?",
                (first_ids[0],),
            )


def test_candidate_validation_keeps_single_source_claim_in_candidate_state(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]

    result = service.validate(candidate_id)

    assert result["validated"] is False
    assert result["failures"] == ["fewer_than_two_original_sources", "low_confidence"]
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "candidate"


def test_learning_report_surfaces_trajectory_and_domain_candidate_status(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    DomainKnowledgeCandidateService(store).capture_briefing(_briefing(source_count=2))

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()

    assert "## Trajectory Evaluation Summary" in report
    assert "Immutable trajectories: 0" in report
    assert "## Domain Knowledge Candidates" in report
    assert "[candidate; confidence medium] industrial AI" in report
    assert "evidence: 2" in report


def _briefing(source_count: int) -> DailyBriefing:
    sources = [
        CollectedSource(
            id=f"source-{index}",
            topic_id="topic-1",
            user_id="user-1",
            title=f"Industrial AI source {index}",
            url=f"https://example.com/industrial-ai/{index}",
            snippet="A source-backed workflow with review gates.",
            platform="web",
            provider=f"provider-{index}",
            query="industrial AI workflow",
            relevance_score=0.9,
            importance_score=0.9,
            retrieved_at=f"2026-08-0{index}T00:00:00Z",
        )
        for index in range(1, source_count + 1)
    ]
    return DailyBriefing(
        id="briefing-1",
        topic_id="topic-1",
        user_id="user-1",
        chat_id="chat-1",
        topic="industrial AI",
        run_date=date(2026, 8, 6),
        search_directions=["industrial AI workflow"],
        keywords=["industrial AI"],
        sources=sources,
        synthesis=BriefingSynthesis(
            search_content_summary="Sources describe guarded industrial AI workflows.",
            short_summary="The workflow keeps human review before execution.",
            detailed_summary="Evidence supports a review-gated execution pattern.",
            themes=[
                BriefingTheme(
                    name="Review-gated execution",
                    analysis="Industrial AI workflows keep a human review gate before production execution.",
                    source_urls=[source.url for source in sources],
                )
            ],
            key_signal_interpretation="Review gates recur across the sources.",
            analysis_judgment="The pattern is useful planning context but still requires fresh source checks.",
            next_search_directions=["Find deployment metrics", "Check failure recovery"],
            landing_suggestions=["Start with review-only mode", "Record rollback outcomes"],
        ),
        markdown="# Industrial AI",
        created_at="2026-08-06T00:00:00Z",
    )
