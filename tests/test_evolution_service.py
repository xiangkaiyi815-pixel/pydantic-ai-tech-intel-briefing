from datetime import date
import json
import sqlite3
import sys
from io import BytesIO, TextIOWrapper

import pytest

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEvidence,
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
    graph_links = store.list_domain_candidate_graph_links(first_ids[0])
    assert len(graph_links) == 1
    assert graph_links[0]["graph_id"] == "industrial-ai"
    assert graph_links[0]["entity_id"] in {"mes", "controlled-work-order-orchestration"}
    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="candidate graph links are immutable"):
            connection.execute(
                "UPDATE domain_knowledge_candidate_graph_links SET note = 'changed' WHERE candidate_id = ?",
                (first_ids[0],),
            )

    validation = service.validate(first_ids[0])
    service.deprecate(first_ids[0], "superseded by a later evidence review")

    assert validation["validated"] is True
    assert validation["candidate_id"] == first_ids[0]
    assert validation["failures"] == []
    assert validation["knowledge_layer"] == "validated_knowledge"
    assert store.get_domain_knowledge_candidate(first_ids[0])["status"] == "deprecated"
    gates = store.list_gate_records()
    assert {gate["gate_type"] for gate in gates} == {
        "domain_knowledge_candidate_validation",
        "domain_knowledge_candidate_rollback",
    }
    assert all(gate["result"] == "passed" for gate in gates)
    rollback_entries = store.list_project_ledger_entries(entry_type="knowledge_rollback")
    assert rollback_entries[0]["status"] == "deprecated"
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
    gates = store.list_gate_records(gate_type="domain_knowledge_candidate_validation")
    assert len(gates) == 1
    assert gates[0]["result"] == "failed"
    assert gates[0]["metadata"]["evidence_url_count"] == 1
    assert gates[0]["metadata"]["knowledge_layer"] == "weak_signal"
    assert "exploratory clue" in gates[0]["metadata"]["intended_use"]
    assert result["knowledge_layer"] == "weak_signal"


def test_candidate_list_can_surface_weak_signals_without_promoting_them(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]

    service.record_validation_gate(candidate_id)
    weak_signals = service.list_candidates(layer="weak_signal")

    assert [candidate["id"] for candidate in weak_signals] == [candidate_id]
    assert weak_signals[0]["status"] == "candidate"
    assert weak_signals[0]["knowledge_layer"] == "weak_signal"
    assert "independent corroborating sources" in weak_signals[0]["knowledge_layer_next_action"]


def test_cli_candidate_list_filters_by_knowledge_layer(tmp_path, monkeypatch):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]
    service.record_validation_gate(candidate_id)
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["knowledge-candidate-list", "--layer", "weak_signal", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert [candidate["id"] for candidate in output] == [candidate_id]
    assert output[0]["knowledge_layer"] == "weak_signal"


def test_candidate_approval_records_human_gate_and_release_ledger(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=2))[0]

    result = service.approve(
        candidate_id,
        reviewer="unit-reviewer",
        reason="safe to use as planning context",
        eval_gate={"passed": True, "reason": "unit eval passed", "report_path": "unit-evaluation-report.json"},
    )

    assert result["approved"] is True
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "validated"
    gates = store.list_gate_records()
    gate_types = {gate["gate_type"] for gate in gates}
    assert "domain_knowledge_candidate_eval_release" in gate_types
    assert "domain_knowledge_candidate_validation" in gate_types
    assert "domain_knowledge_candidate_human_review" in gate_types
    release_entries = store.list_project_ledger_entries(entry_type="knowledge_release")
    assert len(release_entries) == 1
    assert release_entries[0]["status"] == "approved"
    assert result["gate_id"] in release_entries[0]["evidence_refs"]
    assert result["eval_gate"]["gate_id"] in release_entries[0]["evidence_refs"]


def test_candidate_approval_requires_eval_gate_before_release(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=2))[0]

    result = service.approve(candidate_id, reviewer="unit-reviewer")

    assert result["approved"] is False
    assert result["eval_gate"]["passed"] is False
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "candidate"
    gates = store.list_gate_records(gate_type="domain_knowledge_candidate_eval_release")
    assert len(gates) == 1
    assert gates[0]["result"] == "failed"
    assert not store.list_project_ledger_entries(entry_type="knowledge_release")


def test_evolve_command_backfills_existing_candidate_graph_links(tmp_path, capsys):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = "2026-08-06T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-medical-imaging",
        topic="医学影像大模型",
        claim="DICOM/PACS data boundaries shape how medical imaging foundation models enter clinical workflows.",
        applies_when="Use when researching medical imaging AI deployment and validation.",
        evidence=[
            DomainKnowledgeEvidence(
                title="Medical imaging model deployment",
                url="https://example.com/medical-imaging-model",
                provider="example",
                retrieved_at=now,
            )
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-medical-imaging"],
        fingerprint="medical-imaging-candidate-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)

    assert cli.main(["evolve", "--data-dir", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["domain_knowledge_candidate_graph_links"] == {"created": 1, "total": 1}
    links = store.list_domain_candidate_graph_links(candidate.id)
    assert len(links) == 1
    assert links[0]["graph_id"] == "medical-imaging-ai"
    assert links[0]["entity_id"] in {"dicom-pacs", "multimodal-medical-imaging-model"}


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
    assert "graph links: 1" in report


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
                    analysis=(
                        "Industrial AI workflows keep a human review gate before writing work orders "
                        "to MES / Manufacturing Execution System production execution."
                    ),
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
