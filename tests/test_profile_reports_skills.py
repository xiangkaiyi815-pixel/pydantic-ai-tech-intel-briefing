from search_assistant.contracts import AnswerPackage
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.skills.service import SkillDraftService


def test_profile_report_and_skill_draft_are_persisted(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-1",
        answer_text="You ask about Feishu and agent workflows.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["Feishu API detail"],
        calibration={"ran": True, "critique": "Needs source", "revision": "Marked unverified"},
        memory_updates=[{"kind": "topic", "content": "Feishu agent workflow"}],
    )
    store.record_answer(package)

    profile_id = ProfileService(store).update_from_answer(package)
    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()
    draft_path = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        title="Feishu Workflow Answers",
        source_ids=[profile_id],
    )

    assert "Feishu" in report
    assert draft_path.endswith("SKILL.md")
    assert store.list_profile_snapshots()[0]["id"] == profile_id
    assert store.list_skill_drafts()[0]["review_status"] == "draft"
