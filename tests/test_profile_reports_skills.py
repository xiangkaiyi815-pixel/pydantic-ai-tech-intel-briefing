from search_assistant.contracts import AnswerPackage
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.skills.service import SkillDraftService
from search_assistant.contracts import IncomingMessage
from search_assistant.workflow.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


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


def test_learning_report_summarizes_direction_and_suggestions(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="需要围绕飞书 bot、Agent Framework、核验和工作流持续学习。"),
    )
    profile_service = ProfileService(store)
    questions = [
        "我想系统学习 Microsoft Agent Framework 和飞书 bot 集成，请根据我的问题判断学习方向。",
        "请比较飞书机器人事件订阅和消息回复 API 的风险，并告诉我下一步该学什么。",
        "How should I verify current API version answers before replying in Feishu?",
    ]
    for index, question in enumerate(questions, start=1):
        package = workflow.answer(
            IncomingMessage(
                message_id=f"learn-{index}",
                event_id=f"learn-{index}",
                user_id="u-1",
                chat_id="c-1",
                text=question,
                source="cli",
            )
        )
        profile_service.update_from_answer(package)

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()

    assert "## Learning Direction" in report
    assert "Feishu integration" in report
    assert "Microsoft Agent Framework" in report
    assert "## Recommended Next Learning Actions" in report
    assert "Build a Feishu bot callback and reply checklist" in report
