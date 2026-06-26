from search_assistant.contracts import IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.workflow.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def test_hard_question_runs_calibration_and_persists_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="The answer depends on verified API behavior in 2026.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="Compare the newest Agent Framework workflow API with Semantic Kernel and give migration risks.",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "hard"
    assert package.calibration["ran"] is True
    assert runtime.calibration_calls == 1
    assert store.latest_answer_for_dedupe_key("e-1").question_id == package.question_id


def test_duplicate_event_returns_stored_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="First answer.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="simple note",
        source="cli",
    )

    first = workflow.answer(message)
    second = workflow.answer(message)

    assert second.question_id == first.question_id
    assert runtime.answer_calls == 1
