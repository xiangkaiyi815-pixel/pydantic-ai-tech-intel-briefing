from search_assistant.contracts import AnswerPackage, IncomingMessage
from search_assistant.memory.store import MemoryStore


def test_records_interaction_answer_and_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is the latest FastAPI version?",
        source="cli",
    )

    question_id = store.record_interaction(message)
    package = AnswerPackage(
        question_id=question_id,
        answer_text="FastAPI is current only after checking PyPI.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["FastAPI latest version"],
        calibration={"ran": True, "critique": "Needs package source", "revision": "Marked unverified"},
        memory_updates=[],
    )
    store.record_answer(package)

    assert store.has_interaction("e-1")
    assert store.latest_answer_for_dedupe_key("e-1").question_id == question_id


def test_memory_supersession_keeps_old_record(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    old_id = store.add_memory_item("preference", "likes concise answers", "manual")
    new_id = store.add_memory_item("preference", "likes concise answers with sources", "manual", supersedes_id=old_id)

    rows = store.list_memory_items()
    assert [row["id"] for row in rows] == [old_id, new_id]
    assert rows[1]["supersedes_id"] == old_id
