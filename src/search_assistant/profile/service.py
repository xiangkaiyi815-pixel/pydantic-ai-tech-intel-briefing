from __future__ import annotations

from search_assistant.contracts import AnswerPackage
from search_assistant.memory.store import MemoryStore


class ProfileService:
    def __init__(self, store: MemoryStore):
        self.store = store

    def update_from_answer(self, package: AnswerPackage) -> str:
        topics = []
        for update in package.memory_updates:
            content = update.get("content") if isinstance(update, dict) else update.content
            if content:
                topics.append(content)
        if not topics and "Feishu" in package.answer_text:
            topics.append("Feishu")
        summary = {
            "recurring_topics": topics,
            "preferred_answer_style": "precise answers with verification notes",
            "confidence_notes": ["verify aggressively when claims are current or API-related"],
            "source_question_id": package.question_id,
        }
        return self.store.add_profile_snapshot(summary)
