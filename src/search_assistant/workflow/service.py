from __future__ import annotations

from search_assistant.contracts import AnswerPackage, Classification, IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.verification.policy import (
    extract_key_claims,
    requires_calibration,
    requires_verification,
)
from search_assistant.workflow.runtime import AgentRuntime, FakeAgentRuntime


class SearchAssistantWorkflow:
    def __init__(self, store: MemoryStore, runtime: AgentRuntime | None = None):
        self.store = store
        self.runtime = runtime or FakeAgentRuntime()

    def answer(self, message: IncomingMessage) -> AnswerPackage:
        existing = self.store.latest_answer_for_dedupe_key(message.dedupe_key)
        if existing is not None:
            return existing

        question_id = self.store.record_interaction(message)
        classification = self._classify(message.text)
        memory_context = self.store.list_memory_items()
        context: dict[str, object] = {
            "question_id": question_id,
            "classification": classification,
            "memory": memory_context,
        }

        draft = self.runtime.generate_answer(message.text, context)
        claims = extract_key_claims(draft)
        unverified_claims = claims if requires_verification(draft, classification) else []
        calibration: dict[str, object] | None = None
        if requires_calibration(classification, bool(unverified_claims)):
            calibration = self.runtime.calibrate(
                draft,
                {
                    **context,
                    "draft": draft,
                    "unverified_claims": unverified_claims,
                },
            )

        package = AnswerPackage(
            question_id=question_id,
            answer_text=draft,
            classification=classification,
            confidence=self._confidence(classification, bool(unverified_claims)),
            verified_claims=[],
            unverified_claims=unverified_claims,
            calibration=calibration,
            memory_updates=self._memory_updates(message, question_id),
        )
        self.store.record_answer(package)
        for update in package.memory_updates:
            if isinstance(update, dict):
                self.store.add_memory_item(update["kind"], update["content"], question_id)
            else:
                self.store.add_memory_item(update.kind, update.content, question_id)
        return package

    def _classify(self, text: str) -> Classification:
        lowered = text.lower()
        if any(word in lowered for word in ("medical", "legal", "financial", "investment", "safety")):
            return "high_stakes"
        if any(word in lowered for word in ("compare", "migration", "risks", "architecture", "debug", "why")):
            return "hard"
        if any(word in lowered for word in ("latest", "current", "newest", "today", "2026", "api", "version")):
            return "research"
        return "simple"

    def _confidence(self, classification: Classification, has_unverified_claims: bool) -> str:
        if classification == "high_stakes" or has_unverified_claims:
            return "low"
        if classification in {"hard", "research"}:
            return "medium"
        return "high"

    def _memory_updates(self, message: IncomingMessage, question_id: str) -> list[dict[str, str]]:
        lowered = message.text.lower()
        updates: list[dict[str, str]] = []
        if "feishu" in lowered:
            updates.append({"kind": "topic", "content": "Feishu integration", "source_id": question_id})
        if "agent framework" in lowered:
            updates.append({"kind": "topic", "content": "Microsoft Agent Framework", "source_id": question_id})
        return updates
