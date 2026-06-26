from __future__ import annotations

from typing import Protocol


class AgentRuntime(Protocol):
    def generate_answer(self, question: str, context: dict[str, object]) -> str:
        ...

    def calibrate(self, draft: str, context: dict[str, object]) -> dict[str, object]:
        ...


class FakeAgentRuntime:
    def __init__(self, answer_text: str = "This is a local deterministic answer."):
        self.answer_text = answer_text
        self.answer_calls = 0
        self.calibration_calls = 0

    def generate_answer(self, question: str, context: dict[str, object]) -> str:
        self.answer_calls += 1
        return self.answer_text

    def calibrate(self, draft: str, context: dict[str, object]) -> dict[str, object]:
        self.calibration_calls += 1
        unverified = context.get("unverified_claims") or []
        critique = "No blocking issues found."
        if unverified:
            critique = "Draft contains claims that need verification."
        return {
            "ran": True,
            "critique": critique,
            "revision": draft,
        }


class MicrosoftAgentRuntime:
    """Thin production adapter boundary for Microsoft Agent Framework.

    Phase 1 keeps business workflow deterministic and testable. Real model
    wiring lives behind this adapter so the rest of the app does not depend on
    fast-moving provider details.
    """

    def __init__(self, model_provider: str | None = None):
        try:
            import agent_framework
        except ImportError as exc:
            raise RuntimeError("agent-framework is required for MicrosoftAgentRuntime") from exc

        self.agent_framework = agent_framework
        self.model_provider = model_provider
        self.workflow_type = getattr(agent_framework, "FunctionalWorkflow", None)

    def generate_answer(self, question: str, context: dict[str, object]) -> str:
        if not self.model_provider:
            raise RuntimeError("MicrosoftAgentRuntime requires a configured model provider")
        raise RuntimeError("Production model wiring is not configured in Phase 1 local mode")

    def calibrate(self, draft: str, context: dict[str, object]) -> dict[str, object]:
        if not self.model_provider:
            raise RuntimeError("MicrosoftAgentRuntime requires a configured model provider")
        raise RuntimeError("Production calibration wiring is not configured in Phase 1 local mode")
