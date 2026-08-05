from __future__ import annotations

from pathlib import Path

from search_assistant.memory.store import MemoryStore


class ReportService:
    def __init__(self, store: MemoryStore, output_dir: str | Path = "reports"):
        self.store = store
        self.output_dir = Path(output_dir)

    def generate_markdown(self, start: str | None = None, end: str | None = None) -> str:
        answers = self.store.list_answers()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "learning-report.md"
        lines = [
            "# Learning Report",
            "",
            f"Question volume: {len(answers)}",
            "",
            "## Learning Direction",
        ]
        topics = self._learning_topics()
        if topics:
            lines.extend(f"- {topic}" for topic in topics)
        else:
            lines.append("- No learning direction detected yet.")
        lines.extend(
            [
                "",
                "## Recommended Next Learning Actions",
            ]
        )
        lines.extend(f"- {action}" for action in self._recommended_actions(topics))
        lines.extend(
            [
                "",
                "## Important Answers",
            ]
        )
        for answer in answers:
            lines.append(f"- ({answer.classification}, confidence {answer.confidence}) {answer.answer_text}")
        lines.extend(
            [
                "",
                "## Experience Notes",
            ]
        )
        experiences = self.store.list_experience_items()
        if experiences:
            for item in experiences:
                lines.append(f"- {item['title']}: {item['body']}")
        else:
            lines.append("- No experience notes recorded yet.")
        lines.extend(
            [
                "",
                "## Unresolved Or Weakly Verified Areas",
            ]
        )
        weak_claims = [claim for answer in answers for claim in answer.unverified_claims]
        if weak_claims:
            lines.extend(f"- {claim}" for claim in weak_claims)
        else:
            lines.append("- No weakly verified areas recorded.")
        markdown = "\n".join(lines) + "\n"
        report_path.write_text(markdown, encoding="utf-8")
        self.store.add_learning_report(markdown, str(report_path))
        return markdown

    def _learning_topics(self) -> list[str]:
        seen: set[str] = set()
        topics: list[str] = []
        for item in self.store.list_memory_items():
            content = item["content"]
            if content not in seen:
                seen.add(content)
                topics.append(content)
        return topics

    def _recommended_actions(self, topics: list[str]) -> list[str]:
        actions: list[str] = []
        topic_set = set(topics)
        if "Feishu integration" in topic_set:
            actions.append("Build a Feishu bot callback and reply checklist")
        if "Microsoft Agent Framework" in topic_set:
            actions.append("Map the Agent Framework workflow, runtime, and tool boundaries")
        if "Verification practice" in topic_set or "API reliability" in topic_set:
            actions.append("Create a key-data verification checklist before answers")
        if "Learning direction planning" in topic_set:
            actions.append("Turn repeated questions into a weekly learning plan")
        if "AI infrastructure and model deployment" in topic_set:
            actions.append("Build a model deployment feasibility checklist")
        if "AI memory architecture" in topic_set:
            actions.append("Map CXL, memory pooling, and accelerator memory tiers")
        if "Physical AI and world models" in topic_set:
            actions.append("Track world-model and embodied-AI releases against primary sources")
        if not actions:
            actions.append("Ask three focused questions in one topic so the assistant can identify a learning direction")
        return actions
