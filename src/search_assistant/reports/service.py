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
            "## Important Answers",
        ]
        for answer in answers:
            lines.append(f"- ({answer.classification}, confidence {answer.confidence}) {answer.answer_text}")
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
