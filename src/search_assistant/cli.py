from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from search_assistant.contracts import IncomingMessage
from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.feishu.events import parse_feishu_event
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.skills.service import SkillDraftService
from search_assistant.workflow.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="search-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    _add_data_dir(ask_parser)

    report_parser = subparsers.add_parser("report")
    _add_data_dir(report_parser)

    skill_parser = subparsers.add_parser("skill-draft")
    skill_parser.add_argument("title")
    _add_data_dir(skill_parser)

    fixture_parser = subparsers.add_parser("feishu-fixture")
    fixture_parser.add_argument("path")
    _add_data_dir(fixture_parser)

    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    store = _store(data_dir)

    if args.command == "ask":
        package = _answer_cli_question(store, args.question)
        print(json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        markdown = ReportService(store, output_dir=data_dir / "reports").generate_markdown()
        path = data_dir / "reports" / "learning-report.md"
        print(str(path))
        if not markdown:
            return 1
        return 0
    if args.command == "skill-draft":
        path = SkillDraftService(store, drafts_dir=data_dir / "skills" / "drafts").create_from_experience(
            args.title,
            source_ids=["manual"],
        )
        print(path)
        return 0
    if args.command == "feishu-fixture":
        reply = _run_feishu_fixture(store, Path(args.path))
        print(json.dumps(reply, ensure_ascii=False, indent=2))
        return 0
    return 2


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=".local-data")


def _store(data_dir: Path) -> MemoryStore:
    store = MemoryStore(data_dir / "assistant.sqlite3")
    store.initialize()
    return store


def _answer_cli_question(store: MemoryStore, question: str):
    workflow = SearchAssistantWorkflow(store=store, runtime=FakeAgentRuntime())
    message = IncomingMessage(
        message_id=f"cli_{uuid.uuid4().hex}",
        event_id=None,
        user_id="local-user",
        chat_id="local-cli",
        text=question,
        source="cli",
    )
    package = workflow.answer(message)
    ProfileService(store).update_from_answer(package)
    return package


def _run_feishu_fixture(store: MemoryStore, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    client = FakeFeishuClient()
    workflow = SearchAssistantWorkflow(store=store, runtime=FakeAgentRuntime())
    message = parse_feishu_event(payload)
    package = workflow.answer(message)
    return client.reply_text(message.message_id, package.answer_text)


if __name__ == "__main__":
    raise SystemExit(main())
