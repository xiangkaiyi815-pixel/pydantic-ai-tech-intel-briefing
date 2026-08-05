from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from search_assistant.contracts import IncomingMessage
from search_assistant.briefing.service import DailyBriefingService
from search_assistant.diagnostics.readiness import run_readiness_diagnostics
from search_assistant.evaluation.service import EvaluationService
from search_assistant.feishu.client import FakeFeishuClient, FeishuHttpClient
from search_assistant.feishu.events import parse_feishu_event
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.search.provider import search_client_from_settings
from search_assistant.skills.service import SkillDraftService
from search_assistant.config import Settings
from search_assistant.verification.backfill import VerificationBackfillService
from search_assistant.workflow.runtime import runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="search-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    _add_data_dir(ask_parser)

    topic_add_parser = subparsers.add_parser("topic-add")
    topic_add_parser.add_argument("topic")
    topic_add_parser.add_argument("--user-id", default="local-user")
    topic_add_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_add_parser)

    topic_list_parser = subparsers.add_parser("topic-list")
    topic_list_parser.add_argument("--user-id", default="local-user")
    topic_list_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_list_parser)

    briefing_parser = subparsers.add_parser("brief-run")
    briefing_parser.add_argument("topic")
    briefing_parser.add_argument("--user-id", default="local-user")
    briefing_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(briefing_parser)

    briefing_loop_parser = subparsers.add_parser("brief-loop")
    briefing_loop_parser.add_argument("topic")
    briefing_loop_parser.add_argument("--user-id", default="local-user")
    briefing_loop_parser.add_argument("--chat-id", default="local-cli")
    briefing_loop_parser.add_argument("--interval-seconds", type=float, default=86400.0)
    briefing_loop_parser.add_argument("--max-runs", type=int, default=None)
    _add_data_dir(briefing_loop_parser)

    feedback_parser = subparsers.add_parser("brief-feedback")
    feedback_parser.add_argument("topic")
    feedback_parser.add_argument("body")
    feedback_parser.add_argument("--url", default=None)
    feedback_parser.add_argument("--user-id", default="local-user")
    feedback_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(feedback_parser)

    report_parser = subparsers.add_parser("report")
    _add_data_dir(report_parser)

    evolve_parser = subparsers.add_parser("evolve")
    _add_data_dir(evolve_parser)

    evolve_loop_parser = subparsers.add_parser("evolve-loop")
    _add_data_dir(evolve_loop_parser)
    evolve_loop_parser.add_argument("--interval-seconds", type=float, default=86400.0)
    evolve_loop_parser.add_argument("--max-runs", type=int, default=None)

    eval_parser = subparsers.add_parser("eval-suite")
    eval_parser.add_argument("--questions", default=None)
    eval_parser.add_argument("--max-questions", type=_positive_int, default=None)
    _add_data_dir(eval_parser)

    evidence_backfill_parser = subparsers.add_parser("evidence-backfill")
    _add_data_dir(evidence_backfill_parser)

    skill_parser = subparsers.add_parser("skill-draft")
    skill_parser.add_argument("title")
    _add_data_dir(skill_parser)

    promote_parser = subparsers.add_parser("skill-promote")
    promote_parser.add_argument("name_or_slug")
    _add_data_dir(promote_parser)

    skill_list_parser = subparsers.add_parser("skill-list")
    _add_data_dir(skill_list_parser)

    fixture_parser = subparsers.add_parser("feishu-fixture")
    fixture_parser.add_argument("path")
    _add_data_dir(fixture_parser)

    long_connection_parser = subparsers.add_parser("feishu-long-connection")
    _add_data_dir(long_connection_parser)

    status_parser = subparsers.add_parser("feishu-status")
    _add_data_dir(status_parser)

    reply_probe_parser = subparsers.add_parser("feishu-reply-probe")
    reply_probe_target = reply_probe_parser.add_mutually_exclusive_group(required=True)
    reply_probe_target.add_argument("--message-id")
    reply_probe_target.add_argument("--latest", action="store_true")
    reply_probe_parser.add_argument("--text", default="Search assistant Feishu reply probe.")
    _add_data_dir(reply_probe_parser)

    wait_reply_parser = subparsers.add_parser("feishu-wait-reply")
    wait_reply_parser.add_argument("--channel", default="feishu-long-connection")
    wait_reply_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    wait_reply_parser.add_argument("--poll-seconds", type=float, default=2.0)
    _add_data_dir(wait_reply_parser)

    poll_once_parser = subparsers.add_parser("feishu-poll-once")
    poll_once_target = poll_once_parser.add_mutually_exclusive_group(required=True)
    poll_once_target.add_argument("--chat-id")
    poll_once_target.add_argument("--latest-chat", action="store_true")
    poll_once_parser.add_argument("--lookback-seconds", type=int, default=3600)
    _add_data_dir(poll_once_parser)

    poll_loop_parser = subparsers.add_parser("feishu-poll-loop")
    poll_loop_target = poll_loop_parser.add_mutually_exclusive_group(required=True)
    poll_loop_target.add_argument("--chat-id")
    poll_loop_target.add_argument("--latest-chat", action="store_true")
    poll_loop_parser.add_argument("--lookback-seconds", type=int, default=3600)
    poll_loop_parser.add_argument("--interval-seconds", type=float, default=5.0)
    poll_loop_parser.add_argument("--max-runs", type=int, default=None)
    _add_data_dir(poll_loop_parser)

    doctor_parser = subparsers.add_parser("doctor")
    _add_data_dir(doctor_parser)

    subparsers.add_parser("feishu-doctor")

    args = parser.parse_args(argv)
    if args.command == "feishu-doctor":
        report = run_feishu_diagnostics()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    data_dir = Path(args.data_dir) if args.data_dir else Settings.from_env().data_dir
    if args.command == "feishu-long-connection":
        run_long_connection(data_dir=data_dir)
        return 0

    store = _store(data_dir)

    if args.command == "ask":
        package = _answer_cli_question(store, args.question)
        print(json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "topic-add":
        subscription = store.upsert_topic(args.user_id, args.chat_id, args.topic)
        print(json.dumps(subscription.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "topic-list":
        topics = store.list_topics(args.user_id, args.chat_id)
        print(json.dumps([topic.model_dump(mode="json") for topic in topics], ensure_ascii=False, indent=2))
        return 0
    if args.command == "brief-feedback":
        feedback_id = _briefing_service(store).add_feedback(
            args.topic,
            args.user_id,
            args.chat_id,
            args.body,
            args.url,
        )
        print(json.dumps({"feedback_id": feedback_id, "topic": args.topic}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "brief-run":
        briefing, path = _run_briefing(store, args.topic, args.user_id, args.chat_id)
        print(json.dumps({"path": str(path), "sources": len(briefing.sources), "topic": briefing.topic}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "brief-loop":
        result = _run_briefing_loop(
            store,
            args.topic,
            args.user_id,
            args.chat_id,
            args.interval_seconds,
            args.max_runs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-status":
        print(json.dumps(_feishu_status(store), ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-reply-probe":
        result = _run_feishu_reply_probe(
            store,
            message_id=args.message_id,
            use_latest=args.latest,
            text=args.text,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "feishu-wait-reply":
        result = _wait_for_reply_attempt(
            store,
            channel=args.channel,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "feishu-poll-once":
        result = _run_feishu_poll_once(
            store,
            chat_id=_resolve_poll_chat_id(store, args.chat_id, args.latest_chat),
            lookback_seconds=args.lookback_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("failed", 0) == 0 else 1
    if args.command == "feishu-poll-loop":
        result = _run_feishu_poll_loop(
            store,
            chat_id=_resolve_poll_chat_id(store, args.chat_id, args.latest_chat),
            lookback_seconds=args.lookback_seconds,
            interval_seconds=args.interval_seconds,
            max_runs=args.max_runs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "doctor":
        report = run_readiness_diagnostics(store, data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    if args.command == "report":
        markdown = ReportService(store, output_dir=data_dir / "reports").generate_markdown()
        path = data_dir / "reports" / "learning-report.md"
        print(str(path))
        if not markdown:
            return 1
        return 0
    if args.command == "evolve":
        result = _run_evolution(store, data_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evolve-loop":
        result = _run_evolution_loop(
            store,
            data_dir,
            interval_seconds=args.interval_seconds,
            max_runs=args.max_runs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-suite":
        result = _run_evaluation(
            store,
            data_dir,
            questions_source=args.questions,
            max_questions=args.max_questions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evidence-backfill":
        result = VerificationBackfillService(store).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "skill-draft":
        path = SkillDraftService(store, drafts_dir=data_dir / "skills" / "drafts").create_from_experience(
            args.title,
            source_ids=_skill_source_ids(store),
        )
        print(path)
        return 0
    if args.command == "skill-promote":
        service = SkillDraftService(
            store,
            drafts_dir=data_dir / "skills" / "drafts",
            active_dir=data_dir / "skills" / "active",
        )
        path = service.promote(args.name_or_slug)
        draft = _skill_draft_by_path(store, path)
        print(
            json.dumps(
                {
                    "status": "promoted",
                    "name": draft["name"] if draft else args.name_or_slug,
                    "path": path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "skill-list":
        result = SkillDraftService(
            store,
            drafts_dir=data_dir / "skills" / "drafts",
            active_dir=data_dir / "skills" / "active",
        ).list_review_status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-fixture":
        reply = _run_feishu_fixture(store, Path(args.path))
        print(json.dumps(reply, ensure_ascii=False, indent=2))
        return 0
    return 2


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _store(data_dir: Path) -> MemoryStore:
    store = MemoryStore(data_dir / "assistant.sqlite3")
    store.initialize()
    return store


def _briefing_service(store: MemoryStore) -> DailyBriefingService:
    settings = Settings.from_env()
    return DailyBriefingService(
        store=store,
        search_client=search_client_from_settings(settings),
        runtime=runtime_from_settings(settings),
        max_queries=settings.briefing_max_queries,
        results_per_query=settings.briefing_results_per_query,
        max_sources=settings.briefing_max_sources,
        model_max_sources=settings.briefing_model_max_sources,
        search_budget_seconds=settings.briefing_search_budget_seconds,
        timezone_name=settings.briefing_timezone,
    )


def _run_briefing(store: MemoryStore, topic: str, user_id: str, chat_id: str):
    briefing = _briefing_service(store).run(topic, user_id, chat_id)
    output_dir = _store_data_dir(store) / "briefs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{briefing.run_date.isoformat()}-{_safe_filename(topic)}.md"
    path.write_text(briefing.markdown, encoding="utf-8")
    return briefing, path


def _run_briefing_loop(
    store: MemoryStore,
    topic: str,
    user_id: str,
    chat_id: str,
    interval_seconds: float,
    max_runs: int | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "topic": topic,
        "user_id": user_id,
        "chat_id": chat_id,
        "interval_seconds": interval_seconds,
        "max_runs": max_runs,
    }
    session_id = store.record_runtime_session("brief-loop", _current_process_id(), "started", metadata)
    results: list[dict[str, object]] = []
    runs = 0
    try:
        while max_runs is None or runs < max_runs:
            briefing, path = _run_briefing(store, topic, user_id, chat_id)
            results.append({"path": str(path), "sources": len(briefing.sources), "run_date": briefing.run_date.isoformat()})
            runs += 1
            if max_runs is not None and runs >= max_runs:
                break
            time.sleep(max(1.0, interval_seconds))
    finally:
        metadata["runs"] = runs
        store.update_runtime_session_status(session_id, "completed", metadata)
    return {"runtime_session_id": session_id, "runs": runs, "results": results}


def _safe_filename(value: str) -> str:
    slug = "-".join(part for part in "".join(char if char.isalnum() else " " for char in value).split())
    return slug[:80] or "briefing"


def _answer_cli_question(store: MemoryStore, question: str):
    settings = Settings.from_env()
    data_dir = _store_data_dir(store)
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
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
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    client = FakeFeishuClient()
    settings = Settings.from_env()
    data_dir = _store_data_dir(store)
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    message = parse_feishu_event(payload)
    package = workflow.answer(message)
    ProfileService(store).update_from_answer(package)
    return client.reply_text(message.message_id, package.answer_text)


def _run_evaluation(
    store: MemoryStore,
    data_dir: Path,
    questions_source: str | None = None,
    max_questions: int | None = None,
) -> dict[str, Any]:
    settings = Settings.from_env()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    service = EvaluationService(
        store=store,
        workflow=workflow,
        output_dir=data_dir / "evaluations",
        report_output_dir=data_dir / "reports",
    )
    return service.run(
        _load_evaluation_questions(questions_source) if questions_source else None,
        max_questions=max_questions,
    )


def _load_evaluation_questions(source: str | Path) -> list[str]:
    source_text = str(source).strip()
    if source_text.startswith("["):
        data = json.loads(source_text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Evaluation questions JSON must be a list of strings")
        return [item.strip() for item in data if item.strip()]

    path = Path(source)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Evaluation questions JSON must be a list of strings")
        return [item.strip() for item in data if item.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _feishu_status(store: MemoryStore) -> dict[str, Any]:
    return {
        "counts": store.diagnostic_counts(),
        "latest_interactions": [_safe_interaction(row) for row in store.list_interactions(limit=5)],
        "latest_reply_attempts": [_safe_reply_attempt(row) for row in store.list_reply_attempts(limit=5)],
        "latest_runtime_sessions": store.list_runtime_sessions(limit=5),
    }


def _safe_interaction(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    return {
        "question_id": row.get("id"),
        "message_id": row.get("message_id"),
        "event_id": row.get("event_id"),
        "source": row.get("source"),
        "chat_id": row.get("chat_id"),
        "answer_id": row.get("answer_id"),
        "created_at": row.get("created_at"),
        "text_preview": _preview_text(text),
    }


def _preview_text(text: str, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _safe_reply_attempt(row: dict[str, Any]) -> dict[str, Any]:
    response_json = row.get("response_json") or "{}"
    return {
        "id": row.get("id"),
        "question_id": row.get("question_id"),
        "message_id": row.get("message_id"),
        "channel": row.get("channel"),
        "status": row.get("status"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "response_recorded": response_json not in {"", "{}"},
    }


def _run_feishu_reply_probe(
    store: MemoryStore,
    message_id: str | None,
    use_latest: bool,
    text: str,
) -> dict[str, Any]:
    question_id = "manual-probe"
    resolved_message_id = message_id
    if use_latest:
        interactions = store.list_interactions(limit=1)
        if not interactions:
            return {
                "ok": False,
                "detail": "no interaction exists for --latest",
            }
        latest = interactions[0]
        question_id = latest["id"]
        resolved_message_id = latest["message_id"]
    if not resolved_message_id:
        return {"ok": False, "detail": "message id is required"}

    try:
        response = feishu_client_from_settings(Settings.from_env()).reply_text(resolved_message_id, text)
    except Exception as exc:
        store.record_reply_attempt(
            question_id=question_id,
            message_id=resolved_message_id,
            channel="feishu-probe",
            status="failure",
            error=_safe_cli_error(exc),
        )
        return {
            "ok": False,
            "question_id": question_id,
            "message_id": resolved_message_id,
            "channel": "feishu-probe",
            "error": _safe_cli_error(exc),
        }

    store.record_reply_attempt(
        question_id=question_id,
        message_id=resolved_message_id,
        channel="feishu-probe",
        status="success",
        response=response,
    )
    return {
        "ok": True,
        "question_id": question_id,
        "message_id": resolved_message_id,
        "channel": "feishu-probe",
    }


def _wait_for_reply_attempt(
    store: MemoryStore,
    channel: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    not_before = _latest_started_runtime_created_at(store, channel)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        matching = [
            attempt
            for attempt in store.list_reply_attempts()
            if attempt["channel"] == channel and attempt["status"] == "success"
            if not_before is None or str(attempt.get("created_at", "")) > not_before
        ]
        if matching:
            return {
                "ok": True,
                "channel": channel,
                "not_before": not_before,
                "attempt": _safe_reply_attempt(matching[0]),
            }

        latest_attempts = store.list_reply_attempts(limit=1)
        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "channel": channel,
                "detail": f"timed out waiting for successful {channel} reply attempt",
                "not_before": not_before,
                "latest_attempt": _safe_reply_attempt(latest_attempts[0]) if latest_attempts else None,
            }
        time.sleep(max(0.1, poll_seconds))


def _latest_started_runtime_created_at(store: MemoryStore, kind: str) -> str | None:
    for session in store.list_runtime_sessions():
        if session.get("kind") == kind and session.get("status") == "started":
            return str(session.get("created_at") or "")
    return None


def _resolve_poll_chat_id(store: MemoryStore, chat_id: str | None, latest_chat: bool) -> str:
    if chat_id:
        return chat_id
    if latest_chat:
        interactions = [
            row
            for row in store.list_interactions()
            if str(row.get("source", "")).startswith("feishu-") and str(row.get("chat_id", "")).startswith("oc_")
        ]
        if not interactions:
            raise RuntimeError("No stored interaction exists for --latest-chat")
        return str(interactions[0]["chat_id"])
    raise RuntimeError("chat id is required")


def _run_feishu_poll_once(store: MemoryStore, chat_id: str, lookback_seconds: int) -> dict[str, Any]:
    settings = Settings.from_env()
    from search_assistant.feishu.polling import FeishuPollingService

    service = FeishuPollingService(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        feishu_client=feishu_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        admin_user_ids=set(settings.admin_user_ids),
    )
    return service.poll_once(chat_id=chat_id, lookback_seconds=lookback_seconds)


def _run_feishu_poll_loop(
    store: MemoryStore,
    chat_id: str,
    lookback_seconds: int,
    interval_seconds: float,
    max_runs: int | None,
) -> dict[str, Any]:
    runs = 0
    results: list[dict[str, Any]] = []
    while max_runs is None or runs < max_runs:
        results.append(_run_feishu_poll_once(store, chat_id=chat_id, lookback_seconds=lookback_seconds))
        runs += 1
        if max_runs is not None and runs >= max_runs:
            break
        time.sleep(interval_seconds)
    return {"runs": runs, "chat_id": chat_id, "results": results}


def feishu_client_from_settings(settings: Settings):
    return FeishuHttpClient(settings.feishu_app_id, settings.feishu_app_secret)


def _safe_cli_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("tenant_access_token", "Authorization", "app_secret"):
        text = text.replace(marker, "[redacted]")
    return text


def _store_data_dir(store: MemoryStore) -> Path:
    return Path(store.database_path).parent


def _skill_source_ids(store: MemoryStore) -> list[str]:
    profiles = store.list_profile_snapshots()
    if profiles:
        return [profiles[-1]["id"]]
    answers = store.list_answers()
    if answers:
        return [answers[-1].question_id]
    return ["manual"]


def _skill_draft_by_path(store: MemoryStore, active_path: str) -> dict[str, Any] | None:
    active_slug = Path(active_path).parent.name
    for draft in store.list_skill_drafts():
        if Path(str(draft["path"])).parent.name == active_slug:
            return draft
    return None


def _run_evolution(store: MemoryStore, data_dir: Path) -> dict[str, object]:
    markdown = ReportService(store, output_dir=data_dir / "reports").generate_markdown()
    report_path = data_dir / "reports" / "learning-report.md"
    skill_service = SkillDraftService(store, drafts_dir=data_dir / "skills" / "drafts")
    refreshed_skill_paths = skill_service.refresh_reviewable_drafts()
    skill_paths = skill_service.auto_create_from_experience(refresh_existing=False)
    return {
        "report_path": str(report_path),
        "report_written": bool(markdown),
        "skill_paths": skill_paths,
        "refreshed_skill_paths": refreshed_skill_paths,
    }


def _run_evolution_loop(
    store: MemoryStore,
    data_dir: Path,
    interval_seconds: float,
    max_runs: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "interval_seconds": interval_seconds,
        "max_runs": max_runs,
    }
    session_id = store.record_runtime_session(
        "evolve-loop",
        process_id=_current_process_id(),
        status="started",
        metadata=metadata,
    )
    runs = 0
    results: list[dict[str, object]] = []
    while max_runs is None or runs < max_runs:
        results.append(_run_evolution(store, data_dir))
        runs += 1
        if max_runs is not None and runs >= max_runs:
            break
        time.sleep(interval_seconds)
    metadata["runs"] = runs
    store.update_runtime_session_status(session_id, "completed", metadata=metadata)
    return {
        "runs": runs,
        "interval_seconds": interval_seconds,
        "results": results,
        "runtime_session_id": session_id,
    }


def run_long_connection(data_dir: Path | None = None) -> None:
    from search_assistant.feishu.long_connection import run_long_connection as run

    run(data_dir=data_dir)


def run_feishu_diagnostics(settings=None) -> dict[str, Any]:
    from search_assistant.feishu.diagnostics import run_feishu_diagnostics as run

    return run(settings=settings)


def _current_process_id() -> int:
    import os

    return os.getpid()


if __name__ == "__main__":
    raise SystemExit(main())
