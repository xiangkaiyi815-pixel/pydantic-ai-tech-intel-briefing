from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_assistant.contracts import AnswerPackage, IncomingMessage


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    message_id TEXT NOT NULL,
                    event_id TEXT,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    raw_event_json TEXT NOT NULL,
                    answer_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS answers (
                    question_id TEXT PRIMARY KEY,
                    answer_text TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    package_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    source TEXT,
                    checked_at TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experience_items (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_snapshots (
                    id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_reports (
                    id TEXT PRIMARY KEY,
                    markdown_body TEXT NOT NULL,
                    path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skill_drafts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def record_interaction(self, message: IncomingMessage) -> str:
        question_id = _new_id("q")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    id, dedupe_key, message_id, event_id, user_id, chat_id, text,
                    source, raw_event_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    message.dedupe_key,
                    message.message_id,
                    message.event_id,
                    message.user_id,
                    message.chat_id,
                    message.text,
                    message.source,
                    json.dumps(message.raw_event, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return question_id

    def record_answer(self, package: AnswerPackage) -> None:
        payload = package.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO answers (
                    question_id, answer_text, classification, confidence, package_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    package.question_id,
                    package.answer_text,
                    package.classification,
                    package.confidence,
                    json.dumps(payload, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            connection.execute(
                "UPDATE interactions SET answer_id = ? WHERE id = ?",
                (package.question_id, package.question_id),
            )
            for claim in package.verified_claims:
                connection.execute(
                    """
                    INSERT INTO evidence (
                        id, question_id, claim, verdict, source, checked_at, notes, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id("ev"),
                        package.question_id,
                        claim.claim,
                        claim.verdict,
                        claim.source,
                        claim.checked_at,
                        claim.notes,
                        _now_iso(),
                    ),
                )

    def has_interaction(self, dedupe_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM interactions WHERE dedupe_key = ? LIMIT 1",
                (dedupe_key,),
            ).fetchone()
        return row is not None

    def latest_answer_for_dedupe_key(self, dedupe_key: str) -> AnswerPackage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.package_json
                FROM interactions i
                JOIN answers a ON a.question_id = i.answer_id
                WHERE i.dedupe_key = ?
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (dedupe_key,),
            ).fetchone()
        if row is None:
            return None
        return AnswerPackage.model_validate_json(row["package_json"])

    def add_memory_item(
        self,
        kind: str,
        content: str,
        source_id: str,
        supersedes_id: str | None = None,
    ) -> str:
        item_id = _new_id("mem")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_items (id, kind, content, source_id, supersedes_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, kind, content, source_id, supersedes_id, _now_iso()),
            )
        return item_id

    def list_memory_items(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_items ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_answers(self) -> list[AnswerPackage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT package_json FROM answers ORDER BY created_at ASC"
            ).fetchall()
        return [AnswerPackage.model_validate_json(row["package_json"]) for row in rows]

    def add_profile_snapshot(self, summary: dict[str, Any]) -> str:
        snapshot_id = _new_id("profile")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO profile_snapshots (id, summary_json, created_at) VALUES (?, ?, ?)",
                (snapshot_id, json.dumps(summary, ensure_ascii=False), _now_iso()),
            )
        return snapshot_id

    def list_profile_snapshots(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM profile_snapshots ORDER BY created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_learning_report(self, markdown_body: str, path: str | None = None) -> str:
        report_id = _new_id("report")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO learning_reports (id, markdown_body, path, created_at) VALUES (?, ?, ?, ?)",
                (report_id, markdown_body, path, _now_iso()),
            )
        return report_id

    def add_skill_draft(self, name: str, path: str, source_ids: list[str]) -> str:
        draft_id = _new_id("skill")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO skill_drafts (id, name, path, source_ids_json, review_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (draft_id, name, path, json.dumps(source_ids), "draft", _now_iso()),
            )
        return draft_id

    def list_skill_drafts(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM skill_drafts ORDER BY created_at ASC").fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
