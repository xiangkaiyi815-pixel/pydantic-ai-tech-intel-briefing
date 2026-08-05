from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_assistant.contracts import (
    AnswerPackage,
    CollectedSource,
    DailyBriefing,
    IncomingMessage,
    TopicSubscription,
    VerifiedClaim,
)


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

                CREATE TABLE IF NOT EXISTS reply_attempts (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_sessions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS topic_subscriptions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, chat_id, topic)
                );

                CREATE TABLE IF NOT EXISTS topic_feedback (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_url TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS collected_sources (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    importance_score REAL NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(topic_id, url)
                );

                CREATE TABLE IF NOT EXISTS daily_briefings (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    package_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(topic_id, run_date)
                );
                """
            )
            self._ensure_column(connection, "memory_items", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "memory_items", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "experience_items", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "experience_items", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "skill_drafts", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "skill_drafts", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def record_interaction(self, message: IncomingMessage) -> str:
        question_id = _new_id("q")
        with self._connect() as connection:
            try:
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
            except sqlite3.IntegrityError:
                row = connection.execute(
                    """
                    SELECT id FROM interactions
                    WHERE dedupe_key = ? OR message_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (message.dedupe_key, message.message_id),
                ).fetchone()
                if row is None:
                    raise
                return str(row["id"])
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
            self._insert_missing_verified_claims(connection, package.question_id, package.verified_claims)

    def has_interaction(self, dedupe_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM interactions WHERE dedupe_key = ? OR message_id = ? LIMIT 1",
                (dedupe_key, dedupe_key),
            ).fetchone()
        return row is not None

    def list_interactions(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM interactions ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

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

    def latest_answer_for_message_id(self, message_id: str) -> AnswerPackage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.package_json
                FROM interactions i
                JOIN answers a ON a.question_id = i.answer_id
                WHERE i.message_id = ?
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (message_id,),
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
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        item_id = _new_id("mem")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_items (
                    id, kind, content, source_id, supersedes_id, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, kind, content, source_id, supersedes_id, user_id, chat_id, _now_iso()),
            )
        return item_id

    def list_memory_items(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM memory_items"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_answers(self) -> list[AnswerPackage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT package_json FROM answers ORDER BY created_at ASC"
            ).fetchall()
        return [AnswerPackage.model_validate_json(row["package_json"]) for row in rows]

    def list_evidence(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_answer_verification(
        self,
        question_id: str,
        verified_claims: list[VerifiedClaim],
        unverified_claims: list[str],
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT package_json FROM answers WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"answer not found: {question_id}")
            package = AnswerPackage.model_validate_json(row["package_json"])
            package.verified_claims = verified_claims
            package.unverified_claims = unverified_claims
            payload = package.model_dump(mode="json")
            connection.execute(
                """
                UPDATE answers
                SET package_json = ?
                WHERE question_id = ?
                """,
                (json.dumps(payload, ensure_ascii=False), question_id),
            )
            self._delete_stale_verified_claims(connection, question_id, verified_claims)
            return self._insert_missing_verified_claims(connection, question_id, verified_claims)

    def add_experience_item(
        self,
        title: str,
        body: str,
        source_ids: list[str],
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        experience_id = _new_id("exp")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experience_items (
                    id, title, body, source_ids_json, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience_id,
                    title,
                    body,
                    json.dumps(source_ids, ensure_ascii=False),
                    user_id,
                    chat_id,
                    _now_iso(),
                ),
            )
        return experience_id

    def list_experience_items(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experience_items"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def record_reply_attempt(
        self,
        question_id: str,
        message_id: str,
        channel: str,
        status: str,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        attempt_id = _new_id("reply")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reply_attempts (
                    id, question_id, message_id, channel, status, response_json, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    question_id,
                    message_id,
                    channel,
                    status,
                    json.dumps(response or {}, ensure_ascii=False),
                    error,
                    _now_iso(),
                ),
            )
        return attempt_id

    def list_reply_attempts(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM reply_attempts ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def has_successful_reply(self, message_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reply_attempts WHERE message_id = ? AND status = 'success' LIMIT 1",
                (message_id,),
            ).fetchone()
        return row is not None

    def record_runtime_session(
        self,
        kind: str,
        process_id: int,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = _new_id("runtime")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_sessions (
                    id, kind, process_id, status, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    process_id,
                    status,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return session_id

    def update_runtime_session_status(
        self,
        session_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            if metadata is None:
                connection.execute(
                    "UPDATE runtime_sessions SET status = ? WHERE id = ?",
                    (status, session_id),
                )
            else:
                connection.execute(
                    "UPDATE runtime_sessions SET status = ?, metadata_json = ? WHERE id = ?",
                    (status, json.dumps(metadata, ensure_ascii=False), session_id),
                )

    def list_runtime_sessions(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM runtime_sessions ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def diagnostic_counts(self) -> dict[str, int]:
        tables = (
            "interactions",
            "answers",
            "evidence",
            "memory_items",
            "experience_items",
            "profile_snapshots",
            "learning_reports",
            "skill_drafts",
            "reply_attempts",
            "runtime_sessions",
            "topic_subscriptions",
            "topic_feedback",
            "collected_sources",
            "daily_briefings",
        )
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

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

    def list_learning_reports(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM learning_reports ORDER BY created_at ASC").fetchall()
        return [dict(row) for row in rows]

    def add_skill_draft(
        self,
        name: str,
        path: str,
        source_ids: list[str],
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        draft_id = _new_id("skill")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO skill_drafts (
                    id, name, path, source_ids_json, review_status, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (draft_id, name, path, json.dumps(source_ids), "draft", user_id, chat_id, _now_iso()),
            )
        return draft_id

    def update_skill_draft_status(self, draft_id: str, review_status: str, path: str | None = None) -> None:
        with self._connect() as connection:
            if path is None:
                connection.execute(
                    "UPDATE skill_drafts SET review_status = ? WHERE id = ?",
                    (review_status, draft_id),
                )
            else:
                connection.execute(
                    "UPDATE skill_drafts SET review_status = ?, path = ? WHERE id = ?",
                    (review_status, path, draft_id),
                )

    def list_skill_drafts(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM skill_drafts"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def upsert_topic(self, user_id: str, chat_id: str, topic: str, enabled: bool = True) -> TopicSubscription:
        normalized = " ".join(topic.split())
        if not normalized:
            raise ValueError("topic must not be empty")
        now = _now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM topic_subscriptions WHERE user_id = ? AND chat_id = ? AND topic = ?",
                (user_id, chat_id, normalized),
            ).fetchone()
            if row is None:
                topic_id = _new_id("topic")
                connection.execute(
                    """
                    INSERT INTO topic_subscriptions (id, user_id, chat_id, topic, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (topic_id, user_id, chat_id, normalized, int(enabled), now, now),
                )
                row = connection.execute("SELECT * FROM topic_subscriptions WHERE id = ?", (topic_id,)).fetchone()
            else:
                connection.execute(
                    "UPDATE topic_subscriptions SET enabled = ?, updated_at = ? WHERE id = ?",
                    (int(enabled), now, row["id"]),
                )
                row = connection.execute("SELECT * FROM topic_subscriptions WHERE id = ?", (row["id"],)).fetchone()
        return self._topic_from_row(row)

    def list_topics(self, user_id: str, chat_id: str, enabled_only: bool = False) -> list[TopicSubscription]:
        query = "SELECT * FROM topic_subscriptions WHERE user_id = ? AND chat_id = ?"
        parameters: tuple[object, ...] = (user_id, chat_id)
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY updated_at DESC, id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._topic_from_row(row) for row in rows]

    def add_topic_feedback(
        self,
        topic_id: str,
        user_id: str,
        chat_id: str,
        body: str,
        source_url: str | None = None,
    ) -> str:
        feedback_id = _new_id("feedback")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO topic_feedback (id, topic_id, user_id, chat_id, body, source_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, topic_id, user_id, chat_id, body, source_url, _now_iso()),
            )
        return feedback_id

    def list_topic_feedback(self, topic_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM topic_feedback WHERE topic_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (topic_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_collected_source(self, source: CollectedSource) -> CollectedSource:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM collected_sources WHERE topic_id = ? AND url = ?",
                (source.topic_id, source.url),
            ).fetchone()
            source_id = str(existing["id"]) if existing else source.id
            connection.execute(
                """
                INSERT INTO collected_sources (
                    id, topic_id, user_id, title, url, snippet, platform, provider, query_text,
                    relevance_score, importance_score, retrieved_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic_id, url) DO UPDATE SET
                    title = excluded.title,
                    snippet = excluded.snippet,
                    platform = excluded.platform,
                    provider = excluded.provider,
                    query_text = excluded.query_text,
                    relevance_score = excluded.relevance_score,
                    importance_score = excluded.importance_score,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    source_id,
                    source.topic_id,
                    source.user_id,
                    source.title,
                    source.url,
                    source.snippet,
                    source.platform,
                    source.provider,
                    source.query,
                    source.relevance_score,
                    source.importance_score,
                    source.retrieved_at,
                    _now_iso(),
                ),
            )
        return source.model_copy(update={"id": source_id})

    def list_collected_sources(self, topic_id: str, limit: int = 100) -> list[CollectedSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM collected_sources WHERE topic_id = ?
                ORDER BY importance_score DESC, retrieved_at DESC, id DESC LIMIT ?
                """,
                (topic_id, limit),
            ).fetchall()
        return [
            CollectedSource(
                id=str(row["id"]),
                topic_id=str(row["topic_id"]),
                user_id=str(row["user_id"]),
                title=str(row["title"]),
                url=str(row["url"]),
                snippet=str(row["snippet"]),
                platform=str(row["platform"]),
                provider=str(row["provider"]),
                query=str(row["query_text"]),
                relevance_score=float(row["relevance_score"]),
                importance_score=float(row["importance_score"]),
                retrieved_at=str(row["retrieved_at"]),
            )
            for row in rows
        ]

    def record_daily_briefing(self, briefing: DailyBriefing) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO daily_briefings (
                    id, topic_id, user_id, chat_id, run_date, markdown, package_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic_id, run_date) DO UPDATE SET
                    markdown = excluded.markdown,
                    package_json = excluded.package_json,
                    created_at = excluded.created_at
                """,
                (
                    briefing.id,
                    briefing.topic_id,
                    briefing.user_id,
                    briefing.chat_id,
                    briefing.run_date.isoformat(),
                    briefing.markdown,
                    json.dumps(briefing.model_dump(mode="json"), ensure_ascii=False),
                    briefing.created_at,
                ),
            )

    def latest_daily_briefing(self, topic_id: str) -> DailyBriefing | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT package_json FROM daily_briefings WHERE topic_id = ? ORDER BY run_date DESC LIMIT 1",
                (topic_id,),
            ).fetchone()
        return DailyBriefing.model_validate_json(row["package_json"]) if row is not None else None

    @staticmethod
    def _topic_from_row(row: sqlite3.Row) -> TopicSubscription:
        return TopicSubscription(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            chat_id=str(row["chat_id"]),
            topic=str(row["topic"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _insert_missing_verified_claims(
        self,
        connection: sqlite3.Connection,
        question_id: str,
        verified_claims: list[VerifiedClaim],
    ) -> int:
        inserted = 0
        for claim in verified_claims:
            existing = connection.execute(
                """
                SELECT 1 FROM evidence
                WHERE question_id = ? AND claim = ? AND verdict = ? AND COALESCE(source, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (question_id, claim.claim, claim.verdict, claim.source),
            ).fetchone()
            if existing is not None:
                continue
            connection.execute(
                """
                INSERT INTO evidence (
                    id, question_id, claim, verdict, source, checked_at, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id("ev"),
                    question_id,
                    claim.claim,
                    claim.verdict,
                    claim.source,
                    claim.checked_at,
                    claim.notes,
                    _now_iso(),
                ),
            )
            inserted += 1
        return inserted

    def _delete_stale_verified_claims(
        self,
        connection: sqlite3.Connection,
        question_id: str,
        verified_claims: list[VerifiedClaim],
    ) -> None:
        current_keys = {
            (claim.claim, claim.verdict, claim.source or "")
            for claim in verified_claims
        }
        rows = connection.execute(
            """
            SELECT id, claim, verdict, COALESCE(source, '') AS source
            FROM evidence
            WHERE question_id = ?
            """,
            (question_id,),
        ).fetchall()
        stale_ids = [
            row["id"]
            for row in rows
            if (row["claim"], row["verdict"], row["source"]) not in current_keys
        ]
        if not stale_ids:
            return
        connection.executemany(
            "DELETE FROM evidence WHERE id = ?",
            [(evidence_id,) for evidence_id in stale_ids],
        )
