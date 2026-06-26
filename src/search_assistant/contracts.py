from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Classification = Literal["simple", "research", "hard", "high_stakes"]
Confidence = Literal["low", "medium", "high"]


class IncomingMessage(BaseModel):
    message_id: str
    event_id: str | None = None
    user_id: str
    chat_id: str
    text: str
    source: str = "cli"
    raw_event: dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return self.event_id or self.message_id


class VerifiedClaim(BaseModel):
    claim: str
    verdict: Literal["verified", "unverified", "contradicted"]
    source: str | None = None
    checked_at: str
    notes: str | None = None


class CalibrationResult(BaseModel):
    ran: bool
    critique: str
    revision: str


class MemoryUpdate(BaseModel):
    kind: str
    content: str
    source_id: str | None = None


class AnswerPackage(BaseModel):
    question_id: str
    answer_text: str
    classification: Classification
    confidence: Confidence
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    calibration: CalibrationResult | dict[str, Any] | None = None
    memory_updates: list[MemoryUpdate | dict[str, Any]] = Field(default_factory=list)
