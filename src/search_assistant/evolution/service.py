from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from search_assistant.contracts import (
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEvidence,
)
from search_assistant.memory.store import MemoryStore


class EvolutionDiagnosisService:
    """Route structured evaluation failures to a reviewable update carrier."""

    def diagnose(self, item: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
        result_flags = list(item["result_verification"]["flags"])
        process_flags = list(item["process_verification"]["flags"])
        quality_flags = list(item["quality_verification"]["flags"])
        all_flags = list(dict.fromkeys(result_flags + process_flags + quality_flags))

        if "blocked_answer" in all_flags or "task_blocked" in all_flags:
            carrier = "program_harness"
            target = "failure recovery and safe fallback routing"
        elif any(flag in all_flags for flag in ("required_search_not_executed", "search_returned_no_sources")):
            carrier = "prompt_skill"
            target = "search planning and source acquisition"
        elif any(flag in all_flags for flag in ("unverified_claims", "review_rejected", "low_confidence")):
            carrier = "prompt_skill"
            target = "claim verification and final review"
        elif "missing_calibration" in all_flags:
            carrier = "prompt_skill"
            target = "calibration policy"
        else:
            carrier = "none"
            target = "no candidate update"

        evidence_refs = [f"trajectory:{trajectory_id}", f"question:{item['question_id']}"]
        evidence_refs.extend(f"source:{url}" for url in item.get("source_urls", []))
        return {
            "root_causes": all_flags,
            "recommended_update_carrier": carrier,
            "target": target,
            "evidence_refs": evidence_refs,
            "confidence": "high" if len(all_flags) >= 2 else "medium" if all_flags else "high",
            "candidate_only": True,
        }


class DomainKnowledgeCandidateService:
    """Turn briefing evidence into deduplicated, reviewable knowledge candidates."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def capture_briefing(self, briefing: DailyBriefing) -> list[str]:
        if not briefing.sources:
            return []

        source_by_url = {source.url: source for source in briefing.sources}
        candidate_ids: list[str] = []
        for theme in briefing.synthesis.themes:
            claim = (theme.analysis or theme.what_is_happening or briefing.synthesis.analysis_judgment).strip()
            if not claim:
                continue
            evidence_urls = [url for url in theme.source_urls if url in source_by_url]
            if not evidence_urls:
                evidence_urls = [source.url for source in briefing.sources[:3]]
            evidence = [
                DomainKnowledgeEvidence(
                    title=source_by_url[url].title,
                    url=url,
                    provider=source_by_url[url].provider,
                    retrieved_at=source_by_url[url].retrieved_at,
                )
                for url in dict.fromkeys(evidence_urls)
            ]
            now = datetime.now(UTC).isoformat()
            fingerprint = self._fingerprint(briefing.topic, claim)
            candidate = DomainKnowledgeCandidate(
                id=f"knowledge_{uuid.uuid4().hex}",
                topic=briefing.topic,
                claim=claim,
                applies_when=(
                    f"Use as planning context when researching {briefing.topic}; "
                    "re-open original sources before making current factual claims."
                ),
                evidence=evidence,
                contradictions=[],
                confidence=self._confidence(len(evidence)),
                status="candidate",
                source_ids=[briefing.id, *[source_by_url[url].id for url in dict.fromkeys(evidence_urls)]],
                fingerprint=fingerprint,
                created_at=now,
                updated_at=now,
            )
            candidate_ids.append(self.store.add_domain_knowledge_candidate(candidate))
        return list(dict.fromkeys(candidate_ids))

    def validate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        evidence_urls = {
            str(item.get("url", ""))
            for item in candidate["evidence"]
            if str(item.get("url", "")).startswith(("http://", "https://"))
        }
        failures: list[str] = []
        if len(evidence_urls) < 2:
            failures.append("fewer_than_two_original_sources")
        if candidate["contradictions"]:
            failures.append("unresolved_contradictions")
        if candidate["confidence"] == "low":
            failures.append("low_confidence")
        if failures:
            return {"validated": False, "candidate_id": candidate_id, "failures": failures}

        self.store.update_domain_knowledge_candidate_status(
            candidate_id,
            "validated",
            "passed traceability gate: multiple original sources, no unresolved contradictions, non-low confidence",
        )
        return {"validated": True, "candidate_id": candidate_id, "failures": []}

    def deprecate(self, candidate_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("deprecation reason must not be empty")
        self.store.update_domain_knowledge_candidate_status(candidate_id, "deprecated", reason.strip())

    @staticmethod
    def _fingerprint(topic: str, claim: str) -> str:
        normalized = re.sub(r"\s+", " ", f"{topic.strip().lower()}\n{claim.strip().lower()}")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _confidence(evidence_count: int) -> str:
        if evidence_count >= 3:
            return "high"
        if evidence_count >= 2:
            return "medium"
        return "low"
