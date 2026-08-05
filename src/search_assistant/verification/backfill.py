from __future__ import annotations

from typing import Any

from search_assistant.memory.store import MemoryStore
from search_assistant.verification.policy import extract_key_claims, verify_claims_against_sources


class VerificationBackfillService:
    def __init__(self, store: MemoryStore):
        self.store = store

    def run(self) -> dict[str, Any]:
        answers = self.store.list_answers()
        result = {
            "answers_scanned": len(answers),
            "answers_with_sources": 0,
            "answers_updated": 0,
            "verified_claims": 0,
            "unverified_claims": 0,
            "evidence_inserted": 0,
        }

        for package in answers:
            if not package.sources:
                continue
            result["answers_with_sources"] += 1
            claims = extract_key_claims(package.answer_text)
            verified, unverified = verify_claims_against_sources(claims, package.sources)
            if not verified and not unverified:
                continue
            inserted = self.store.update_answer_verification(
                package.question_id,
                verified_claims=verified,
                unverified_claims=unverified,
            )
            result["answers_updated"] += 1
            result["verified_claims"] += len(verified)
            result["unverified_claims"] += len(unverified)
            result["evidence_inserted"] += inserted

        return result
