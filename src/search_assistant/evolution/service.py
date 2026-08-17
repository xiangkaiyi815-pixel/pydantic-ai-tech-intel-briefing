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
from search_assistant.knowledge_graph.embedding import EmbeddingProvider, build_embedding_provider
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore


KNOWLEDGE_LAYER_VALIDATED = "validated_knowledge"
KNOWLEDGE_LAYER_WEAK_SIGNAL = "weak_signal"
KNOWLEDGE_LAYER_REJECTED = "rejected_noise"
KNOWLEDGE_LAYER_UNREVIEWED = "unreviewed_candidate"


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

    def __init__(
        self,
        store: MemoryStore,
        graph_service: DomainKnowledgeGraphService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.store = store
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.graph_service = graph_service or DomainKnowledgeGraphService(
            store, embedding_provider=self.embedding_provider
        )

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
            candidate_ids.append(self.store.upsert_domain_knowledge_candidate(candidate)[0])
        unique_ids = list(dict.fromkeys(candidate_ids))
        self.link_candidates_to_knowledge_graphs(unique_ids)
        return unique_ids

    def link_candidates_to_knowledge_graphs(
        self,
        candidate_ids: list[str] | None = None,
        limit_per_candidate: int = 1,
    ) -> dict[str, Any]:
        """Attach reviewable candidates to reviewed graph entities without promoting them."""

        seed_result = self._ensure_default_graphs()
        requested_ids = {candidate_id for candidate_id in candidate_ids or []}
        candidates = [
            candidate
            for candidate in self.store.list_domain_knowledge_candidates()
            if not requested_ids or candidate["id"] in requested_ids
        ]
        existing_keys = {
            (
                str(link["candidate_id"]),
                str(link["graph_id"]),
                str(link["entity_id"] or ""),
                str(link["relation_id"] or ""),
                str(link["link_type"]),
            )
            for link in self.store.list_domain_candidate_graph_links()
        }

        created_links: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["status"] == "deprecated":
                continue
            query = self._candidate_graph_query(candidate)
            hits = self.graph_service.query_relevant(query, limit=limit_per_candidate, min_score=5.0)
            for hit in hits:
                key = (
                    str(candidate["id"]),
                    hit.graph_id,
                    hit.entity_id,
                    "",
                    "candidate_matches_entity",
                )
                if key in existing_keys:
                    continue
                link_id = self.store.link_domain_candidate_to_graph(
                    candidate_id=str(candidate["id"]),
                    graph_id=hit.graph_id,
                    entity_id=hit.entity_id,
                    link_type="candidate_matches_entity",
                    note=(
                        "Matched by self-evolution candidate text; "
                        f"score={hit.score}; entity={hit.entity_name}"
                    ),
                )
                existing_keys.add(key)
                created_links.append(
                    {
                        "id": link_id,
                        "candidate_id": str(candidate["id"]),
                        "graph_id": hit.graph_id,
                        "entity_id": hit.entity_id,
                        "score": hit.score,
                    }
                )

        return {
            "seeded_graphs": seed_result["seeded_graphs"],
            "candidates_considered": len(candidates),
            "created_links": len(created_links),
            "links": created_links,
        }

    def record_validation_gate(self, candidate_id: str) -> dict[str, Any]:
        """Record the automatic traceability gate without promoting the candidate."""
        return self._run_validation_gate(candidate_id, promote=False)

    def validate(self, candidate_id: str) -> dict[str, Any]:
        return self._run_validation_gate(candidate_id, promote=True)

    def list_candidates(self, status: str | None = None, layer: str | None = None) -> list[dict[str, Any]]:
        candidates = self.store.list_domain_knowledge_candidates(status)
        latest_layers = self._latest_validation_layers()
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            validation = latest_layers.get(str(candidate["id"]), {})
            item = {
                **candidate,
                "knowledge_layer": validation.get("knowledge_layer", KNOWLEDGE_LAYER_UNREVIEWED),
                "knowledge_layer_reason": validation.get("reason", "validation gate has not run"),
                "knowledge_layer_next_action": validation.get(
                    "next_action",
                    "run knowledge-candidate-validate or generate a new briefing",
                ),
            }
            if layer is None or item["knowledge_layer"] == layer:
                enriched.append(item)
        return enriched

    def _run_validation_gate(self, candidate_id: str, promote: bool) -> dict[str, Any]:
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        evidence_urls = {
            str(item.get("url", ""))
            for item in candidate["evidence"]
            if str(item.get("url", "")).startswith(("http://", "https://"))
        }
        failures: list[str] = []
        if not evidence_urls:
            failures.append("no_original_source")
        elif len(evidence_urls) < 2:
            failures.append("fewer_than_two_original_sources")
        if candidate["contradictions"]:
            failures.append("unresolved_contradictions")
        independent_trajectories = self.independent_trajectory_count(candidate)
        if independent_trajectories < 2:
            failures.append("fewer_than_two_independent_trajectories")
        if candidate["confidence"] == "low":
            failures.append("low_confidence")
        knowledge_layer = self._knowledge_layer(failures)
        layer_metadata = self._layer_metadata(knowledge_layer)
        if failures:
            self.store.add_gate_record(
                gate_type="domain_knowledge_candidate_validation",
                subject_type="domain_knowledge_candidate",
                subject_id=candidate_id,
                result="failed",
                reason=", ".join(failures),
                evidence_refs=self._candidate_evidence_refs(candidate),
                metadata={
                    "topic": candidate["topic"],
                    "confidence": candidate["confidence"],
                    "evidence_url_count": len(evidence_urls),
                    "independent_trajectory_count": independent_trajectories,
                    "candidate_status": candidate["status"],
                    "knowledge_layer": knowledge_layer,
                    **layer_metadata,
                },
            )
            return {
                "validated": False,
                "candidate_id": candidate_id,
                "failures": failures,
                "independent_trajectory_count": independent_trajectories,
                "knowledge_layer": knowledge_layer,
                **layer_metadata,
            }

        if promote:
            self.store.update_domain_knowledge_candidate_status(
                candidate_id,
                "validated",
                "passed traceability gate: multiple original sources across at least two independent trajectories, "
                "no unresolved contradictions, non-low confidence",
            )
        self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_validation",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason="multiple original sources across at least two independent trajectories, no unresolved "
            "contradictions, non-low confidence",
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={
                "topic": candidate["topic"],
                "confidence": candidate["confidence"],
                "evidence_url_count": len(evidence_urls),
                "independent_trajectory_count": independent_trajectories,
                "candidate_status": candidate["status"],
                "knowledge_layer": knowledge_layer,
                **layer_metadata,
            },
        )
        return {
            "validated": True,
            "candidate_id": candidate_id,
            "failures": [],
            "independent_trajectory_count": independent_trajectories,
            "knowledge_layer": knowledge_layer,
            **layer_metadata,
        }

    def distill_candidates(self, limit: int | None = None) -> dict[str, Any]:
        """Re-run validation gates for non-deprecated candidates.

        Called by the offline evolution loop after trajectories are verified:
        a candidate whose claim reappeared in an independent briefing now has
        merged evidence and can move from ``weak_signal`` toward
        ``validated_knowledge``.  Promotion to status ``validated`` still
        requires the explicit :meth:`validate` (or human approval) path; this
        method only re-records the automatic gate.
        """
        candidates = [
            candidate
            for candidate in self.store.list_domain_knowledge_candidates()
            if str(candidate["status"]) != "deprecated"
        ]
        if limit is not None:
            candidates = candidates[:limit]
        results = [self.record_validation_gate(str(candidate["id"])) for candidate in candidates]
        layer_counts: dict[str, int] = {}
        for result in results:
            layer = str(result["knowledge_layer"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return {
            "considered": len(candidates),
            "layer_counts": layer_counts,
            "validated": layer_counts.get(KNOWLEDGE_LAYER_VALIDATED, 0),
            "weak_signal": layer_counts.get(KNOWLEDGE_LAYER_WEAK_SIGNAL, 0),
            "rejected_noise": layer_counts.get(KNOWLEDGE_LAYER_REJECTED, 0),
            "details": results,
        }

    @staticmethod
    def independent_trajectory_count(candidate: dict[str, Any]) -> int:
        """Count distinct briefing runs (trajectories) that support a candidate.

        ``capture_briefing`` records the briefing id (``brief_...``) as the
        first source id; merged candidates accumulate every briefing id that
        produced the same claim, so this is the cross-trajectory support count
        used by the validation gate.
        """
        source_ids = candidate.get("source_ids") or []
        return len(
            {
                str(source_id)
                for source_id in source_ids
                if re.match(r"^brief", str(source_id), flags=re.IGNORECASE)
            }
        )

    def approve(
        self,
        candidate_id: str,
        reviewer: str = "local-reviewer",
        reason: str = "approved after human review",
        eval_gate: dict[str, Any] | None = None,
        require_eval_gate: bool = True,
    ) -> dict[str, Any]:
        eval_release_gate: dict[str, Any] | None = None
        if require_eval_gate:
            eval_release_gate = self._record_eval_release_gate(candidate_id, eval_gate)
            if not eval_release_gate["passed"]:
                return {
                    "approved": False,
                    "candidate_id": candidate_id,
                    "gate_id": eval_release_gate["gate_id"],
                    "validation": None,
                    "eval_gate": eval_release_gate,
                }
        elif eval_gate is not None:
            eval_release_gate = self._record_eval_release_gate(candidate_id, eval_gate)

        validation = self.validate(candidate_id)
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        if not validation["validated"]:
            gate_id = self.store.add_gate_record(
                gate_type="domain_knowledge_candidate_human_review",
                subject_type="domain_knowledge_candidate",
                subject_id=candidate_id,
                result="failed",
                reason="human approval blocked because validation gate failed: "
                + ", ".join(validation["failures"]),
                evidence_refs=self._candidate_evidence_refs(candidate),
                metadata={"reviewer": reviewer, "topic": candidate["topic"]},
            )
            return {
                "approved": False,
                "candidate_id": candidate_id,
                "gate_id": gate_id,
                "validation": validation,
                "eval_gate": eval_release_gate or eval_gate,
            }

        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_human_review",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason=reason.strip() or "approved after human review",
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={"reviewer": reviewer, "topic": candidate["topic"]},
        )
        self.store.add_project_ledger_entry(
            entry_type="knowledge_release",
            subject=candidate_id,
            status="approved",
            summary=f"Domain knowledge candidate approved for reviewed use: {candidate['topic']}",
            evidence_refs=[
                gate_id,
                *([str(eval_release_gate["gate_id"])] if eval_release_gate else []),
                *self._candidate_evidence_refs(candidate),
            ],
            risk="Candidate is planning context only; source URLs must be reopened before current factual claims.",
            rollback="Run knowledge-candidate-deprecate with a reason to remove the candidate from active use.",
            metadata={"reviewer": reviewer, "claim": candidate["claim"], "eval_gate": eval_release_gate or eval_gate or {}},
        )
        return {
            "approved": True,
            "candidate_id": candidate_id,
            "gate_id": gate_id,
            "validation": validation,
            "eval_gate": eval_release_gate or eval_gate,
        }

    def deprecate(self, candidate_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("deprecation reason must not be empty")
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        self.store.update_domain_knowledge_candidate_status(candidate_id, "deprecated", reason.strip())
        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_rollback",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason=reason.strip(),
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={"topic": candidate["topic"]},
        )
        self.store.add_project_ledger_entry(
            entry_type="knowledge_rollback",
            subject=candidate_id,
            status="deprecated",
            summary=f"Domain knowledge candidate deprecated: {candidate['topic']}",
            evidence_refs=[gate_id, *self._candidate_evidence_refs(candidate)],
            risk="Deprecated candidates remain in the audit trail but should not guide new synthesis.",
            rollback="Create a fresh candidate from newer evidence and approve it through validation and human review.",
            metadata={"reason": reason.strip(), "claim": candidate["claim"]},
        )

    def _record_eval_release_gate(self, candidate_id: str, eval_gate: dict[str, Any] | None) -> dict[str, Any]:
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        gate = eval_gate or {
            "passed": False,
            "reason": "evaluation report is required before knowledge release",
            "report_path": "",
        }
        passed = bool(gate.get("passed"))
        waived = bool(gate.get("waived"))
        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_eval_release",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="waived" if waived else "passed" if passed else "failed",
            reason=str(
                gate.get("reason")
                or ("evaluation gate waived" if waived else "evaluation gate passed" if passed else "evaluation gate failed")
            ),
            evidence_refs=[
                ref
                for ref in (
                    str(gate.get("report_path") or ""),
                    *self._candidate_evidence_refs(candidate),
                )
                if ref
            ],
            metadata={"topic": candidate["topic"], "eval_gate": gate},
        )
        return {
            "passed": passed,
            "waived": waived,
            "gate_id": gate_id,
            "reason": str(gate.get("reason") or ""),
            "report_path": str(gate.get("report_path") or ""),
        }

    def _ensure_default_graphs(self) -> dict[str, object]:
        if self.graph_service.list_graphs():
            return {"seeded_graphs": 0, "graphs": self.graph_service.list_graphs()}
        return self.graph_service.seed_default_graphs()

    @staticmethod
    def _knowledge_layer(failures: list[str]) -> str:
        if not failures:
            return KNOWLEDGE_LAYER_VALIDATED
        if "no_original_source" in failures or "unresolved_contradictions" in failures:
            return KNOWLEDGE_LAYER_REJECTED
        return KNOWLEDGE_LAYER_WEAK_SIGNAL

    @staticmethod
    def _layer_metadata(layer: str) -> dict[str, str]:
        if layer == KNOWLEDGE_LAYER_VALIDATED:
            return {
                "intended_use": "stable planning context after eval and human review gates pass",
                "next_action": "eligible for knowledge-candidate-approve after eval gate passes",
            }
        if layer == KNOWLEDGE_LAYER_WEAK_SIGNAL:
            return {
                "intended_use": "exploratory clue for the next search plan; do not use as trusted knowledge",
                "next_action": "search for independent corroborating sources before promotion",
            }
        return {
            "intended_use": "audit trail only; do not use for planning or synthesis",
            "next_action": "ignore unless a human reviewer supplies corrected evidence",
        }

    def _latest_validation_layers(self) -> dict[str, dict[str, str]]:
        latest: dict[str, dict[str, str]] = {}
        for gate in self.store.list_gate_records(gate_type="domain_knowledge_candidate_validation"):
            candidate_id = str(gate["subject_id"])
            if candidate_id in latest:
                continue
            metadata = gate.get("metadata") or {}
            latest[candidate_id] = {
                "knowledge_layer": str(metadata.get("knowledge_layer") or KNOWLEDGE_LAYER_UNREVIEWED),
                "reason": str(gate.get("reason") or ""),
                "next_action": str(metadata.get("next_action") or ""),
            }
        return latest

    @staticmethod
    def _candidate_graph_query(candidate: dict[str, Any]) -> str:
        evidence_titles = " ".join(str(item.get("title", "")) for item in candidate.get("evidence", []))
        return " ".join(
            part.strip()
            for part in (
                str(candidate.get("topic", "")),
                str(candidate.get("claim", "")),
                str(candidate.get("applies_when", "")),
                evidence_titles,
            )
            if part and part.strip()
        )

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

    @staticmethod
    def _candidate_evidence_refs(candidate: dict[str, Any]) -> list[str]:
        refs = [f"candidate:{candidate['id']}"]
        refs.extend(str(source_id) for source_id in candidate.get("source_ids", []))
        refs.extend(
            str(item.get("url"))
            for item in candidate.get("evidence", [])
            if str(item.get("url", "")).startswith(("http://", "https://"))
        )
        return list(dict.fromkeys(ref for ref in refs if ref))
