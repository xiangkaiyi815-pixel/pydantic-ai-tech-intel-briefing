from __future__ import annotations

import json
import re
from pathlib import Path

from search_assistant.memory.store import MemoryStore


class SkillDraftService:
    def __init__(
        self,
        store: MemoryStore,
        drafts_dir: str | Path = "skills/drafts",
        active_dir: str | Path | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
    ):
        self.store = store
        self.drafts_dir = Path(drafts_dir)
        self.active_dir = Path(active_dir) if active_dir is not None else self.drafts_dir.parent / "active"
        self.user_id = user_id
        self.chat_id = chat_id

    def create_from_experience(self, title: str, source_ids: list[str]) -> str:
        slug = _slugify(title)
        skill_dir = self.drafts_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        path.write_text(self._draft_body(title, source_ids), encoding="utf-8")
        self.store.add_skill_draft(
            title,
            str(path),
            source_ids,
            user_id=self.user_id or "system",
            chat_id=self.chat_id or "system",
        )
        return str(path)

    def auto_create_from_experience(self, max_drafts: int = 3, refresh_existing: bool = True) -> list[str]:
        if refresh_existing:
            self.refresh_reviewable_drafts()
        profiles = self.store.list_profile_snapshots()
        existing_names = {draft["name"] for draft in self._drafts()}
        created: list[str] = []
        for snapshot in reversed(profiles):
            topics = self._profile_topics(snapshot)
            for topic in topics:
                title = f"{topic} Practice"
                if title in existing_names:
                    continue
                created.append(self.create_from_experience(title, source_ids=[snapshot["id"]]))
                existing_names.add(title)
                if len(created) >= max_drafts:
                    return created
        for item in reversed(self._experiences()):
            title = self._experience_skill_title(item)
            if title in existing_names:
                continue
            created.append(self.create_from_experience(title, source_ids=[item["id"]]))
            existing_names.add(title)
            if len(created) >= max_drafts:
                return created
        return created

    def refresh_reviewable_drafts(self) -> list[str]:
        refreshed: list[str] = []
        for draft in self._drafts():
            if str(draft.get("review_status")) == "promoted":
                continue
            path = Path(str(draft.get("path", "")))
            try:
                self._ensure_under(path, self.drafts_dir, "draft")
            except ValueError:
                continue
            if path.name != "SKILL.md" or not path.exists():
                continue
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not self._should_refresh_draft(current):
                continue
            source_ids = _parse_source_ids(str(draft.get("source_ids_json") or "[]"))
            path.write_text(self._draft_body(str(draft["name"]), source_ids), encoding="utf-8")
            refreshed.append(str(path))
        return refreshed

    def promote(self, name_or_slug: str) -> str:
        draft = self._find_draft(name_or_slug)
        draft_path = Path(str(draft["path"]))
        self._ensure_under(draft_path, self.drafts_dir, "draft")
        if draft_path.name != "SKILL.md" or not draft_path.exists():
            raise FileNotFoundError(f"Draft SKILL.md not found for {name_or_slug}")

        slug = _slugify(str(draft["name"]))
        active_path = self.active_dir / slug / "SKILL.md"
        self._ensure_under(active_path, self.active_dir, "active")
        if active_path.exists():
            raise FileExistsError(f"Active skill already exists: {active_path}")

        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            self._active_body_from_draft(
                draft_path.read_text(encoding="utf-8"),
                title=str(draft["name"]),
            ),
            encoding="utf-8",
        )
        self.store.update_skill_draft_status(str(draft["id"]), "promoted", path=str(active_path))
        return str(active_path)

    def list_review_status(self) -> dict[str, object]:
        skills: list[dict[str, object]] = []
        counts = {
            "total": 0,
            "draft": 0,
            "promoted": 0,
            "active_files": 0,
            "missing_files": 0,
        }
        for draft in self._drafts():
            name = str(draft["name"])
            path = Path(str(draft["path"]))
            status = str(draft["review_status"])
            file_exists = path.exists()
            active = status == "promoted" and file_exists
            source_ids = _parse_source_ids(str(draft.get("source_ids_json") or "[]"))
            item = {
                "id": str(draft["id"]),
                "name": name,
                "slug": _slugify(name),
                "path": str(path),
                "review_status": status,
                "source_ids": source_ids,
                "file_exists": file_exists,
                "active": active,
                "created_at": str(draft["created_at"]),
            }
            skills.append(item)
            counts["total"] += 1
            if status == "promoted":
                counts["promoted"] += 1
                if file_exists:
                    counts["active_files"] += 1
                else:
                    counts["missing_files"] += 1
            else:
                counts["draft"] += 1

        return {
            "counts": counts,
            "skills": skills,
        }

    def _draft_body(self, title: str, source_ids: list[str]) -> str:
        slug = _slugify(title)
        return (
            "---\n"
            f"name: {slug}\n"
            f"description: Draft skill generated from reviewed assistant experience: {title}\n"
            "---\n\n"
            f"# {title}\n\n"
            "Use this draft only after human review. It was generated from source ids:\n\n"
            + "\n".join(f"- {source_id}" for source_id in source_ids)
            + "\n\n"
            + self._evidence_summary(source_ids)
            + "\n\n"
            + self._topic_specific_guidance(title, source_ids)
        )

    def _active_body_from_draft(self, draft_body: str, title: str) -> str:
        active_body = re.sub(
            r"^description:\s*Draft skill generated from reviewed assistant experience:.*$",
            f"description: Use when applying reviewed local guidance for {title}",
            draft_body,
            count=1,
            flags=re.MULTILINE,
        )
        active_body = active_body.replace(
            "Use this draft only after human review. It was generated from source ids:",
            (
                "This skill has been reviewed and promoted as procedural guidance. "
                "Do not treat it as factual evidence.\n\n"
                "Source ids reviewed before promotion:"
            ),
        )
        if "This skill has been reviewed and promoted as procedural guidance." not in active_body:
            active_body = self._insert_active_promotion_notice(active_body)
        return active_body

    def _insert_active_promotion_notice(self, body: str) -> str:
        notice = (
            "This skill has been reviewed and promoted as procedural guidance. "
            "Do not treat it as factual evidence."
        )
        lines = body.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = index + 1
                lines[insert_at:insert_at] = ["", notice]
                return "\n".join(lines).rstrip() + "\n"
        return notice + "\n\n" + body.rstrip() + "\n"

    def _topic_specific_guidance(self, title: str, source_ids: list[str]) -> str:
        topic_text = f"{title}\n{self._evidence_summary(source_ids)}".lower()
        if any(marker in topic_text for marker in ("infrastructure", "deployment", "gb10", "deepseek", "model deployment")):
            return (
                "## Search Discipline\n\n"
                "- For model deployment and parameter questions, search ModelScope/魔搭/魔塔, Hugging Face, GitHub, and official model pages before estimating feasibility.\n"
                "- For hardware constraints, verify NVIDIA official pages first; explicitly check unified memory, memory bandwidth, interconnect, and ConnectX-7/CX7 networking when GB10/DGX Spark is involved.\n"
                "- Search for contradictory evidence and missing benchmarks before finalizing a deployment conclusion.\n\n"
                "## Answer Discipline\n\n"
                "- Separate confirmed facts, assumptions, estimates, and unknowns.\n"
                "- Show memory math, quantization assumptions, active-parameter assumptions, batch/KV-cache constraints, and network bottlenecks when relevant.\n"
                "- Do not present tok/s, cluster scaling, full-precision feasibility, or vendor benchmark numbers as confirmed unless they appear in retrieved sources.\n"
                "- If public benchmarks are missing, give a conditional estimate with formulas and mark it as an estimate.\n"
            )
        if any(marker in topic_text for marker in ("memory architecture", "cxl", "memory pooling", "accelerator memory")):
            return (
                "## Search Discipline\n\n"
                "- Prefer CXL Consortium material, vendor architecture notes, and accelerator/server documentation.\n"
                "- Verify whether the claim is about cache coherency, memory pooling, memory expansion, PCIe transport, GPU connectivity, or software support.\n\n"
                "## Answer Discipline\n\n"
                "- Distinguish CXL.io, CXL.cache, and CXL.mem only when the source supports that level of detail.\n"
                "- Explain latency, bandwidth, coherency, NUMA/software, and deployment maturity separately.\n"
                "- Avoid implying that CXL directly replaces HBM or solves GPU memory bandwidth unless sourced evidence supports it.\n"
            )
        if any(marker in topic_text for marker in ("physical ai", "world model", "embodied", "physics foundation")):
            return (
                "## Search Discipline\n\n"
                "- Prefer primary sources such as NVIDIA Cosmos/GR00T pages, Google DeepMind Genie pages, arXiv papers, project pages, and official release notes.\n"
                "- Search for dates, capability boundaries, benchmarks, and demos separately so progress claims are time-scoped.\n\n"
                "## Answer Discipline\n\n"
                "- Distinguish world models, video generation, robotics/embodied AI, simulation, and physics foundation models.\n"
                "- Start with what is demonstrated now, then list unresolved limits such as controllability, long-horizon consistency, real-world transfer, and evaluation gaps.\n"
                "- Do not merge demos, papers, and products into one maturity level without explicit source support.\n"
            )
        if "feishu" in topic_text:
            return (
                "## Search Discipline\n\n"
                "- Prefer Feishu/Open Platform official docs for event subscription, long connection, permissions, message receive events, and reply APIs.\n"
                "- Check tenant/app permission status and delivery evidence before assuming a bot code bug.\n\n"
                "## Answer Discipline\n\n"
                "- Separate event delivery, dedupe, model generation, and reply API failures.\n"
                "- Include exact diagnostic commands and avoid printing secrets or access tokens.\n"
            )
        return (
            "## Search Discipline\n\n"
            "- Search primary sources first, then compare secondary sources for disagreement.\n"
            "- Keep query planning tied to the user's concrete entities, dates, versions, and constraints.\n\n"
            "## Answer Discipline\n\n"
            "- Answer similar future questions with explicit assumptions, verification notes, uncertainty, and a short final recommendation.\n"
            "- Do not treat this skill as factual evidence; use it only as reviewed procedure guidance.\n"
        )

    def _should_refresh_draft(self, content: str) -> bool:
        if "## Search Discipline" in content and "## Answer Discipline" in content:
            return False
        return "Answer similar future questions with explicit assumptions, verification notes, and a short final recommendation." in content

    def _evidence_summary(self, source_ids: list[str]) -> str:
        topics: list[str] = []
        experience_notes: list[str] = []
        for snapshot in self.store.list_profile_snapshots():
            if snapshot["id"] not in source_ids:
                continue
            try:
                summary = json.loads(snapshot["summary_json"])
            except json.JSONDecodeError:
                continue
            for topic in summary.get("recurring_topics", []):
                if isinstance(topic, str) and topic not in topics:
                    topics.append(topic)
        for item in self._experiences():
            if item["id"] not in source_ids:
                continue
            note = f"{item['title']}: {item['body']}"
            if note not in experience_notes:
                experience_notes.append(note)

        if not topics and not experience_notes:
            return "## Evidence Summary\n\n- No profile evidence was found for these source ids."
        lines = ["## Evidence Summary", ""]
        lines.extend(f"- {topic}" for topic in topics)
        lines.extend(f"- {note}" for note in experience_notes)
        return "\n".join(lines)

    def _profile_topics(self, snapshot: dict[str, object]) -> list[str]:
        try:
            summary = json.loads(str(snapshot["summary_json"]))
        except (KeyError, json.JSONDecodeError):
            return []
        topics: list[str] = []
        for topic in summary.get("recurring_topics", []):
            if isinstance(topic, str) and topic and topic not in topics:
                topics.append(topic)
        return topics

    def _experience_skill_title(self, item: dict[str, object]) -> str:
        title = str(item["title"])
        if title != "Evaluation quality issue":
            return title
        flags = _evaluation_quality_flags(str(item.get("body", "")))
        if flags & {"review_rejected", "blocked_answer"}:
            return "Evaluation Review Rejection Recovery Practice"
        if flags & {"missing_search_record", "no_sources"}:
            return "Evaluation Search Coverage Recovery Practice"
        if "unverified_claims" in flags:
            return "Evaluation Verification Recovery Practice"
        if "missing_calibration" in flags:
            return "Evaluation Calibration Recovery Practice"
        if "low_confidence" in flags:
            return "Evaluation Low Confidence Recovery Practice"
        return "Evaluation Quality Recovery Practice"

    def _find_draft(self, name_or_slug: str) -> dict[str, object]:
        slug = _slugify(name_or_slug)
        matches = []
        for draft in self._drafts():
            draft_name = str(draft["name"])
            draft_slug = _slugify(draft_name)
            draft_path_slug = Path(str(draft["path"])).parent.name
            if name_or_slug in {str(draft["id"]), draft_name, str(draft["path"])}:
                matches.append(draft)
            elif slug in {draft_slug, draft_path_slug}:
                matches.append(draft)

        if not matches:
            raise KeyError(f"No skill draft found for {name_or_slug}")
        if len(matches) > 1:
            raise ValueError(f"Multiple skill drafts match {name_or_slug}; use the draft id")
        return matches[0]

    def _ensure_under(self, path: Path, base: Path, label: str) -> None:
        resolved_path = path.resolve()
        resolved_base = base.resolve()
        if not resolved_path.is_relative_to(resolved_base):
            raise ValueError(f"{label} path escapes configured directory: {path}")

    def _drafts(self) -> list[dict[str, object]]:
        if self.user_id is None or self.chat_id is None:
            return self.store.list_skill_drafts()
        return self.store.list_skill_drafts(self.user_id, self.chat_id)

    def _experiences(self) -> list[dict[str, object]]:
        if self.user_id is None or self.chat_id is None:
            return self.store.list_experience_items()
        return self.store.list_experience_items(self.user_id, self.chat_id)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "draft-skill"


def _parse_source_ids(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _evaluation_quality_flags(body: str) -> set[str]:
    match = re.search(r"^quality_flags=(.*)$", body, flags=re.MULTILINE)
    if not match:
        return set()
    return {flag.strip() for flag in match.group(1).split(",") if flag.strip()}
