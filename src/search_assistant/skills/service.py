from __future__ import annotations

import re
from pathlib import Path

from search_assistant.memory.store import MemoryStore


class SkillDraftService:
    def __init__(self, store: MemoryStore, drafts_dir: str | Path = "skills/drafts"):
        self.store = store
        self.drafts_dir = Path(drafts_dir)

    def create_from_experience(self, title: str, source_ids: list[str]) -> str:
        slug = _slugify(title)
        skill_dir = self.drafts_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        body = (
            "---\n"
            f"name: {slug}\n"
            f"description: Draft skill generated from reviewed assistant experience: {title}\n"
            "---\n\n"
            f"# {title}\n\n"
            "Use this draft only after human review. It was generated from source ids:\n\n"
            + "\n".join(f"- {source_id}" for source_id in source_ids)
            + "\n\n## Instructions\n\n"
            "Answer similar future questions with explicit assumptions, verification notes, and a short final recommendation.\n"
        )
        path.write_text(body, encoding="utf-8")
        self.store.add_skill_draft(title, str(path), source_ids)
        return str(path)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "draft-skill"
