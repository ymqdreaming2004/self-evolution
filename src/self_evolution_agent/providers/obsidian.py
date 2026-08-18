from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True)
class ObsidianNote:
    path: Path
    link: str


class ObsidianVault:
    """Persists original knowledge as Markdown and returns an Obsidian URI."""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path

    def write_knowledge(
        self,
        *,
        document_id: str,
        title: str,
        content: str,
        tags: list[str],
        source: str,
        created_at: datetime,
    ) -> ObsidianNote:
        note_path = self._note_path(document_id, title, created_at)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            self._render_note(
                document_id=document_id,
                title=title,
                content=content,
                tags=tags,
                source=source,
                created_at=created_at,
            ),
            encoding="utf-8",
        )
        return ObsidianNote(
            path=note_path,
            link=f"obsidian://open?path={quote(str(note_path.resolve()), safe='')}",
        )

    def _note_path(self, document_id: str, title: str, created_at: datetime) -> Path:
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")
        safe_title = safe_title[:80] or "untitled"
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", document_id)[:40]
        return (
            self.vault_path
            / "knowledge"
            / f"{created_at:%Y}"
            / f"{created_at:%m}"
            / f"{created_at:%Y-%m-%d}_{safe_title}_{safe_id}.md"
        )

    @staticmethod
    def _yaml_string(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'"{escaped}"'

    def _render_note(
        self,
        *,
        document_id: str,
        title: str,
        content: str,
        tags: list[str],
        source: str,
        created_at: datetime,
    ) -> str:
        tag_lines = "\n".join(f"  - {self._yaml_string(tag)}" for tag in tags) or "  - \"未分类\""
        frontmatter = "\n".join(
            [
                "---",
                f"id: {self._yaml_string(document_id)}",
                f"title: {self._yaml_string(title)}",
                "tags:",
                tag_lines,
                f"created_at: {self._yaml_string(created_at.isoformat())}",
                f"source: {self._yaml_string(source)}",
                "---",
            ]
        )
        return f"{frontmatter}\n\n# {title}\n\n{content.strip()}\n"
