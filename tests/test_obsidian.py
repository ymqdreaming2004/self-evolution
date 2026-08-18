from datetime import UTC, datetime
from urllib.parse import quote

from self_evolution_agent.providers.obsidian import ObsidianVault


def test_vault_writes_markdown_with_an_obsidian_link(tmp_path) -> None:
    vault = ObsidianVault(tmp_path / "vault")
    note = vault.write_knowledge(
        document_id="doc-1",
        title="测试 / 知识",
        content="正文内容",
        tags=["RAG", "测试"],
        source="feishu:message-1",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )

    text = note.path.read_text(encoding="utf-8")
    assert note.path.exists()
    assert note.path.parent == vault.vault_path / "knowledge" / "2026" / "08"
    assert "测试 _ 知识" in note.path.name
    assert 'title: "测试 / 知识"' in text
    assert 'source: "feishu:message-1"' in text
    assert note.link == f"obsidian://open?path={quote(str(note.path.resolve()), safe='')}"
