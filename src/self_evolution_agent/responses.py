from __future__ import annotations

from collections.abc import Sequence

from .schemas import KnowledgeHit


def inspiration_recorded(*, idea_type: str, title: str) -> str:
    return f"已记录{idea_type}：{title}"


def knowledge_stored(*, document_count: int, chunk_count: int) -> str:
    return f"已沉淀 {document_count} 条知识，共 {chunk_count} 个索引片段。"


def knowledge_not_found() -> str:
    return "知识库中没有找到相关内容。"


def knowledge_search_results(hits: Sequence[KnowledgeHit]) -> str:
    lines = ["知识检索结果："]
    for index, hit in enumerate(hits, start=1):
        excerpt = " ".join(hit.content.split())[:240]
        lines.append(f"{index}. {hit.title}\n   {excerpt}\n   来源：{hit.source}")
        if hit.note_link:
            lines.append(f"   笔记：{hit.note_link}")
    return "\n".join(lines)
