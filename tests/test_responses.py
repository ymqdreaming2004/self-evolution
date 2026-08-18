from datetime import UTC, datetime

from self_evolution_agent.responses import knowledge_search_results
from self_evolution_agent.schemas import KnowledgeHit


def test_knowledge_search_results_uses_fixed_format() -> None:
    result = knowledge_search_results(
        [
            KnowledgeHit(
                title="测试笔记",
                content="这是可检索的知识内容。",
                source="feishu:message-1",
                note_link="obsidian://open?path=test",
                created_at=datetime.now(UTC),
            )
        ]
    )

    assert result == (
        "知识检索结果：\n"
        "1. 测试笔记\n"
        "   这是可检索的知识内容。\n"
        "   来源：feishu:message-1\n"
        "   笔记：obsidian://open?path=test"
    )
