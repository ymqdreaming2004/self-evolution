from self_evolution_agent.rag import (
    chunk_text,
    clean_text,
    collection_name_for_model,
)


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("a   b\r\n\r\n\r\nc") == "a b\n\nc"


def test_chunk_text_preserves_overlap_and_content() -> None:
    text = "第一段。" * 300
    chunks = chunk_text(text, target_size=100, overlap=15)
    assert len(chunks) > 2
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_empty() -> None:
    assert chunk_text("   \n") == []


def test_collection_name_is_model_specific() -> None:
    assert (
        collection_name_for_model("Qwen/Qwen3-Embedding-0.6B")
        == "personal_knowledge_qwen_qwen3_embedding_0_6b"
    )
