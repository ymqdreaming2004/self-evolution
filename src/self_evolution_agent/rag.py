from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

from .config import Settings
from .schemas import KnowledgeChunk, KnowledgeHit


def collection_name_for_model(model_name: str) -> str:
    """Keep embeddings from different models in separate Chroma collections."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"personal_knowledge_{normalized}"[:512]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, target_size: int = 650, overlap: int = 80) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    separators = "。！？；\n"
    while start < len(cleaned):
        ideal_end = min(len(cleaned), start + target_size)
        end = ideal_end
        if ideal_end < len(cleaned):
            candidates = [
                cleaned.rfind(char, start + target_size // 2, ideal_end) for char in separators
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


class KnowledgeStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._client = chromadb.PersistentClient(path=str(Path(settings.chroma_path)))
        self._collection = self._client.get_or_create_collection(
            collection_name_for_model(settings.embedding_model), metadata={"hnsw:space": "cosine"}
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model, device="cpu")
        return self._model

    def add_document(
        self,
        *,
        content: str,
        title: str,
        tags: list[str],
        source: str,
        note_link: str = "",
        created_at: datetime,
        document_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        doc_id = document_id or str(uuid4())
        chunks = [
            KnowledgeChunk(
                document_id=doc_id,
                chunk_id=f"{doc_id}:{index}",
                content=value,
                title=title,
                tags=tags,
                source=source,
                created_at=created_at,
            )
            for index, value in enumerate(chunk_text(content))
        ]
        if not chunks:
            raise ValueError("knowledge document has no content")
        embeddings = self.model.encode(
            [item.content for item in chunks], normalize_embeddings=True
        ).tolist()
        self._collection.add(
            ids=[item.chunk_id for item in chunks],
            documents=[item.content for item in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "tags": ",".join(item.tags),
                    "source": item.source,
                    "note_link": note_link,
                    "created_at": item.created_at.isoformat(),
                    "created_ts": item.created_at.timestamp(),
                }
                for item in chunks
            ],
        )
        return chunks

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> list[KnowledgeHit]:
        where_parts: list[dict[str, object]] = []
        if start_at:
            where_parts.append({"created_ts": {"$gte": start_at.timestamp()}})
        if end_at:
            where_parts.append({"created_ts": {"$lte": end_at.timestamp()}})
        where = None
        if len(where_parts) == 1:
            where = where_parts[0]
        elif where_parts:
            where = {"$and": where_parts}
        embedding = self.model.encode(
            [query], prompt_name="query", normalize_embeddings=True
        ).tolist()
        result = self._collection.query(
            query_embeddings=embedding,
            n_results=top_k or self.settings.knowledge_top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            KnowledgeHit(
                content=document,
                title=metadata.get("title", "未命名"),
                source=metadata.get("source", "未知来源"),
                note_link=metadata.get("note_link", ""),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                score=1 - distance if distance is not None else None,
            )
            for document, metadata, distance in zip(documents, metadatas, distances, strict=False)
        ]

    def healthy(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False
