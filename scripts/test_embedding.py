from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from self_evolution_agent.config import get_settings

DEFAULT_QUERY = "冰箱里的牛奶快过期了，能做什么菜？"
DEFAULT_CANDIDATES = [
    "用临期牛奶制作奶香炖蛋和白酱意面。",
    "LangGraph 可以使用状态图组织多个 Agent。",
    "记录一个周末去爬山的灵感。",
    "牛奶开封后应冷藏，并尽快饮用。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the configured embedding model")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query text")
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Candidate text; repeat this option to provide multiple values",
    )
    args = parser.parse_args()

    settings = get_settings()
    candidates = args.candidates or DEFAULT_CANDIDATES
    texts = [args.query, *candidates]

    print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    scores = embeddings[1:] @ embeddings[0]
    ranking = sorted(zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True)

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Query vector norm: {(embeddings[0] @ embeddings[0]) ** 0.5:.6f}")
    print(f"Query: {args.query}\n")
    print("Cosine similarity ranking:")
    for index, (candidate, score) in enumerate(ranking, start=1):
        print(f"{index}. {score:.6f}  {candidate}")


if __name__ == "__main__":
    main()
