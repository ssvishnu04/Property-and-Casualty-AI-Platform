from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DOC_PATH = Path("data/rag_documents")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once at import time
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_documents() -> list[dict[str, Any]]:
    docs = []

    print(f"Looking for documents in: {DOC_PATH.resolve()}")

    if not DOC_PATH.exists():
        print(f"Document path does not exist: {DOC_PATH}")
        return docs

    for file in DOC_PATH.glob("**/*.txt"):
        with file.open("r", encoding="utf-8") as f:
            content = f.read()

        docs.append(
            {
                "file_path": str(file),
                "content": content,
                "embedding": model.encode(content),
            }
        )

    print(f"Total documents loaded: {len(docs)}")
    return docs


def cosine_similarity(a, b) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def keyword_bonus(query: str, content: str) -> float:
    query_words = set(query.lower().split())
    content_lower = content.lower()

    matches = sum(1 for word in query_words if word in content_lower)

    return matches * 0.05


def simple_retriever(
    query: str,
    docs: list[dict[str, Any]],
    top_k: int = 2,
    min_score: float = 0.30,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    query_embedding = model.encode(query)

    results = []

    for doc in docs:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        score = similarity + keyword_bonus(query, doc["content"])

        if score >= min_score:
            results.append(
                {
                    "file_path": doc["file_path"],
                    "content": doc["content"],
                    "score": float(score),
                    "similarity_score": float(similarity),
                }
            )

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]