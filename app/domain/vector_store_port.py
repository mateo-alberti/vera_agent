from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence


@dataclass(frozen=True)
class VectorEmbeddingItem:
    id: str
    embedding: List[float]
    document: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchMatch:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStorePort(Protocol):
    def upsert_embeddings(self, items: Sequence[VectorEmbeddingItem]) -> int:
        ...

    def search_by_embedding(
        self, query_embedding: Sequence[float], k: int = 5
    ) -> List[VectorSearchMatch]:
        ...


def get_vector_store_port() -> VectorStorePort:
    from app.infrastructure.chroma_adapter import ChromaAdapter

    return ChromaAdapter()
