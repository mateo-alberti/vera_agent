from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class AnswerResult:
    text: str
    raw: object


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    raw: object


class AnswerPort(Protocol):
    def generate_answer(self, prompt: str, *, system: Optional[str] = None) -> AnswerResult:
        ...


class EmbeddingsPort(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        ...


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
