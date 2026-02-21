from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    raw: object


class EmbeddingsPort(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        ...


def get_embeddings_port() -> EmbeddingsPort:
    from app.infrastructure.openai_adapter import OpenAIAdapter

    return OpenAIAdapter()
