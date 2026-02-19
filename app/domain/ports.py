from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


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
