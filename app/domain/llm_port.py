from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class AnswerResult:
    text: str
    raw: object


class LLMPort(Protocol):
    def generate_answer(self, prompt: str, *, system: Optional[str] = None) -> AnswerResult:
        ...

    def get_chat_model(self) -> Any:
        ...


def get_llm_port() -> LLMPort:
    from app.infrastructure.openai_adapter import OpenAIAdapter

    return OpenAIAdapter()
