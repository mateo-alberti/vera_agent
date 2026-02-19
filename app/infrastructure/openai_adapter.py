from __future__ import annotations

from typing import Optional, Sequence

from openai import OpenAI

from app.core.config import Settings
from app.domain.ports import AnswerPort, AnswerResult, EmbeddingsPort, EmbeddingResult


class OpenAIAdapter(AnswerPort, EmbeddingsPort):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def generate_answer(self, prompt: str, *, system: Optional[str] = None) -> AnswerResult:
        input_items = []
        if system:
            input_items.append({"role": "system", "content": system})
        input_items.append({"role": "user", "content": prompt})

        response = self.client.responses.create(
            model=self.settings.openai_answer_model,
            input=input_items,
        )

        return AnswerResult(text=response.output_text, raw=response)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=list(texts),
            encoding_format="float",
        )

        vectors = [item.embedding for item in response.data]
        return EmbeddingResult(vectors=vectors, raw=response)
