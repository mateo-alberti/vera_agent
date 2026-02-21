from __future__ import annotations

from typing import Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings
from app.domain.embeddings_port import EmbeddingResult, EmbeddingsPort
from app.domain.llm_port import AnswerResult, LLMPort


class OpenAIAdapter(LLMPort, EmbeddingsPort):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.llm = ChatOpenAI(
            model=self.settings.openai_answer_model,
            api_key=self.settings.openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
        )

    def generate_answer(self, prompt: str, *, system: Optional[str] = None) -> AnswerResult:
        system_message = system or "You are a helpful assistant."
        template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("user", "{input}")]
        )
        chain = template | self.llm | StrOutputParser()
        text = chain.invoke({"input": prompt})
        return AnswerResult(text=text, raw=text)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = self.embeddings.embed_documents(list(texts))
        return EmbeddingResult(vectors=vectors, raw=vectors)

    def get_chat_model(self) -> object:
        return self.llm
