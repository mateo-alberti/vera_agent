from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.services.agents.agent_vera import VeraAgent


@dataclass
class GenerateAnswerService:
    def respond(self, user_message: str) -> str:
        settings = Settings()
        llm = ChatOpenAI(
            model=settings.openai_answer_model,
            api_key=settings.openai_api_key,
        )
        agent = VeraAgent(llm=llm)
        return agent.respond(user_message)
