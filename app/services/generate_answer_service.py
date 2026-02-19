from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.openai_adapter import OpenAIAdapter
from app.services.agents.agent_vera import VeraAgent


@dataclass
class GenerateAnswerService:
    def respond(self, user_message: str) -> str:
        agent = VeraAgent(answer_port=OpenAIAdapter())
        return agent.respond(user_message)
