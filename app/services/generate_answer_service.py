from __future__ import annotations

from dataclasses import dataclass

from app.domain.llm_port import get_llm_port
from app.services.agents.agent_vera import VeraAgent


@dataclass
class GenerateAnswerService:
    def respond(self, user_message: str) -> str:
        llm_port = get_llm_port()
        llm = llm_port.get_chat_model()
        agent = VeraAgent(llm=llm)
        return agent.respond(user_message)
