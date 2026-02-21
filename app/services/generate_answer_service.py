from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from app.domain.llm_port import get_llm_port
from app.services.agents.agent_vera import VeraAgent


@dataclass
class GenerateAnswerService:
    def respond(self, user_message: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
        llm_port = get_llm_port()
        llm = llm_port.get_chat_model()
        agent = VeraAgent(llm=llm)
        return agent.respond(user_message, conversation_id=conversation_id)
