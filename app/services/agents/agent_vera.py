from __future__ import annotations

from dataclasses import dataclass

from app.domain.ports import AnswerPort


@dataclass
class VeraAgent:
    answer_port: AnswerPort

    def respond(self, user_message: str) -> str:
        result = self.answer_port.generate_answer(user_message)
        return result.text
