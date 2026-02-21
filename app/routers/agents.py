from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter

from app.services.generate_answer_service import GenerateAnswerService

router = APIRouter(prefix="/agents", tags=["agents"])


class AnswerRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    conversation_id: str


@router.post("/answer", response_model=AnswerResponse)
def generate_answer(payload: AnswerRequest) -> AnswerResponse:
    service = GenerateAnswerService()
    answer, conversation_id = service.respond(
        payload.message, conversation_id=payload.conversation_id
    )
    return AnswerResponse(answer=answer, conversation_id=conversation_id)
