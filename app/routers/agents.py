from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter

from app.services.generate_answer_service import GenerateAnswerService

router = APIRouter(prefix="/agents", tags=["agents"])


class AnswerRequest(BaseModel):
    message: str


class AnswerResponse(BaseModel):
    answer: str


@router.post("/answer", response_model=AnswerResponse)
def generate_answer(payload: AnswerRequest) -> AnswerResponse:
    service = GenerateAnswerService()
    answer = service.respond(payload.message)
    return AnswerResponse(answer=answer)
