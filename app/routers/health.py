from fastapi import APIRouter

from app.domain.llm_port import get_llm_port

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict:
    return {"status": "healthy"}


@router.get("/llm")
def llm_health_check() -> dict:
    llm_port = get_llm_port()
    try:
        result = llm_port.generate_answer("ping", system="health check")
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    chat_model = llm_port.get_chat_model()
    model = getattr(chat_model, "model", None) or getattr(chat_model, "model_name", None)

    return {
        "status": "ok",
        "model": model,
        "sample": result.text,
    }
