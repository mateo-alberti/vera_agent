from fastapi import APIRouter

from app.infrastructure.openai_adapter import OpenAIAdapter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict:
    return {"status": "healthy"}


@router.get("/openai")
def openai_health_check() -> dict:
    adapter = OpenAIAdapter()
    try:
        result = adapter.generate_answer("ping", system="health check")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "provider": "openai", "error": str(exc)}

    return {
        "status": "ok",
        "provider": "openai",
        "model": adapter.settings.openai_answer_model,
        "sample": result.text,
    }
