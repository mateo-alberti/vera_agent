from fastapi import FastAPI

from app.routers.health import router as health_router
from app.routers.agents import router as agents_router

app = FastAPI(
    title="Vera Agent",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(health_router)
app.include_router(agents_router)


@app.get("/")
def read_root() -> dict:
    return {"status": "ok", "message": "Vera Agent API"}
