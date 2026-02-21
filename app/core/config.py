from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

# Load .env from the project root (or CWD) before reading env vars.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_answer_model: str = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini-2025-04-14")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "vera_docs")
    openmeteo_base_url: str = os.getenv("OPENMETEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")
    alphavantage_api_key: str | None = os.getenv("ALPHAVANTAGE_API_KEY")
    alphavantage_base_url: str = os.getenv("ALPHAVANTAGE_BASE_URL", "https://www.alphavantage.co/query")
