from __future__ import annotations

import ast
import json
from threading import RLock
from typing import Any, Dict, Optional


_MEMORY_LOCK = RLock()
_MEMORY_BY_ID: Dict[str, list[dict[str, str]]] = {}
_MEMORY_WINDOW = 6


def get_history(conversation_id: str) -> list[dict[str, str]]:
    with _MEMORY_LOCK:
        return list(_MEMORY_BY_ID.get(conversation_id, []))


def store_turn(conversation_id: str, user_message: str, assistant_message: str) -> None:
    with _MEMORY_LOCK:
        history = _MEMORY_BY_ID.setdefault(conversation_id, [])
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        max_messages = _MEMORY_WINDOW * 2
        if len(history) > max_messages:
            del history[:-max_messages]


def message_role(message: Any) -> Optional[str]:
    if isinstance(message, dict):
        return message.get("role") or message.get("type")
    return getattr(message, "type", None)


def message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def sources_line_from_content(content: Any) -> str:
    if isinstance(content, dict):
        return content.get("sources_line", "") or ""
    if isinstance(content, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(content)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("sources_line"):
                return str(parsed["sources_line"])
    return ""


def extract_sources_line(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = message_role(message)
        if role == "tool":
            sources_line = sources_line_from_content(message_content(message))
            if sources_line:
                return sources_line
    return ""


def extract_last_ai_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = message_role(message)
        if role in {"ai", "assistant"}:
            content = message_content(message)
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def split_sources_line(text: Any) -> tuple[str, str]:
    if text is None:
        return "", ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return "", ""
    stripped = text.rstrip()
    if not stripped:
        return text, ""
    lines = stripped.splitlines()
    if not lines:
        return text, ""
    last_line = lines[-1].strip()
    if not last_line.startswith("Sources:"):
        return text, ""
    answer = "\n".join(lines[:-1]).rstrip()
    return answer, last_line
