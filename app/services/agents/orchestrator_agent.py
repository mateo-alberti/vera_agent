from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Tuple
from uuid import uuid4

from langchain.agents import create_agent
from app.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from app.services.agents.market_weather_agent import get_market_weather_agent_tool
from app.services.agents.vera_agent import get_vera_agent_tool
from app.services.agents.shared.context_memory import (
    extract_last_ai_message,
    extract_sources_line,
    get_history,
    store_turn,
)


def _scoped_conversation_id(base_id: str, scope: str) -> str:
    return f"{scope}:{base_id}"


def _format_context(history: list[dict[str, str]]) -> Optional[str]:
    if not history:
        return None
    return "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in history
    )


@dataclass
class OrchestratorAgent:
    llm: Any
    system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT
    name: str = "router"

    def respond(self, user_message: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
        logger = logging.getLogger("vera.orchestrator")
        logger.info("agent_start name=%s input=%s", self.name, user_message)

        if conversation_id is None or not conversation_id.strip():
            conversation_id = uuid4().hex

        router_conversation_id = _scoped_conversation_id(conversation_id, self.name)
        history = get_history(router_conversation_id)

        context_text = _format_context(history)
        tools = [
            get_market_weather_agent_tool(
                self.llm, default_context=context_text
            ),
            get_vera_agent_tool(self.llm, default_context=context_text),
        ]

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt,
            name=self.name,
        )
        messages = history + [{"role": "user", "content": user_message}]
        result = agent.invoke({"messages": messages})
        result_messages = result.get("messages", []) if isinstance(result, dict) else []
        assistant_message = extract_last_ai_message(result_messages) or ""
        output = assistant_message

        sources_line = extract_sources_line(result_messages)
        if sources_line:
            output = f"{output}\n\n{sources_line}"

        store_turn(router_conversation_id, user_message, assistant_message)
        return output, conversation_id
