from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Tuple
from uuid import uuid4

from langchain.agents import create_agent

from app.prompts import VERA_SYSTEM_PROMPT
from app.services.agents.shared.context_memory import (
    extract_last_ai_message,
    extract_sources_line,
    get_history,
    store_turn,
)
from app.services.tools import (
    get_current_weather_tool,
    get_knowledge_base_tool,
    get_stock_price_tool,
)


@dataclass
class VeraAgent:
    llm: Any
    system_prompt: str = VERA_SYSTEM_PROMPT
    name: str = "vera"

    def respond(self, user_message: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
        logger = logging.getLogger("vera.agent")
        logger.info("agent_start name=%s input=%s", self.name, user_message)

        if conversation_id is None or not conversation_id.strip():
            conversation_id = uuid4().hex
        history = get_history(conversation_id)

        tools = [
            get_current_weather_tool(),
            get_stock_price_tool(),
            get_knowledge_base_tool(),
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

        store_turn(conversation_id, user_message, assistant_message)
        return output, conversation_id
