from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Tuple
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from app.prompts import MARKET_WEATHER_SYSTEM_PROMPT
from app.services.agents.shared.context_memory import (
    extract_last_ai_message,
    extract_sources_line,
)
from app.services.tools import get_current_weather_tool, get_stock_price_tool


@dataclass
class MarketWeatherAgent:
    llm: Any
    system_prompt: str = MARKET_WEATHER_SYSTEM_PROMPT
    name: str = "market_weather"

    def respond(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        *,
        context: Optional[str] = None,
    ) -> Tuple[str, str]:
        logger = logging.getLogger("vera.agent.market_weather")
        logger.info("agent_start name=%s input=%s", self.name, user_message)

        if conversation_id is None or not conversation_id.strip():
            conversation_id = uuid4().hex

        tools = [
            get_current_weather_tool(),
            get_stock_price_tool(),
        ]

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt,
            name=self.name,
        )
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_message})
        result = agent.invoke({"messages": messages})
        result_messages = result.get("messages", []) if isinstance(result, dict) else []
        assistant_message = extract_last_ai_message(result_messages) or ""
        output = assistant_message

        sources_line = extract_sources_line(result_messages)
        if sources_line:
            output = f"{output}\n\n{sources_line}"

        return output, conversation_id


def _scoped_conversation_id(base_id: str) -> str:
    return f"market_weather:{base_id}"


def get_market_weather_agent_tool(
    llm: Any,
    conversation_id: str,
    *,
    default_context: Optional[str] = None,
) -> StructuredTool:
    def ask_market_weather(message: str, context: Optional[str] = None) -> dict[str, str]:
        scoped_id = _scoped_conversation_id(conversation_id)
        context_value = context if context is not None else default_context
        answer, _ = MarketWeatherAgent(llm=llm).respond(
            message, conversation_id=scoped_id, context=context_value
        )
        return {"answer": answer}

    return StructuredTool.from_function(
        func=ask_market_weather,
        name="ask_market_weather",
        description=(
            "Use for current weather or stock price questions. "
            "Include relevant prior context via the 'context' argument when helpful."
        ),
    )
