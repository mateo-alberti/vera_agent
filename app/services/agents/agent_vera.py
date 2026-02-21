from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from app.prompts import VERA_SYSTEM_PROMPT
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

    def respond(self, user_message: str) -> str:
        logger = logging.getLogger("vera.agent")
        logger.info("agent_start name=%s input=%s", self.name, user_message)

        tools = [
            get_current_weather_tool(),
            get_stock_price_tool(),
            get_knowledge_base_tool(),
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
        result = executor.invoke({"input": user_message})
        output = result["output"]

        sources_line = ""
        for step in result.get("intermediate_steps", []):
            if not step or len(step) < 2:
                continue
            _, observation = step
            if isinstance(observation, dict) and observation.get("sources_line"):
                sources_line = observation["sources_line"]
                break

        if sources_line:
            output = f"{output}\n\n{sources_line}"

        return output
