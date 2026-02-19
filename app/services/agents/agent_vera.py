from __future__ import annotations

from dataclasses import dataclass
import logging

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.services.tools import get_current_weather_tool, get_stock_price_tool


@dataclass
class VeraAgent:
    llm: ChatOpenAI
    system_prompt: str = "You are Vera, a concise and helpful assistant."
    name: str = "vera"

    def respond(self, user_message: str) -> str:
        logger = logging.getLogger("vera.agent")
        logger.info("agent_start name=%s input=%s", self.name, user_message)

        tools = [get_current_weather_tool(), get_stock_price_tool()]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        result = executor.invoke({"input": user_message})
        return result["output"]
