from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import RLock
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.prompts import VERA_SYSTEM_PROMPT
from app.services.tools import (
    get_current_weather_tool,
    get_knowledge_base_tool,
    get_stock_price_tool,
)


_MEMORY_LOCK = RLock()
_MEMORY_BY_ID: Dict[str, ConversationBufferWindowMemory] = {}
_MEMORY_WINDOW = 6


def _get_or_create_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    with _MEMORY_LOCK:
        memory = _MEMORY_BY_ID.get(conversation_id)
        if memory is None:
            memory = ConversationBufferWindowMemory(
                k=_MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output",
            )
            _MEMORY_BY_ID[conversation_id] = memory
        return memory


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
        memory = _get_or_create_memory(conversation_id)

        tools = [
            get_current_weather_tool(),
            get_stock_price_tool(),
            get_knowledge_base_tool(),
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
        )
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

        return output, conversation_id
