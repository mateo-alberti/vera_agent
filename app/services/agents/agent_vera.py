from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class VeraAgent:
    llm: ChatOpenAI
    system_prompt: str = "You are Vera, a concise and helpful assistant."

    def respond(self, user_message: str) -> str:
        template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", "{input}")]
        )
        chain = template | self.llm | StrOutputParser()
        return chain.invoke({"input": user_message})
