from app.services.agents import agent_vera


def test_vera_agent_appends_sources_line(monkeypatch):
    captured = {}

    monkeypatch.setattr(agent_vera, "get_current_weather_tool", lambda: "weather")
    monkeypatch.setattr(agent_vera, "get_stock_price_tool", lambda: "stock")
    monkeypatch.setattr(agent_vera, "get_knowledge_base_tool", lambda: "kb")

    def fake_create_tool_calling_agent(llm, tools, prompt):
        captured["tools"] = tools
        return "agent"

    class DummyExecutor:
        def __init__(self, agent=None, tools=None, return_intermediate_steps=False):
            self.agent = agent
            self.tools = tools
            self.return_intermediate_steps = return_intermediate_steps

        def invoke(self, payload):
            return {
                "output": "answer",
                "intermediate_steps": [("tool", {"sources_line": "Sources: kb"})],
            }

    monkeypatch.setattr(agent_vera, "create_tool_calling_agent", fake_create_tool_calling_agent)
    monkeypatch.setattr(agent_vera, "AgentExecutor", DummyExecutor)

    agent = agent_vera.VeraAgent(llm="llm")
    result = agent.respond("hi")

    assert captured["tools"] == ["weather", "stock", "kb"]
    assert result == "answer\n\nSources: kb"


def test_vera_agent_without_sources_line(monkeypatch):
    monkeypatch.setattr(agent_vera, "get_current_weather_tool", lambda: "weather")
    monkeypatch.setattr(agent_vera, "get_stock_price_tool", lambda: "stock")
    monkeypatch.setattr(agent_vera, "get_knowledge_base_tool", lambda: "kb")

    class DummyExecutor:
        def __init__(self, agent=None, tools=None, return_intermediate_steps=False):
            self.agent = agent
            self.tools = tools
            self.return_intermediate_steps = return_intermediate_steps

        def invoke(self, payload):
            return {"output": "answer", "intermediate_steps": []}

    monkeypatch.setattr(agent_vera, "create_tool_calling_agent", lambda *args, **kwargs: "agent")
    monkeypatch.setattr(agent_vera, "AgentExecutor", DummyExecutor)

    agent = agent_vera.VeraAgent(llm="llm")
    result = agent.respond("hi")

    assert result == "answer"
