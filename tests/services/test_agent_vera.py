from app.services.agents import agent_vera


def test_vera_agent_appends_sources_line(monkeypatch):
    captured = {}

    monkeypatch.setattr(agent_vera, "get_current_weather_tool", lambda: "weather")
    monkeypatch.setattr(agent_vera, "get_stock_price_tool", lambda: "stock")
    monkeypatch.setattr(agent_vera, "get_knowledge_base_tool", lambda: "kb")

    class DummyAgent:
        def invoke(self, payload):
            captured["messages"] = payload["messages"]
            return {
                "messages": [
                    {"role": "assistant", "content": "answer"},
                    {"role": "tool", "content": {"sources_line": "Sources: kb"}},
                ]
            }

    def fake_create_agent(model=None, tools=None, system_prompt=None, name=None):
        captured["tools"] = tools
        return DummyAgent()

    monkeypatch.setattr(agent_vera, "create_agent", fake_create_agent)

    agent = agent_vera.VeraAgent(llm="llm")
    result, conversation_id = agent.respond("hi", conversation_id="cid-1")

    assert captured["tools"] == ["weather", "stock", "kb"]
    assert conversation_id == "cid-1"
    assert result == "answer\n\nSources: kb"


def test_vera_agent_without_sources_line(monkeypatch):
    monkeypatch.setattr(agent_vera, "get_current_weather_tool", lambda: "weather")
    monkeypatch.setattr(agent_vera, "get_stock_price_tool", lambda: "stock")
    monkeypatch.setattr(agent_vera, "get_knowledge_base_tool", lambda: "kb")

    class DummyAgent:
        def invoke(self, payload):
            return {"messages": [{"role": "assistant", "content": "answer"}]}

    monkeypatch.setattr(agent_vera, "create_agent", lambda **kwargs: DummyAgent())

    agent = agent_vera.VeraAgent(llm="llm")
    result, conversation_id = agent.respond("hi", conversation_id="cid-2")

    assert conversation_id == "cid-2"
    assert result == "answer"
