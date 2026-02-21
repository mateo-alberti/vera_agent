from app.services.agents import orchestrator_agent


def test_orchestrator_appends_sources_line(monkeypatch):
    captured = {}

    class DummyAgent:
        def invoke(self, payload, config=None):
            captured["messages"] = payload["messages"]
            return {
                "messages": [
                    {"role": "assistant", "content": "answer"},
                    {"role": "tool", "content": {"sources_line": "Sources: kb"}},
                ]
            }

    def fake_create_agent(model=None, tools=None, system_prompt=None, name=None):
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["name"] = name
        return DummyAgent()

    monkeypatch.setattr(orchestrator_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(orchestrator_agent, "get_history", lambda _cid: [])

    stored = {}

    def fake_store_turn(conversation_id, user_message, assistant_message):
        stored["conversation_id"] = conversation_id
        stored["user_message"] = user_message
        stored["assistant_message"] = assistant_message

    monkeypatch.setattr(orchestrator_agent, "store_turn", fake_store_turn)
    monkeypatch.setattr(orchestrator_agent, "get_market_weather_agent_tool", lambda *args, **kwargs: "mw")
    monkeypatch.setattr(orchestrator_agent, "get_vera_agent_tool", lambda *args, **kwargs: "vera")

    agent = orchestrator_agent.OrchestratorAgent(llm="llm")
    result, conversation_id = agent.respond("hello", conversation_id="cid-1")

    assert result == "answer\n\nSources: kb"
    assert conversation_id == "cid-1"
    assert stored["conversation_id"] == "router:cid-1"
    assert stored["user_message"] == "hello"
    assert stored["assistant_message"] == "answer"
    assert captured["tools"] == ["mw", "vera"]
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
