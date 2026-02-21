from app.services.agents import vera_agent


def test_vera_agent_appends_sources_line(monkeypatch):
    captured = {}

    monkeypatch.setattr(vera_agent, "get_knowledge_base_tool", lambda: "kb")

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
        return DummyAgent()

    monkeypatch.setattr(vera_agent, "create_agent", fake_create_agent)

    agent = vera_agent.VeraAgent(llm="llm")
    result, conversation_id = agent.respond("hi", conversation_id="cid-1")

    assert captured["tools"] == ["kb"]
    assert conversation_id == "cid-1"
    assert result == "answer\n\nSources: kb"


def test_vera_agent_without_sources_line(monkeypatch):
    monkeypatch.setattr(vera_agent, "get_knowledge_base_tool", lambda: "kb")

    class DummyAgent:
        def invoke(self, payload, config=None):
            return {"messages": [{"role": "assistant", "content": "answer"}]}

    monkeypatch.setattr(vera_agent, "create_agent", lambda **kwargs: DummyAgent())

    agent = vera_agent.VeraAgent(llm="llm")
    result, conversation_id = agent.respond("hi", conversation_id="cid-2")

    assert conversation_id == "cid-2"
    assert result == "answer"


def test_vera_agent_tool_splits_sources_line(monkeypatch):
    captured = {}

    class DummyVeraAgent:
        def __init__(self, llm=None):
            captured["llm"] = llm

        def respond(self, message, conversation_id=None, context=None):
            captured["message"] = message
            captured["conversation_id"] = conversation_id
            captured["context"] = context
            return "answer\n\nSources: kb", "cid-1"

    monkeypatch.setattr(vera_agent, "VeraAgent", DummyVeraAgent)

    tool = vera_agent.get_vera_agent_tool(
        llm="llm",
        conversation_id="cid-1",
        default_context="ctx",
    )
    result = tool.func("hi")

    assert result == {"answer": "answer", "sources_line": "Sources: kb"}
    assert captured["llm"] == "llm"
    assert captured["message"] == "hi"
    assert captured["conversation_id"] == "vera:cid-1"
    assert captured["context"] == "ctx"


def test_vera_agent_tool_without_sources_line(monkeypatch):
    class DummyVeraAgent:
        def __init__(self, llm=None):
            pass

        def respond(self, message, conversation_id=None, context=None):
            return "answer", "cid-2"

    monkeypatch.setattr(vera_agent, "VeraAgent", DummyVeraAgent)

    tool = vera_agent.get_vera_agent_tool(
        llm="llm",
        conversation_id="cid-2",
    )
    result = tool.func("hi")

    assert result == {"answer": "answer"}
