from app.services.agents import market_weather_agent


def test_market_weather_agent_tool_returns_answer_only(monkeypatch):
    captured = {}

    class DummyMarketWeatherAgent:
        def __init__(self, llm=None):
            captured["llm"] = llm

        def respond(self, message, conversation_id=None, context=None):
            captured["message"] = message
            captured["conversation_id"] = conversation_id
            captured["context"] = context
            return "forecast\n\nSources: kb", "cid-1"

    monkeypatch.setattr(market_weather_agent, "MarketWeatherAgent", DummyMarketWeatherAgent)

    tool = market_weather_agent.get_market_weather_agent_tool(
        llm="llm",
        conversation_id="cid-1",
        default_context="ctx",
    )
    result = tool.func("weather")

    assert result == {"answer": "forecast\n\nSources: kb"}
    assert captured["llm"] == "llm"
    assert captured["message"] == "weather"
    assert captured["conversation_id"] == "market_weather:cid-1"
    assert captured["context"] == "ctx"
