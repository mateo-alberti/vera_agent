from app.services import generate_answer_service


def test_generate_answer_service_uses_agent(monkeypatch):
    captured = {}

    class DummySettings:
        openai_answer_model = "gpt-test"
        openai_api_key = "key"

    class DummyChat:
        def __init__(self, model=None, api_key=None):
            captured["model"] = model
            captured["api_key"] = api_key

    class DummyAgent:
        def __init__(self, llm=None):
            captured["llm"] = llm

        def respond(self, user_message):
            captured["message"] = user_message
            return "ok"

    monkeypatch.setattr(generate_answer_service, "Settings", DummySettings)
    monkeypatch.setattr(generate_answer_service, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(generate_answer_service, "VeraAgent", DummyAgent)

    service = generate_answer_service.GenerateAnswerService()
    result = service.respond("hello")

    assert result == "ok"
    assert captured["model"] == "gpt-test"
    assert captured["api_key"] == "key"
    assert captured["message"] == "hello"
