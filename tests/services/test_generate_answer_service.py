from app.services import generate_answer_service


def test_generate_answer_service_uses_agent(monkeypatch):
    captured = {}

    class DummyLLMPort:
        def get_chat_model(self):
            captured["llm"] = "llm"
            return "llm"

    class DummyAgent:
        def __init__(self, llm=None):
            captured["llm"] = llm

        def respond(self, user_message):
            captured["message"] = user_message
            return "ok"

    monkeypatch.setattr(
        generate_answer_service, "get_llm_port", lambda: DummyLLMPort()
    )
    monkeypatch.setattr(generate_answer_service, "VeraAgent", DummyAgent)

    service = generate_answer_service.GenerateAnswerService()
    result = service.respond("hello")

    assert result == "ok"
    assert captured["llm"] == "llm"
    assert captured["message"] == "hello"
