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

        def respond(self, user_message, conversation_id=None):
            captured["message"] = user_message
            captured["conversation_id"] = conversation_id
            return "ok", "cid-1"

    monkeypatch.setattr(
        generate_answer_service, "get_llm_port", lambda: DummyLLMPort()
    )
    monkeypatch.setattr(generate_answer_service, "OrchestratorAgent", DummyAgent)

    service = generate_answer_service.GenerateAnswerService()
    result, conversation_id = service.respond("hello", conversation_id="cid-1")

    assert result == "ok"
    assert conversation_id == "cid-1"
    assert captured["llm"] == "llm"
    assert captured["message"] == "hello"
    assert captured["conversation_id"] == "cid-1"
