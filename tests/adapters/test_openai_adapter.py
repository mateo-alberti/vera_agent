from app.infrastructure import openai_adapter
from app.domain.embeddings_port import EmbeddingResult


def test_openai_adapter_generate_answer(monkeypatch):
    captured = {}

    class DummySettings:
        openai_answer_model = "gpt-test"
        openai_api_key = "key"
        openai_embedding_model = "embed-test"

    class DummyChat:
        def __init__(self, model=None, api_key=None):
            captured["model"] = model
            captured["api_key"] = api_key

    class DummyEmbeddings:
        def __init__(self, model=None, api_key=None):
            captured["embed_model"] = model
            captured["embed_key"] = api_key

        def embed_documents(self, docs):
            return [[0.1, 0.2] for _ in docs]

    class DummyChain:
        def __init__(self, value):
            self._value = value

        def __or__(self, other):
            return self

        def invoke(self, payload):
            return self._value

    def fake_from_messages(messages):
        captured["messages"] = messages

        class DummyPrompt:
            def __or__(self, other):
                return DummyChain("answer")

        return DummyPrompt()

    monkeypatch.setattr(openai_adapter, "Settings", DummySettings)
    monkeypatch.setattr(openai_adapter, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(openai_adapter, "OpenAIEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(openai_adapter.ChatPromptTemplate, "from_messages", fake_from_messages)
    monkeypatch.setattr(openai_adapter, "StrOutputParser", lambda: object())

    adapter = openai_adapter.OpenAIAdapter()
    result = adapter.generate_answer("hello", system="System")

    assert result.text == "answer"
    assert captured["model"] == "gpt-test"
    assert captured["api_key"] == "key"
    assert captured["messages"][0] == ("system", "System")
    assert captured["messages"][1] == ("user", "{input}")


def test_openai_adapter_generate_answer_defaults_system_prompt(monkeypatch):
    captured = {}

    class DummySettings:
        openai_answer_model = "gpt-test"
        openai_api_key = "key"
        openai_embedding_model = "embed-test"

    class DummyChat:
        def __init__(self, model=None, api_key=None):
            pass

    class DummyEmbeddings:
        def __init__(self, model=None, api_key=None):
            pass

    class DummyChain:
        def __init__(self, value):
            self._value = value

        def __or__(self, other):
            return self

        def invoke(self, payload):
            return self._value

    def fake_from_messages(messages):
        captured["messages"] = messages

        class DummyPrompt:
            def __or__(self, other):
                return DummyChain("answer")

        return DummyPrompt()

    monkeypatch.setattr(openai_adapter, "Settings", DummySettings)
    monkeypatch.setattr(openai_adapter, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(openai_adapter, "OpenAIEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(openai_adapter.ChatPromptTemplate, "from_messages", fake_from_messages)
    monkeypatch.setattr(openai_adapter, "StrOutputParser", lambda: object())

    adapter = openai_adapter.OpenAIAdapter()
    adapter.generate_answer("hello")

    assert captured["messages"][0] == ("system", "You are a helpful assistant.")


def test_openai_adapter_uses_provided_settings(monkeypatch):
    class DummySettings:
        openai_answer_model = "gpt-local"
        openai_api_key = "local-key"
        openai_embedding_model = "embed-local"

    settings = DummySettings()

    def _should_not_call_settings():
        raise AssertionError("Settings() should not be called when settings are provided")

    monkeypatch.setattr(openai_adapter, "Settings", _should_not_call_settings)

    class DummyChat:
        def __init__(self, model=None, api_key=None):
            self.model = model
            self.api_key = api_key

    class DummyEmbeddings:
        def __init__(self, model=None, api_key=None):
            self.model = model
            self.api_key = api_key

        def embed_documents(self, docs):
            return [[0.0, 0.0] for _ in docs]

    monkeypatch.setattr(openai_adapter, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(openai_adapter, "OpenAIEmbeddings", DummyEmbeddings)

    adapter = openai_adapter.OpenAIAdapter(settings=settings)

    assert adapter.llm.model == "gpt-local"
    assert adapter.llm.api_key == "local-key"
    assert adapter.embeddings.model == "embed-local"
    assert adapter.embeddings.api_key == "local-key"


def test_openai_adapter_embed_texts(monkeypatch):
    class DummySettings:
        openai_answer_model = "gpt-test"
        openai_api_key = "key"
        openai_embedding_model = "embed-test"

    class DummyChat:
        def __init__(self, model=None, api_key=None):
            self.model = model
            self.api_key = api_key

    class DummyEmbeddings:
        def __init__(self, model=None, api_key=None):
            self.model = model
            self.api_key = api_key

        def embed_documents(self, docs):
            return [[1.0, 2.0] for _ in docs]

    monkeypatch.setattr(openai_adapter, "Settings", DummySettings)
    monkeypatch.setattr(openai_adapter, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(openai_adapter, "OpenAIEmbeddings", DummyEmbeddings)

    adapter = openai_adapter.OpenAIAdapter()
    result = adapter.embed_texts(["a", "b"])

    assert isinstance(result, EmbeddingResult)
    assert result.vectors == [[1.0, 2.0], [1.0, 2.0]]
