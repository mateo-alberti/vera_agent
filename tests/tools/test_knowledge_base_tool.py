from app.domain.ports import EmbeddingResult, VectorSearchMatch
from app.services.tools import knowledge_base_tool


def test_knowledge_base_search_collects_sources(monkeypatch):
    class DummyAdapter:
        def embed_texts(self, texts):
            return EmbeddingResult(vectors=[[0.1, 0.2]], raw=None)

    class DummyStore:
        def search_by_embedding(self, query_embedding, k=5):
            return [
                VectorSearchMatch(
                    id="1",
                    text="one",
                    metadata={"file_name": "doc1.md"},
                    distance=0.1,
                ),
                VectorSearchMatch(
                    id="2",
                    text="two",
                    metadata={"file_name": "doc1.md"},
                    distance=0.2,
                ),
                VectorSearchMatch(
                    id="3",
                    text="three",
                    metadata={},
                    distance=0.3,
                ),
            ]

    store = DummyStore()
    monkeypatch.setattr(knowledge_base_tool, "OpenAIAdapter", DummyAdapter)
    monkeypatch.setattr(knowledge_base_tool, "get_chroma_vector_store", lambda: store)

    result = knowledge_base_tool.knowledge_base_search("pricing")

    assert result["status"] == "ok"
    assert result["query"] == "pricing"
    assert result["count"] == 3
    assert result["sources"] == ["doc1.md"]
    assert result["sources_line"] == "Sources: doc1.md"
    assert result["results"][0]["id"] == "1"
