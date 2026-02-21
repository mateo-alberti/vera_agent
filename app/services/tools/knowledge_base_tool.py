from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from app.domain.embeddings_port import get_embeddings_port
from app.domain.vector_store_port import (
    VectorSearchMatch,
    VectorStorePort,
    get_vector_store_port,
)


def knowledge_base_search(
    query: str,
    k: int = 5,
) -> dict[str, Any]:
    """Search the vector store for semantically similar text."""
    embeddings_port = get_embeddings_port()
    embedding_result = embeddings_port.embed_texts([query])
    store: VectorStorePort = get_vector_store_port()
    results: list[VectorSearchMatch] = store.search_by_embedding(
        query_embedding=embedding_result.vectors[0],
        k=k,
    )
    sources: list[str] = []
    for result in results:
        file_name = (result.metadata or {}).get("file_name")
        if file_name and file_name not in sources:
            sources.append(str(file_name))
    sources_line = f"Sources: {', '.join(sources)}" if sources else ""
    return {
        "status": "ok",
        "query": query,
        "count": len(results),
        "sources": sources,
        "sources_line": sources_line,
        "results": [
            {
                "id": result.id,
                "text": result.text,
                "metadata": result.metadata,
                "distance": result.distance,
            }
            for result in results
        ],
    }


def get_knowledge_base_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=knowledge_base_search,
        name="knowledge_base_search",
        description=(
            "Search the internal knowledge base about Vera, Vera terms and conditions and fintech regulations using semantic similarity. "
            "Provide a short query and optional k for number of results."
        ),
    )
