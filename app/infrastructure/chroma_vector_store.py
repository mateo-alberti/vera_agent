from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import List, Sequence

import chromadb

from app.core.config import Settings
from app.domain.ports import VectorEmbeddingItem, VectorSearchMatch, VectorStorePort


@dataclass
class ChromaVectorStore(VectorStorePort):
    def __init__(self) -> None:
        settings = Settings()
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection
        )
        self._logger = logging.getLogger("vera.vector_store")

    def upsert_embeddings(self, items: Sequence[VectorEmbeddingItem]) -> int:
        items_list = list(items)
        if not items_list:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []
        embeddings: List[List[float]] = []

        for item in items_list:
            ids.append(item.id)
            documents.append(item.document)
            metadatas.append(item.metadata or {})
            embeddings.append(item.embedding)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._logger.info("vector_upsert count=%s", len(ids))
        return len(ids)

    def search_by_embedding(
        self, query_embedding: Sequence[float], k: int = 5
    ) -> List[VectorSearchMatch]:
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: List[VectorSearchMatch] = []
        for idx in range(len(ids)):
            matches.append(
                VectorSearchMatch(
                    id=str(ids[idx]),
                    text=documents[idx] or "",
                    metadata=metadatas[idx] or {},
                    distance=float(distances[idx]),
                )
            )

        self._logger.info("vector_search k=%s count=%s", k, len(matches))
        return matches


def get_chroma_vector_store() -> VectorStorePort:
    return ChromaVectorStore()
