"""
vector_store.py
---------------
Manages the ChromaDB vector database.

"""

import logging
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """
    Wraps ChromaDB and OpenAI embeddings into a simple add/search interface.
    """

    COLLECTION_NAME = "policy_documents"

    def __init__(self, api_key: str, persist_dir: str = "./chroma_db"):
        # OpenAI embedding model — converts text → vector of numbers
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        self.persist_dir = persist_dir

        # Create or reload the ChromaDB collection
        self.db = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Embed and store a list of chunk dicts.
        Uses upsert semantics — re-ingesting the same file is safe.
        """
        if not chunks:
            return

        texts     = [c["content"]  for c in chunks]
        metadatas = [
            {
                "source":   c["source"],
                "chunk_id": str(c["chunk_id"]),
                "page":     str(c.get("page", "")),
            }
            for c in chunks
        ]
        # Stable IDs mean re-ingesting the same file just updates existing rows
        ids = [f"{c['source']}__chunk{c['chunk_id']}" for c in chunks]

        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logging.info(f"Added {len(chunks)} chunks to vector store.")

    def search(self, query: str, k: int = 6) -> List[Dict]:
        """
        Semantic search: find the k most relevant chunks for the query.
        Returns a list of dicts with content, source, page, and relevance score.
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=k)

        chunks = []
        for doc, score in results:
            # Filter out irrelevant noise below 0.25 relevance
            if score >= 0.25:
                chunks.append(
                    {
                        "content":  doc.page_content,
                        "source":   doc.metadata.get("source", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "page":     doc.metadata.get("page", ""),
                        "score":    round(score, 3),
                    }
                )

        logging.info(f"Search returned {len(chunks)} relevant chunks for: '{query[:60]}'")
        return chunks

    def is_empty(self) -> bool:
        """Returns True if no documents have been ingested yet."""
        return self.db._collection.count() == 0

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self.db._collection.count()

    def clear(self) -> None:
        """Delete all stored chunks (useful for re-ingestion)."""
        self.db.delete_collection()
        self.db = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        logging.info("Vector store cleared.")
