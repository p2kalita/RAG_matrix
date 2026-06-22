"""
vector_store.py — Singleton ChromaDB manager with HuggingFace embeddings.

Changes from old version:
- OpenAIEmbeddings → HuggingFaceEmbeddings (local, free)
- Removed .persist() calls  — Chroma ≥ 0.4 auto-persists
- Fixed similarity_search API
- Added get_all_pdf_chunks() for BM25 corpus building
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from components.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
import uuid
from datetime import datetime


def _build_filter(*conditions: dict) -> dict:
    """
    Build a ChromaDB-compatible where-filter.

    ChromaDB 1.5+ requires exactly ONE top-level operator:
      - Single condition  → pass through as-is  e.g. {"type": "pdf_chunk"}
      - Multiple conditions → wrap in $and       e.g. {"$and": [{...}, {...}]}
    """
    active = [c for c in conditions if c]  # drop any None/empty dicts
    if len(active) == 1:
        return active[0]
    return {"$and": active}


class VectorStoreManager:
    """Singleton that owns the single ChromaDB collection used by all pipelines."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = Chroma(
            collection_name="unified_store",
            embedding_function=self.embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )

    # ── PDF Chunks ────────────────────────────────────────────────────────────

    def add_pdf_chunks(self, chunks: list, file_name: str):
        """Embed and store DocumentChunk objects. Returns the Chroma collection."""
        texts     = [c.text for c in chunks]
        metadatas = [
            {**c.metadata, "type": "pdf_chunk", "file_name": file_name}
            for c in chunks
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return self.store

    def get_all_pdf_chunks(self, file_name: str | None = None) -> list[dict]:
        """
        Return all stored PDF chunk dicts for BM25 corpus building.
        Each dict has keys: text, metadata.
        """
        results = self.store.get(
            where={"type": "pdf_chunk"}
            if file_name is None
            else {"$and": [{"type": "pdf_chunk"}, {"file_name": file_name}]},
            include=["documents", "metadatas"],
        )
        chunks = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({"text": text, "metadata": meta})
        return chunks

    # ── Conversation History ──────────────────────────────────────────────────

    def add_conversation(self, question: str, answer: str, rag_type: str):
        """Store a Q-A pair for later history retrieval."""
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.store.add_texts(
            texts=[f"Question: {question}\nAnswer: {answer}"],
            metadatas=[
                {
                    "conversation_id": conversation_id,
                    "timestamp":       timestamp,
                    "rag_type":        rag_type,
                    "type":            "conversation",
                    "question":        question,
                    "answer":          answer,
                }
            ],
            ids=[conversation_id],
        )

    def get_relevant_history(self, query: str, rag_type: str, k: int = 5) -> list[dict]:
        """Retrieve semantically similar past Q-A pairs for the given rag_type."""
        results = self.store.similarity_search(
            query=query,
            k=k,
            filter={"$and": [{"type": "conversation"}, {"rag_type": rag_type}]},
        )

        history = []
        for doc in results:
            history.append(
                {
                    "question":  doc.metadata["question"],
                    "answer":    doc.metadata["answer"],
                    "timestamp": doc.metadata["timestamp"],
                }
            )

        history.sort(key=lambda x: x["timestamp"])
        return history

    # ── Retriever ─────────────────────────────────────────────────────────────

    def get_retriever(self, file_name: str | None = None, k: int = 4):
        """Return a LangChain retriever scoped to PDF chunks (optionally by file)."""
        f = _build_filter(
            {"type": "pdf_chunk"},
            {"file_name": file_name} if file_name else {},
        )
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": f},
        )

    def similarity_search(self, query: str, k: int = 4, file_name: str | None = None):
        """Direct similarity search returning (Document, score) pairs."""
        f = _build_filter(
            {"type": "pdf_chunk"},
            {"file_name": file_name} if file_name else {},
        )
        return self.store.similarity_search_with_score(query=query, k=k, filter=f)