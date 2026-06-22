"""
rag_handler.py — Traditional RAG pipeline.

Strategy:
  1. Optionally expand query using conversation history
  2. Dense similarity search via ChromaDB
  3. Single LLM call with retrieved context

Changes from old version:
  - Removed LangChain ConversationalRetrievalChain and ConversationBufferMemory
  - LLM calls go through config.get_llm_response() (LiteLLM)
  - Query expansion is now a direct LLM prompt, same logic preserved
"""

from components.config import get_llm_response
from components.vector_store import VectorStoreManager

# System prompt shared across all interactions
_SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer ONLY from the provided context chunks.
If the answer is not in the context, say: "I cannot answer this based on the provided document."
Always include page number citations in your response (e.g., [Page 3])."""

_QA_TEMPLATE = """\
Context chunks from the document:
{context}

Conversation history (for reference only):
{history}

Question: {question}

Provide a detailed answer with explicit citations referencing page numbers from the context.
Use only the information in the context above — no external knowledge.
"""

_EXPAND_TEMPLATE = """\
Given the conversation history and current question, rewrite the question to be \
self-contained and capture all relevant context from the history.

History:
{history}

Current Question: {question}

Rewritten Question (one sentence):"""


class TraditionalRAG:
    """Single-pass dense retrieval + LLM answer with optional query expansion."""

    def __init__(self, vectorstore, file_name: str | None = None):
        self.vs_manager = VectorStoreManager()
        self.file_name  = file_name

    def get_response(
        self,
        query: str,
        history: list[dict],
        model: str = "Llama 3.3 70B (Groq)",
    ) -> tuple[str, str]:
        """
        Args:
            query:   The user's question.
            history: List of past {question, answer, timestamp} dicts.
            model:   Display name from config.AVAILABLE_MODELS.

        Returns:
            (answer_text, citations_text)
        """
        formatted_history = self._format_history(history)

        # Query expansion when history is available
        expanded_query = self._expand_query(query, formatted_history, model)

        # Dense retrieval
        results = self.vs_manager.similarity_search(
            expanded_query, k=4, file_name=self.file_name
        )

        # Build context string
        context_parts: list[str] = []
        citations: list[str] = []
        for doc, score in results:
            page = doc.metadata.get("page_number", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")[:8]
            context_parts.append(f"[Page {page}] {doc.page_content}")
            citations.append(f"Page {page} | Chunk {chunk_id} | Score {score:.3f}")

        context_text = "\n\n".join(context_parts)

        # Single LLM call
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _QA_TEMPLATE.format(
                    context=context_text,
                    history=formatted_history or "None",
                    question=expanded_query,
                ),
            },
        ]
        answer = get_llm_response(messages, model_display_name=model)

        citations_text = "**Sources:**\n" + "\n".join(f"- {c}" for c in citations)
        return answer, citations_text

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        )

    def _expand_query(self, query: str, history: str, model: str) -> str:
        if not history:
            return query
        messages = [
            {
                "role": "user",
                "content": _EXPAND_TEMPLATE.format(history=history, question=query),
            }
        ]
        try:
            return get_llm_response(messages, model_display_name=model, max_tokens=200)
        except Exception:
            return query  # fallback: use original