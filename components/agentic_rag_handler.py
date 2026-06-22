"""
agentic_rag_handler.py — Agentic RAG via a ReAct-style iterative agent loop.

Strategy (preserves the 3-role design from the original crewai version):
  - Role 1 (Context Analyzer): Expand and decompose the query using history
  - Role 2 (Researcher):       Iteratively retrieve + assess sufficiency (up to 3 rounds)
  - Role 3 (Technical Writer): Synthesize final answer with citations

Changes from old version:
  - Removed crewai entirely
  - All 3 roles are now sequential LLM prompts using config.get_llm_response()
  - ReAct loop drives multi-step retrieval until the researcher deems context sufficient
"""

import json
import re

from components.config import get_llm_response
from components.vector_store import VectorStoreManager

MAX_ITERATIONS = 3  # Maximum retrieval rounds for the researcher agent


# ── Role Prompts ──────────────────────────────────────────────────────────────

_ANALYZER_SYSTEM = """You are a Context Analyzer agent.
Your job is to understand conversation history and decompose the user's question
into 1-3 focused sub-queries that will be used to retrieve relevant document chunks.
Return ONLY a JSON array of sub-query strings, e.g.: ["sub-query 1", "sub-query 2"]"""

_RESEARCHER_SYSTEM = """You are a Research Analyst agent.
You receive retrieved document chunks and must determine if they contain
sufficient information to fully answer the question.
Respond with JSON: {"sufficient": true/false, "summary": "<brief summary of what you found>", "missing": "<what is still missing, or null>"}
Do not answer the question directly — only assess the retrieved evidence."""

_WRITER_SYSTEM = """You are a Technical Writer agent.
Based strictly on the research notes provided, write a comprehensive answer to the user's question.
Rules:
- Use ONLY information from the research notes (no external knowledge)
- Cite page numbers explicitly using [Page N] format
- If research notes are insufficient, clearly state what could not be answered
- End with a 'Sources' section listing all cited pages"""


class AgenticRAG:
    """
    Three-agent ReAct loop:
      Analyzer → [Researcher ↔ Retriever] (up to MAX_ITERATIONS) → Writer
    """

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
            query:   User's question.
            history: Past Q-A conversation pairs.
            model:   Display name from config.AVAILABLE_MODELS.

        Returns:
            (final_answer, citations_text)
        """
        formatted_history = self._format_history(history)

        # ── Agent 1: Context Analyzer ─────────────────────────────────────────
        sub_queries = self._run_analyzer(query, formatted_history, model)

        # ── Agent 2: Researcher (ReAct loop) ─────────────────────────────────
        all_chunks:     list[str] = []
        all_citations:  list[str] = []

        for iteration in range(MAX_ITERATIONS):
            # Retrieve for each sub-query
            iteration_chunks: list[str] = []
            for sq in sub_queries:
                results = self.vs_manager.similarity_search(
                    sq, k=3, file_name=self.file_name
                )
                for doc, score in results:
                    page     = doc.metadata.get("page_number", "?")
                    chunk_id = str(doc.metadata.get("chunk_id", "?"))[:8]
                    chunk_text = f"[Page {page}] {doc.page_content}"
                    if chunk_text not in all_chunks:
                        all_chunks.append(chunk_text)
                        iteration_chunks.append(chunk_text)
                        all_citations.append(
                            f"Page {page} | Chunk {chunk_id} | Score {score:.3f}"
                        )

            # Researcher assesses sufficiency
            assessment = self._run_researcher(
                query=query,
                chunks=all_chunks,
                model=model,
                iteration=iteration,
            )

            if assessment.get("sufficient", False):
                break

            # If not sufficient, reformulate sub-queries around missing info
            missing = assessment.get("missing", "")
            if missing:
                sub_queries = [missing]  # narrow next retrieval round

        # ── Agent 3: Technical Writer ─────────────────────────────────────────
        answer = self._run_writer(
            query=query,
            research_notes="\n\n".join(all_chunks),
            researcher_summary=assessment.get("summary", ""),
            model=model,
        )

        # Extract citations from final answer (page references)
        inline_pages = re.findall(r"\[Page\s*(\d+)\]", answer)
        cited = list(dict.fromkeys(inline_pages))  # deduplicate, preserve order

        citations_text = "**Sources (Agentic ReAct):**\n"
        if cited:
            citations_text += "\n".join(f"- Page {p}" for p in cited)
        elif all_citations:
            citations_text += "\n".join(f"- {c}" for c in all_citations[:6])
        else:
            citations_text += "- No explicit citations found."

        return answer, citations_text

    # ── Agent Implementations ─────────────────────────────────────────────────

    def _run_analyzer(self, query: str, history: str, model: str) -> list[str]:
        """Decompose the query into focused sub-queries."""
        user_content = (
            f"Conversation history:\n{history}\n\nUser question: {query}"
            if history
            else f"User question: {query}"
        )
        messages = [
            {"role": "system", "content": _ANALYZER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        raw = get_llm_response(messages, model_display_name=model, max_tokens=300)
        try:
            sub_queries = json.loads(raw)
            if isinstance(sub_queries, list) and sub_queries:
                return [str(sq) for sq in sub_queries]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: treat entire response as one sub-query, or use original
        return [query]

    def _run_researcher(
        self,
        query: str,
        chunks: list[str],
        model: str,
        iteration: int,
    ) -> dict:
        """Assess whether retrieved chunks are sufficient to answer the query."""
        context_text = "\n\n".join(chunks[:12])  # cap context size
        user_content = (
            f"Question: {query}\n\n"
            f"Retrieved chunks (iteration {iteration + 1}):\n{context_text}"
        )
        messages = [
            {"role": "system", "content": _RESEARCHER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        raw = get_llm_response(messages, model_display_name=model, max_tokens=300)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, assume sufficient to avoid infinite loop
            return {"sufficient": True, "summary": raw, "missing": None}

    def _run_writer(
        self,
        query: str,
        research_notes: str,
        researcher_summary: str,
        model: str,
    ) -> str:
        """Synthesize the final user-facing answer from research notes."""
        user_content = (
            f"User question: {query}\n\n"
            f"Researcher summary: {researcher_summary}\n\n"
            f"Full research notes:\n{research_notes}"
        )
        messages = [
            {"role": "system", "content": _WRITER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        return get_llm_response(messages, model_display_name=model, max_tokens=1500)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        )