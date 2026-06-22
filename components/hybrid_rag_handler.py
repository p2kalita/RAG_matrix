"""
hybrid_rag_handler.py — Hybrid RAG pipeline (BM25 + Dense + RRF fusion).

Strategy:
  1. BM25 keyword retrieval over all PDF chunks (rank_bm25)
  2. Dense vector retrieval via ChromaDB
  3. Reciprocal Rank Fusion (RRF) to merge ranked results
  4. Single LLM call with fused context

The RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60 (standard)
"""

import re
from rank_bm25 import BM25Okapi  # type: ignore

from components.config import get_llm_response
from components.vector_store import VectorStoreManager

_SYSTEM_PROMPT = """You are a precise document Q&A assistant using hybrid search.
Answer ONLY from the provided context chunks (retrieved by both keyword and semantic search).
If the answer is not in the context, say: "I cannot answer this based on the provided document."
Always include page number citations in your response (e.g., [Page 3])."""

_QA_TEMPLATE = """\
Context chunks from the document (hybrid-ranked):
{context}

Conversation history (for reference only):
{history}

Question: {question}

Provide a detailed answer with explicit citations referencing page numbers.
Use only the information in the context above — no external knowledge.
"""

RRF_K = 60  # Standard constant; higher = more rank compression


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class HybridRAG:
    """
    BM25 + Dense retrieval with Reciprocal Rank Fusion.

    The BM25 index is rebuilt each call from the stored PDF chunks so it
    always reflects the latest uploaded document without requiring a
    separate index-build step in the app.
    """

    def __init__(self, vectorstore, file_name: str | None = None):
        self.vs_manager = VectorStoreManager()
        self.file_name  = file_name

    def get_response(
        self,
        query: str,
        history: list[dict],
        model: str = "Llama 3.3 70B (Groq)",
        top_k: int = 4,
    ) -> tuple[str, str]:
        """
        Args:
            query:   User question.
            history: Past Q-A pairs from the vector store.
            model:   Display name from config.AVAILABLE_MODELS.
            top_k:   Number of final fused chunks to use as context.

        Returns:
            (answer_text, citations_text)
        """
        # 1. Fetch all corpus chunks for BM25
        all_chunks = self.vs_manager.get_all_pdf_chunks(self.file_name)
        if not all_chunks:
            return (
                "No document chunks found. Please upload a PDF first.",
                "No sources.",
            )

        corpus_texts = [c["text"] for c in all_chunks]
        corpus_meta  = [c["metadata"] for c in all_chunks]

        # 2. BM25 retrieval
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))

        # Rank documents by BM25 score (descending)
        bm25_ranked = sorted(
            range(len(corpus_texts)), key=lambda i: bm25_scores[i], reverse=True
        )

        # 3. Dense retrieval
        dense_results = self.vs_manager.similarity_search(
            query, k=min(top_k * 2, len(corpus_texts)), file_name=self.file_name
        )
        # Map dense docs back to corpus indices by text matching
        dense_ranked: list[int] = []
        for doc, _ in dense_results:
            for idx, text in enumerate(corpus_texts):
                if doc.page_content.strip() == text.strip():
                    dense_ranked.append(idx)
                    break

        # 4. Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked[: top_k * 3]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused_ranked = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        # 5. Build context + citations
        context_parts: list[str] = []
        citations: list[str] = []
        for rank_pos, idx in enumerate(fused_ranked):
            text = corpus_texts[idx]
            meta = corpus_meta[idx]
            page     = meta.get("page_number", "?")
            chunk_id = str(meta.get("chunk_id", "?"))[:8]
            bm25_s   = f"{bm25_scores[idx]:.2f}"
            rrf_s    = f"{rrf_scores[idx]:.4f}"

            context_parts.append(f"[Page {page}] {text}")
            citations.append(
                f"Page {page} | Chunk {chunk_id} | BM25={bm25_s} | RRF={rrf_s}"
            )

        context_text = "\n\n".join(context_parts)
        formatted_history = self._format_history(history)

        # 6. LLM answer
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _QA_TEMPLATE.format(
                    context=context_text,
                    history=formatted_history or "None",
                    question=query,
                ),
            },
        ]
        answer = get_llm_response(messages, model_display_name=model)

        citations_text = "**Sources (BM25 + Dense RRF):**\n" + "\n".join(
            f"- {c}" for c in citations
        )
        return answer, citations_text

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        )
