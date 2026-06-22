"""
tracking.py — Opik (Comet) trace + span tracking for RAG pipeline comparison.

Data Model
----------
Each user query  → one parent Trace  (project: "rag-comparison-app")
Each pipeline    → one Span inside that trace, typed "llm"

What is recorded per span
-------------------------
  name       : "traditional_rag" | "hybrid_rag" | "agentic_rag"
  type       : "llm"
  input      : {query}
  output     : {response, citations}
  metadata   : {rag_type, model, latency_s, word_count,
                num_citations, guardrail_warnings}
  tags       : [rag_type, short_model_name]
  model      : litellm model string

Opik Dashboard
--------------
  https://www.comet.com/opik  →  project "rag-comparison-app"
  • Each row    = one user question (Trace)
  • Expand row  = 3 spans (Traditional / Hybrid / Agentic)
  • Sort by metadata.latency_s  to find fastest pipeline
  • Filter by tag "hybrid"      to isolate one pipeline across queries

Falls back to stdlib logging if OPIK_API_KEY is not set.
"""

import os
import time
import logging
import uuid
import datetime
from typing import Any

logger = logging.getLogger("rag_tracker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

OPIK_PROJECT   = "rag-comparison-app"
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "p2kalita")

# ── In-memory trace store (keyed by our own trace UUID) ───────────────────────
# We store the Opik Trace object so we can attach spans to it later.
_active_traces: dict[str, Any] = {}

# ── Opik client initialisation ────────────────────────────────────────────────
_opik_client = None

try:
    if os.getenv("OPIK_API_KEY"):
        import opik  # type: ignore

        opik.configure(
            api_key=os.getenv("OPIK_API_KEY"),
            workspace=OPIK_WORKSPACE,
            force=True,        # never block on interactive prompts
            use_local=False,
        )
        _opik_client = opik.Opik()
        logger.info("Opik enabled → workspace=%s  project=%s", OPIK_WORKSPACE, OPIK_PROJECT)
    else:
        logger.info("OPIK_API_KEY not set — local logging only.")
except ImportError:
    logger.warning("opik package not installed — local logging only.")
except Exception as exc:
    logger.warning("Opik init failed (%s) — local logging only.", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ── Public API ────────────────────────────────────────────────────────────────

def start_query_trace(query: str, model: str) -> str:
    """
    Open one parent Opik Trace for a new user query.

    Returns:
        trace_key — a UUID4 string used by subsequent track_pipeline() calls.
    """
    trace_key = str(uuid.uuid4())

    if _opik_client:
        try:
            trace = _opik_client.trace(
                name="user_query",
                project_name=OPIK_PROJECT,
                input={"query": query},
                metadata={"model": model},
                tags=["rag-comparison"],
                start_time=_now(),
            )
            _active_traces[trace_key] = trace
            logger.info("[TRACE START] key=%s  query=%r  model=%s", trace_key, query[:80], model)
        except Exception as exc:
            logger.warning("Opik trace start failed: %s", exc)
    else:
        logger.info("[TRACE START] key=%s  query=%r  model=%s", trace_key, query[:80], model)

    return trace_key


def track_pipeline(
    *,
    trace_id: str,
    rag_type: str,
    query: str,
    response: str,
    citations: str,
    latency_s: float,
    model: str,
    has_guardrail_warnings: bool = False,
) -> None:
    """
    Record one pipeline's result as an LLM Span inside the parent trace.

    Args:
        trace_id:               Key returned by start_query_trace().
        rag_type:               "traditional" | "hybrid" | "agentic"
        query:                  Original user question.
        response:               Final (possibly guardrail-prefixed) response text.
        citations:              Citations text block from the pipeline.
        latency_s:              Wall-clock seconds for this pipeline end-to-end.
        model:                  Display name of LLM used.
        has_guardrail_warnings: True if apply_guardrails() prepended a warning.
    """
    span_name          = f"{rag_type}_rag"
    word_count         = len(response.split())
    num_citations      = len([l for l in citations.splitlines() if l.strip().startswith("-")])
    short_model        = model.split("(")[0].strip()

    metadata = {
        "rag_type":            rag_type,
        "model":               model,
        "latency_s":           round(latency_s, 3),
        "word_count":          word_count,
        "num_citations":       num_citations,
        "guardrail_warnings":  has_guardrail_warnings,
    }

    logger.info(
        "[SPAN] %-15s | %.2fs | %d words | %d cites | warnings=%s",
        span_name, latency_s, word_count, num_citations, has_guardrail_warnings,
    )

    trace = _active_traces.get(trace_id)
    if not trace:
        return

    try:
        span = trace.span(
            name=span_name,
            type="llm",
            input={"query": query},
            output={"response": response, "citations": citations},
            metadata=metadata,
            tags=[rag_type, short_model],
            model=model,
            start_time=_now() - datetime.timedelta(seconds=latency_s),
            end_time=_now(),
        )
        span.end()
    except Exception as exc:
        logger.warning("Opik span failed: %s", exc)


def end_query_trace(trace_id: str, results: dict[str, dict]) -> None:
    """
    Close the parent Opik Trace with a comparison summary output.

    Args:
        trace_id: Key from start_query_trace().
        results:  {rag_type: {"latency_s": float, "word_count": int}}
    """
    ranking = sorted(results, key=lambda k: results[k].get("latency_s", 999))
    summary = {rt: results[rt] for rt in results}
    summary["fastest_pipeline"] = ranking[0] if ranking else "unknown"

    logger.info("[TRACE END] key=%s | summary=%s", trace_id, summary)

    trace = _active_traces.pop(trace_id, None)
    if not trace:
        return

    try:
        trace.update(output=summary, end_time=_now())
    except Exception as exc:
        logger.warning("Opik trace end failed: %s", exc)


# ── Legacy shim ───────────────────────────────────────────────────────────────

def track_interaction(
    query: str,
    rag_type: str,
    response: str | None = None,
    track_id: str | None = None,
) -> str:
    """Backwards-compatible no-op shim — kept so old call sites don't break."""
    if track_id is None:
        track_id = str(uuid.uuid4())
    return track_id