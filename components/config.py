"""
config.py — Unified LLM + Embedding configuration.

All RAG pipelines import from here so model/embedding changes
are made in ONE place.
"""

import os
import time
import logging
from dotenv import load_dotenv
import litellm
from litellm.exceptions import RateLimitError

load_dotenv()

logger = logging.getLogger("rag_config")

# ── API Keys ────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

os.environ.setdefault("GROQ_API_KEY",   GROQ_API_KEY)
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# ── Available Models ─────────────────────────────────────────────────────────
# Format: { display_name: litellm_model_string }
# NOTE on Groq TPM limits (free tier):
#   llama-3.3-70b-versatile : 12,000 TPM  (slower, hits limits with 3 pipelines)
#   llama-3.1-8b-instant    : 30,000 TPM  (recommended for side-by-side comparisons)
#   qwen-qwq-32b            : 6,000 TPM
#   Gemini 2.5 Flash        : no strict TPM limit on free tier
AVAILABLE_MODELS: dict[str, str] = {
    "Llama 3.1 8B (Groq)": "groq/llama-3.1-8b-instant",
    "Llama 3.3 70B (Groq)":              "groq/llama-3.3-70b-versatile",
    "Qwen QWQ 32B (Groq)":              "groq/qwen-qwq-32b",
    "Gemini 2.5 Flash (Google)":        "gemini/gemini-2.5-flash",
}

DEFAULT_MODEL = "Llama 3.1 8B (Groq)"

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Vector Store ─────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "./vector_store_db"

# ── Retry config ─────────────────────────────────────────────────────────────
_MAX_RETRIES   = 4          # total attempts
_RETRY_BACKOFF = (1, 2, 4)  # wait seconds between retries


# ── LLM Wrapper ──────────────────────────────────────────────────────────────
def get_llm_response(
    messages: list[dict],
    model_display_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Unified LLM call via LiteLLM with automatic rate-limit retry + backoff.

    Args:
        messages:            OpenAI-style list of {role, content} dicts.
        model_display_name:  Key from AVAILABLE_MODELS.
        temperature:         Sampling temperature.
        max_tokens:          Maximum tokens in the completion (default 1024
                             to stay within Groq free-tier TPM budgets).

    Returns:
        The assistant message content as a string.

    Raises:
        RateLimitError: if all retries are exhausted.
    """
    # Resolve display name → litellm model string
    model_id = AVAILABLE_MODELS.get(
        model_display_name,
        AVAILABLE_MODELS[DEFAULT_MODEL],
    )

    kwargs: dict = {
        "model":       model_id,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if model_id.startswith("gemini/"):
        kwargs["api_key"] = GEMINI_API_KEY

    last_exc: Exception | None = None
    for attempt, wait in enumerate((*_RETRY_BACKOFF, None), start=1):
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content.strip()

        except RateLimitError as exc:
            last_exc = exc
            if wait is None:
                break  # exhausted all retries
            logger.warning(
                "Rate limit hit on attempt %d/%d — retrying in %ds… (%s)",
                attempt, _MAX_RETRIES, wait, exc,
            )
            time.sleep(wait)

        except Exception as exc:
            # Non-rate-limit errors: fail immediately
            raise exc

    raise last_exc  # type: ignore[misc]
