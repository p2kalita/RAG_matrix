"""
guardrails.py — Pure-Python response validation (replaces guardrails-ai package).

Checks applied:
1. Citation presence  — response should mention page/source references
2. Hedging detection  — flags excessive uncertainty language
3. Minimum length     — response should be substantive (≥ 30 words)

Returns the original response text always (no blocking), but prepends a
warning banner if quality checks fail.
"""

import re

# Words/phrases that indicate unhelpful hedging
_HEDGING_PATTERNS = re.compile(
    r"\b(I don't know|I cannot|I'm not sure|unclear|uncertain|"
    r"might be|could be|possibly|perhaps|I think|I believe|"
    r"hard to say|difficult to determine)\b",
    flags=re.IGNORECASE,
)

# Citation markers we expect to see in a well-formed RAG response
_CITATION_PATTERN = re.compile(
    r"(page\s*\d+|chunk|source|reference|\[[\d,\s]+\]|p\.\s*\d+)",
    flags=re.IGNORECASE,
)

MIN_WORDS = 30


def apply_guardrails(response: str) -> str:
    """
    Validate a RAG response and return it with optional warning prefix.

    Args:
        response: Raw LLM response string.

    Returns:
        The (possibly prefixed) response string.
    """
    warnings: list[str] = []

    # 1. Length check
    word_count = len(response.split())
    if word_count < MIN_WORDS:
        warnings.append(f"⚠️ Response is very short ({word_count} words).")

    # 2. Citation check
    if not _CITATION_PATTERN.search(response):
        warnings.append("⚠️ No source citations detected in response.")

    # 3. Excessive hedging check
    hedges = _HEDGING_PATTERNS.findall(response)
    if len(hedges) >= 3:
        unique = list(dict.fromkeys(h.lower() for h in hedges))
        warnings.append(
            f"⚠️ Response contains excessive hedging language: {', '.join(unique)}."
        )

    if warnings:
        banner = "\n".join(warnings)
        return f"{banner}\n\n---\n\n{response}"

    return response