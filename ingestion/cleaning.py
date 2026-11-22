"""
Text cleaning module for the RAG Hallucination Detection system.

Applies normalisation and cleaning to extracted document text
without removing meaningful content.
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving meaningful content.

    Steps:
        1. Replace non-breaking and special whitespace with regular spaces.
        2. Normalise line endings.
        3. Collapse multiple blank lines to a single blank line.
        4. Collapse multiple spaces to a single space (preserving newlines).
        5. Strip leading/trailing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace non-breaking spaces and other Unicode whitespace with regular space
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\ufeff", "")   # BOM
    text = text.replace("\t", " ")

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of blank lines to at most one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces on the same line
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces at the start/end of each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Final outer strip
    text = text.strip()

    return text


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Apply text cleaning to a list of page dicts (from document loaders).

    Removes pages whose text becomes empty after cleaning.

    Args:
        pages: List of dicts with at least a ``text`` key.

    Returns:
        List of page dicts with cleaned ``text``.
    """
    cleaned: list[dict] = []
    for page in pages:
        cleaned_text = clean_text(page.get("text", ""))
        if cleaned_text:
            cleaned.append({**page, "text": cleaned_text})
        else:
            logger.debug(
                "Page %s from %s is empty after cleaning — skipped.",
                page.get("page_number", "?"),
                page.get("source_name", "unknown"),
            )
    return cleaned
