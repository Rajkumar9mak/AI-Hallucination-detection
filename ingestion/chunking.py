"""
Text chunking module for the RAG Hallucination Detection system.

Splits cleaned document text into overlapping chunks while preserving
document/page metadata for evidence traceability.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _generate_chunk_id(document_id: str, index: int, page_number: Optional[int] = None) -> str:
    """Generate a deterministic, unique chunk ID from document ID, page, and index."""
    page_str = f"p{page_number}::" if page_number is not None else ""
    raw = f"{document_id}::{page_str}chunk_{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _generate_document_id(source_name: str) -> str:
    """Generate a deterministic document ID from the source file name."""
    return hashlib.sha256(source_name.encode()).hexdigest()[:12]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_name: str = "unknown",
    page_number: Optional[int] = None,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Split a text string into overlapping chunks with metadata.

    Uses a character-level sliding window. Attempts to break at sentence
    boundaries within a tolerance window to avoid cutting mid-sentence.

    Args:
        text: The text to chunk.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        source_name: Original document file name.
        page_number: Page number within the source document.
        document_id: Unique document identifier (auto-generated if None).

    Returns:
        List of chunk dicts with keys:
            chunk_id, document_id, source_name, page_number, text, metadata.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    if document_id is None:
        document_id = _generate_document_id(source_name)

    chunks: list[dict] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (., !, ?) within the last 20%
        if end < len(text):
            search_start = end - int(chunk_size * 0.2)
            best_break = -1
            for i in range(end, max(search_start, start), -1):
                if i < len(text) and text[i - 1] in ".!?\n":
                    best_break = i
                    break
            if best_break > start:
                end = best_break

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunk_id = _generate_chunk_id(document_id, chunk_index, page_number)
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_name": source_name,
                "page_number": page_number,
                "text": chunk_text_content,
                "metadata": {
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": start + len(chunk_text_content),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            })
            chunk_index += 1

        start += step
        # Avoid infinite loop for tiny texts
        if step <= 0:
            break

    logger.debug(
        "Created %d chunk(s) from %s (page %s), chunk_size=%d, overlap=%d",
        len(chunks), source_name, page_number, chunk_size, chunk_overlap,
    )
    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Chunk a list of page dicts (from document loaders/cleaning).

    Each page is chunked independently, preserving per-page metadata.

    Args:
        pages: List of page dicts with keys: text, source_name, page_number.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Combined list of chunk dicts across all pages.
    """
    all_chunks: list[dict] = []

    for page in pages:
        source_name = page.get("source_name", "unknown")
        page_number = page.get("page_number")
        text = page.get("text", "")

        page_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_name=source_name,
            page_number=page_number,
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunked %d page(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(pages), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks
