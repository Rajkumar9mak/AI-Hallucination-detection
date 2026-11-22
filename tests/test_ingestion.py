"""Unit tests for document ingestion: loading, cleaning, and chunking."""

import pytest
from pathlib import Path
from ingestion.cleaning import clean_text, clean_pages
from ingestion.chunking import chunk_text, chunk_pages, _generate_chunk_id, _generate_document_id
from ingestion.loaders import load_text, load_document, _LOADERS


def test_clean_text():
    raw = "  Hello   world!  \u00a0\u200b\n\n\n\nLine 2   with   spaces.\r\nLine 3.  "
    cleaned = clean_text(raw)
    assert "\u00a0" not in cleaned
    assert "\u200b" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Line 2 with spaces." in cleaned
    assert cleaned.startswith("Hello world!")


def test_clean_pages():
    pages = [
        {"text": "   Page 1 text  \n\n", "page_number": 1, "source_name": "test.txt"},
        {"text": "   \n\n   ", "page_number": 2, "source_name": "test.txt"},  # empty page
        {"text": "Page 3 content", "page_number": 3, "source_name": "test.txt"},
    ]
    cleaned = clean_pages(pages)
    assert len(cleaned) == 2
    assert cleaned[0]["page_number"] == 1
    assert cleaned[0]["text"] == "Page 1 text"
    assert cleaned[1]["page_number"] == 3
    assert cleaned[1]["text"] == "Page 3 content"


def test_chunk_text_basic_and_metadata():
    text = (
        "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLMs. "
        "It combines retrieval of external knowledge with generative capabilities. "
        "This helps reduce hallucinations in complex domain-specific tasks. "
        "Evaluations show that claim-level verification further improves reliability."
    )
    chunks = chunk_text(
        text=text,
        chunk_size=120,
        chunk_overlap=20,
        source_name="rag_paper.txt",
        page_number=1,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "document_id" in chunk
        assert chunk["source_name"] == "rag_paper.txt"
        assert chunk["page_number"] == 1
        assert "text" in chunk
        assert len(chunk["text"]) <= 120 or chunk["text"].endswith(".")
        assert "metadata" in chunk
        assert "chunk_index" in chunk["metadata"]
        assert "char_start" in chunk["metadata"]
        assert "char_end" in chunk["metadata"]


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=50, chunk_overlap=60)
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=-10, chunk_overlap=0)


def test_chunk_pages():
    pages = [
        {"text": "Short page 1 content.", "page_number": 1, "source_name": "doc1.txt"},
        {"text": "Short page 2 content.", "page_number": 2, "source_name": "doc1.txt"},
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_load_text_file(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello from temporary text file.", encoding="utf-8")
    pages = load_text(sample_file)
    assert len(pages) == 1
    assert pages[0]["text"] == "Hello from temporary text file."
    assert pages[0]["page_number"] == 1
    assert pages[0]["source_name"] == "sample.txt"
