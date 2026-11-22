"""
Document loaders for the RAG Hallucination Detection system.

Supports PDF, plain text, DOCX, and Markdown files.
Each loader extracts text and preserves source metadata (file name, page numbers).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _validate_file(file_path: Path, max_size_mb: int = 50) -> None:
    """Validate that a file exists and is within size limits."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(
            f"File {file_path.name} is {size_mb:.1f} MB, "
            f"exceeding the {max_size_mb} MB limit."
        )


def load_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a PDF file, one entry per page.

    Supports both 'pypdf' and 'PyPDF2' libraries.

    Returns:
        List of dicts with keys: text, page_number, source_name.
    """
    _validate_file(file_path)

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            import importlib
            pypdf2 = importlib.import_module("PyPDF2")
            PdfReader = pypdf2.PdfReader
        except ImportError:
            raise ImportError("Neither 'pypdf' nor 'PyPDF2' is installed. Please install pypdf.")

    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text,
                "page_number": i + 1,
                "source_name": file_path.name,
            })
    if not pages:
        logger.warning("No text extracted from PDF: %s", file_path.name)
    return pages


def load_text(file_path: Path) -> list[dict]:
    """
    Load a plain-text or Markdown file as a single page.

    Returns:
        List with one dict: text, page_number (1), source_name.
    """
    _validate_file(file_path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        logger.warning("Empty text file: %s", file_path.name)
        return []
    return [{
        "text": text,
        "page_number": 1,
        "source_name": file_path.name,
    }]


def load_docx(file_path: Path) -> list[dict]:
    """
    Extract text from a DOCX file.

    DOCX does not have reliable page-number metadata, so the full
    document is returned as page 1.

    Returns:
        List with one dict: text, page_number (1), source_name.
    """
    _validate_file(file_path)

    try:
        import importlib
        docx_mod = importlib.import_module("docx")
        Document = docx_mod.Document
    except ImportError:
        raise ImportError("python-docx is required to read .docx files. Please install python-docx.")

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    if not text.strip():
        logger.warning("No text extracted from DOCX: %s", file_path.name)
        return []
    return [{
        "text": text,
        "page_number": 1,
        "source_name": file_path.name,
    }]


# Mapping from file extension to loader function
_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".docx": load_docx,
}


def load_document(file_path: Path, max_size_mb: int = 50) -> list[dict]:
    """
    Load a document and return its pages/sections with metadata.

    Args:
        file_path: Path to the document file.
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        List of dicts, each with keys: text, page_number, source_name.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exceeds size limits or has an unsupported format.
    """
    file_path = Path(file_path)
    _validate_file(file_path, max_size_mb)

    ext = file_path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: {', '.join(_LOADERS.keys())}"
        )

    logger.info("Loading document: %s (format: %s)", file_path.name, ext)
    pages = loader(file_path)
    logger.info(
        "Extracted %d page(s) from %s", len(pages), file_path.name
    )
    return pages


def load_directory(
    dir_path: Path,
    supported_formats: Optional[tuple[str, ...]] = None,
    max_size_mb: int = 50,
) -> list[dict]:
    """
    Load all supported documents from a directory.

    Args:
        dir_path: Path to the directory containing documents.
        supported_formats: Tuple of file extensions to include.
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        Combined list of page dicts from all documents.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    if supported_formats is None:
        supported_formats = tuple(_LOADERS.keys())

    all_pages: list[dict] = []
    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_formats
    )

    if not files:
        logger.warning("No supported documents found in %s", dir_path)
        return all_pages

    for file_path in files:
        try:
            pages = load_document(file_path, max_size_mb)
            all_pages.extend(pages)
        except Exception:
            logger.exception("Failed to load %s", file_path.name)

    logger.info(
        "Loaded %d page(s) from %d file(s) in %s",
        len(all_pages), len(files), dir_path,
    )
    return all_pages
