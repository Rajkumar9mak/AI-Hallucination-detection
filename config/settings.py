"""
Centralized configuration for the RAG Hallucination Detection system.

Loads settings from environment variables (.env file) with sensible defaults.
All configurable parameters are defined here to ensure reproducibility.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEXES_DIR = DATA_DIR / "indexes"
RESULTS_DIR = DATA_DIR / "results"

# Ensure data directories exist on import
for _dir in (DOCUMENTS_DIR, INDEXES_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    method: str = "hybrid"  # "bm25", "dense", "hybrid"
    top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    chunk_size: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))


@dataclass
class LLMConfig:
    """Configuration for LLM answer generation."""
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model_name: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY", None)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))


@dataclass
class VerificationConfig:
    """Configuration for claim verification."""
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    device: str = os.getenv("NLI_DEVICE", "cpu")
    entailment_threshold: float = float(os.getenv("ENTAILMENT_THRESHOLD", "0.7"))
    contradiction_threshold: float = float(os.getenv("CONTRADICTION_THRESHOLD", "0.7"))
    use_llm_verifier: bool = os.getenv("USE_LLM_VERIFIER", "false").lower() == "true"


@dataclass
class CorrectionConfig:
    """Configuration for hallucination correction."""
    strategy: str = os.getenv("CORRECTION_STRATEGY", "hybrid")
    max_re_retrieval_attempts: int = int(os.getenv("MAX_RE_RETRIEVAL_ATTEMPTS", "2"))
    re_retrieval_top_k: int = int(os.getenv("RE_RETRIEVAL_TOP_K", "10"))


@dataclass
class AppConfig:
    """Main application configuration combining all sub-configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)

    # Supported document formats
    supported_formats: tuple = (".pdf", ".txt", ".docx", ".md")

    # Maximum upload size in MB
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


# Singleton-style accessor
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the application configuration (created once)."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


# ---------------------------------------------------------------------------
# Verdict definitions — canonical labels used throughout the project
# ---------------------------------------------------------------------------

class Verdict:
    """Operational verdict labels for claim verification."""
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT = "Insufficient Evidence"

    ALL = (SUPPORTED, CONTRADICTED, INSUFFICIENT)

    COLORS = {
        SUPPORTED: "#4CAF50",       # Green
        CONTRADICTED: "#F44336",    # Red
        INSUFFICIENT: "#FF9800",    # Amber
    }

    ICONS = {
        SUPPORTED: "✅",
        CONTRADICTED: "❌",
        INSUFFICIENT: "⚠️",
    }
