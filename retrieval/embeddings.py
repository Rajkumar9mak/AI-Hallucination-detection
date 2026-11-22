"""
Sentence Transformers embedding wrapper for the RAG system.

Generates dense vector embeddings for document chunks and queries.
Includes a deterministic TF-IDF / hashing embedding fallback when
sentence-transformers is not installed.
"""

import hashlib
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache for the loaded model
_model = None
_model_name = None
_use_fallback = False


class _FallbackEmbedder:
    """Deterministic hashing-based text embedder for offline / fallback environments."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        vecs = []
        for t in texts:
            words = t.lower().split()
            v = np.zeros(self.dim, dtype=np.float32)
            if not words:
                v[0] = 1.0
            else:
                for w in words:
                    h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                    idx = h % self.dim
                    v[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            vecs.append(v)

        arr = np.array(vecs, dtype=np.float32)
        return arr[0] if single else arr


def _get_model(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    """Load and cache the Sentence Transformer model (or fallback)."""
    global _model, _model_name, _use_fallback
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s (device=%s)", model_name, device)
            _model = SentenceTransformer(model_name, device=device)
            _model_name = model_name
            _use_fallback = False
            logger.info("Embedding model loaded. Dimension: %d", _model.get_sentence_embedding_dimension())
        except ImportError:
            logger.info("sentence-transformers not installed; using fallback deterministic embedder.")
            _model = _FallbackEmbedder(dim=384)
            _model_name = "fallback-hash-384"
            _use_fallback = True
    return _model


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> int:
    """Return the embedding dimension for the configured model."""
    model = _get_model(model_name, device)
    return model.get_sentence_embedding_dimension()


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        model_name: Sentence Transformer model identifier.
        device: Device for inference ("cpu" or "cuda").
        batch_size: Encoding batch size.
        show_progress: Whether to show a progress bar.

    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.array([])

    model = _get_model(model_name, device)
    logger.info("Embedding %d text(s) with %s", len(texts), model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate an embedding for a single query string.

    Args:
        query: The query text.
        model_name: Sentence Transformer model identifier.
        device: Device for inference.

    Returns:
        NumPy array of shape (embedding_dim,).
    """
    model = _get_model(model_name, device)
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding


def embed_chunks(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings for a list of chunk dicts.

    Args:
        chunks: List of chunk dicts with a ``text`` key.
        model_name: Sentence Transformer model identifier.
        device: Device for inference.
        batch_size: Encoding batch size.

    Returns:
        NumPy array of shape (len(chunks), embedding_dim).
    """
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(texts, model_name, device, batch_size)
