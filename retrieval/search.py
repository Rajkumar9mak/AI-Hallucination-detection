"""
Unified retrieval interface for the RAG system.

Provides a single ``retrieve()`` function that dispatches to BM25,
dense (FAISS), or hybrid retrieval based on configuration.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from retrieval.bm25_store import BM25Store
from retrieval.faiss_store import FAISSStore
from retrieval.embeddings import embed_query, embed_chunks
from retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)


_QUERY_EXPANSIONS: dict[str, str] = {
    "train number": "train no number",
    "train no": "train number no",
    "pnr": "pnr number booking ticket",
    "timing": "timing departure arrival time schedule",
    "exam": "exam examination test",
}


def _expand_query(query: str) -> str:
    """Expand common query phrases with domain synonyms for higher recall."""
    q_lower = query.lower()
    expansions = []
    for key, expanded in _QUERY_EXPANSIONS.items():
        if key in q_lower:
            expansions.append(expanded)
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


class RetrieverManager:
    """
    Manages BM25 and FAISS stores and provides a unified retrieval API.

    Usage:
        manager = RetrieverManager()
        manager.index_chunks(chunks, embedding_config)
        results = manager.retrieve(query, method="hybrid", top_k=5)
    """

    def __init__(self):
        self.bm25_store = BM25Store()
        self.faiss_store = FAISSStore()
        self._embedding_model: str = "all-MiniLM-L6-v2"
        self._embedding_device: str = "cpu"

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        chunks: list[dict],
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> dict:
        """
        Build both BM25 and FAISS indexes from document chunks.

        Args:
            chunks: List of chunk dicts with ``text`` and ``chunk_id`` keys.
            model_name: Sentence Transformer model for dense embeddings.
            device: Device for embedding inference.
            batch_size: Embedding batch size.

        Returns:
            Dict with indexing statistics.
        """
        if not chunks:
            raise ValueError("No chunks to index")

        self._embedding_model = model_name
        self._embedding_device = device

        start = time.time()

        # Build BM25 index
        self.bm25_store.build_index(chunks)

        # Generate embeddings and build FAISS index
        embeddings = embed_chunks(chunks, model_name, device, batch_size)
        self.faiss_store.build_index(chunks, embeddings)

        elapsed = time.time() - start
        stats = {
            "total_chunks": len(chunks),
            "bm25_indexed": self.bm25_store.total_documents,
            "faiss_indexed": self.faiss_store.total_vectors,
            "embedding_model": model_name,
            "indexing_time_seconds": round(elapsed, 2),
        }
        logger.info("Indexing complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question.
            method: Retrieval strategy — "bm25", "dense", or "hybrid".
            top_k: Number of results to return.
            alpha: Hybrid fusion weight (dense vs BM25). Ignored for
                   single-method retrieval.

        Returns:
            List of chunk dicts with ``score`` key, sorted by relevance.
        """
        if not query or not query.strip():
            logger.warning("Empty query — returning no results")
            return []

        start = time.time()

        expanded_query = _expand_query(query)

        if method == "bm25":
            results = self._retrieve_bm25(expanded_query, top_k)
        elif method == "dense":
            results = self._retrieve_dense(expanded_query, top_k)
        elif method == "hybrid":
            results = self._retrieve_hybrid(expanded_query, top_k, alpha)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        elapsed = time.time() - start
        logger.info(
            "Retrieval (%s, top_k=%d): %d results in %.3fs",
            method, top_k, len(results), elapsed,
        )
        return results

    def _retrieve_bm25(self, query: str, top_k: int) -> list[dict]:
        if not self.bm25_store.is_built:
            logger.warning("BM25 index not built — returning empty results")
            return []
        return self.bm25_store.search(query, top_k)

    def _retrieve_dense(self, query: str, top_k: int) -> list[dict]:
        if not self.faiss_store.is_built:
            logger.warning("FAISS index not built — returning empty results")
            return []
        qe = embed_query(query, self._embedding_model, self._embedding_device)
        return self.faiss_store.search(qe, top_k)

    def _retrieve_hybrid(
        self, query: str, top_k: int, alpha: float
    ) -> list[dict]:
        if not self.bm25_store.is_built or not self.faiss_store.is_built:
            logger.warning("One or both indexes not built — falling back")
            if self.bm25_store.is_built:
                return self._retrieve_bm25(query, top_k)
            if self.faiss_store.is_built:
                return self._retrieve_dense(query, top_k)
            return []
        qe = embed_query(query, self._embedding_model, self._embedding_device)
        return hybrid_search(
            query=query,
            query_embedding=qe,
            bm25_store=self.bm25_store,
            faiss_store=self.faiss_store,
            top_k=top_k,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Save both indexes to disk."""
        dir_path = Path(dir_path)
        self.bm25_store.save(dir_path / "bm25")
        self.faiss_store.save(dir_path / "faiss")

    def load(self, dir_path: Path) -> None:
        """Load both indexes from disk."""
        dir_path = Path(dir_path)
        bm25_dir = dir_path / "bm25"
        faiss_dir = dir_path / "faiss"
        if bm25_dir.exists():
            self.bm25_store.load(bm25_dir)
        if faiss_dir.exists():
            self.faiss_store.load(faiss_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        return self.bm25_store.is_built or self.faiss_store.is_built

    @property
    def status(self) -> dict:
        return {
            "bm25_ready": self.bm25_store.is_built,
            "faiss_ready": self.faiss_store.is_built,
            "bm25_documents": self.bm25_store.total_documents,
            "faiss_vectors": self.faiss_store.total_vectors,
            "embedding_model": self._embedding_model,
        }
