"""
FAISS-based dense vector store for the RAG system.

Maintains a FAISS index alongside a mapping from vector IDs to
the original chunk dicts (text + metadata).
Includes an exact NumPy-based cosine inner-product index fallback
when FAISS is not installed.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyVectorIndex:
    """Exact inner-product vector index matching FAISS IndexFlatIP."""

    def __init__(self, dim: int):
        self.d = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, x: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, x.astype(np.float32)])

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute inner products with all vectors and return top k."""
        if self.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        # query shape: (1, dim), vectors shape: (N, dim)
        scores = np.dot(query, self.vectors.T)[0]  # shape: (N,)
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]
        return np.array([top_scores], dtype=np.float32), np.array([top_indices], dtype=np.int64)


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class FAISSStore:
    """Dense vector index backed by FAISS (with NumPy fallback)."""

    def __init__(self):
        self._index = None
        self._chunks: list[dict] = []          # ordered by FAISS vector ID
        self._embedding_dim: Optional[int] = None
        self._backend: str = "faiss"

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Build a FAISS index from pre-computed embeddings.

        Args:
            chunks: List of chunk dicts (must match the order of embeddings).
            embeddings: NumPy array of shape (n, dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )
        if len(chunks) == 0:
            raise ValueError("Cannot build index from empty chunks")

        self._embedding_dim = embeddings.shape[1]
        self._chunks = list(chunks)

        if _has_faiss():
            import faiss
            self._backend = "faiss"
            self._index = faiss.IndexFlatIP(self._embedding_dim)
            self._index.add(embeddings.astype(np.float32))
        else:
            logger.info("FAISS not available; using NumPy exact inner-product vector store fallback.")
            self._backend = "numpy"
            self._index = _NumpyVectorIndex(self._embedding_dim)
            self._index.add(embeddings.astype(np.float32))

        logger.info(
            "Vector index built (%s backend): %d vectors, dim=%d",
            self._backend, self._index.ntotal, self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-K most similar chunks for a query embedding.

        Args:
            query_embedding: 1-D NumPy array of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of dicts, each containing the chunk fields plus a ``score`` key.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector index is empty — returning no results")
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query_embedding, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Save the vector index and chunk mapping to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self._backend == "faiss" and _has_faiss():
            import faiss
            if self._index is not None:
                faiss.write_index(self._index, str(dir_path / "faiss.index"))
        else:
            # Save vectors for numpy backend
            if self._index is not None:
                np.save(str(dir_path / "vectors.npy"), self._index.vectors)

        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "dim": self._embedding_dim,
                "backend": self._backend,
            }, f)

        logger.info("Vector store saved to %s", dir_path)

    def load(self, dir_path: Path) -> None:
        """Load a previously saved vector index and chunk mapping."""
        dir_path = Path(dir_path)
        chunks_path = dir_path / "chunks.pkl"
        faiss_path = dir_path / "faiss.index"
        npy_path = dir_path / "vectors.npy"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Vector store not found in {dir_path}")

        with open(chunks_path, "rb") as f:
            meta = pickle.load(f)

        if isinstance(meta, list):
            self._chunks = meta
            self._embedding_dim = None
            self._backend = "faiss"
        else:
            self._chunks = meta["chunks"]
            self._embedding_dim = meta.get("dim")
            self._backend = meta.get("backend", "faiss")

        if faiss_path.exists() and _has_faiss():
            import faiss
            self._index = faiss.read_index(str(faiss_path))
            self._embedding_dim = self._index.d
            self._backend = "faiss"
        elif npy_path.exists():
            vectors = np.load(str(npy_path))
            self._embedding_dim = vectors.shape[1]
            self._index = _NumpyVectorIndex(self._embedding_dim)
            self._index.add(vectors)
            self._backend = "numpy"
        else:
            raise FileNotFoundError(f"Neither faiss.index nor vectors.npy found in {dir_path}")

        logger.info(
            "Vector store loaded (%s backend): %d vectors, %d chunks",
            self._backend, self._index.ntotal, len(self._chunks),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def chunks(self) -> list[dict]:
        return list(self._chunks)

    @property
    def backend(self) -> str:
        return self._backend
