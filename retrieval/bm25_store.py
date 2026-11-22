"""
BM25 lexical retrieval store for the RAG system.

Tokenises and indexes document chunks, then ranks them by BM25 relevance
for a given query. Includes a built-in exact BM25Okapi fallback when
rank-bm25 is not installed.
"""

import math
import logging
import pickle
import re
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


_ABBREVIATIONS: dict[str, list[str]] = {
    "no": ["number", "num"],
    "num": ["number", "no"],
    "number": ["no", "num"],
    "pnr": ["ticket", "booking"],
    "timing": ["time", "departure", "arrival", "schedule"],
    "time": ["timing", "departure", "arrival"],
    "date": ["dt", "schedule"],
    "dt": ["date"],
    "fare": ["amount", "cost", "total", "price"],
    "cost": ["fare", "amount", "price"],
    "price": ["fare", "amount", "cost"],
    "stn": ["station"],
    "exp": ["express"],
    "exam": ["examination", "test"],
}


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer with domain abbreviation expansion."""
    words = re.findall(r"\w+", text.lower())
    expanded = []
    for w in words:
        expanded.append(w)
        if w in _ABBREVIATIONS:
            expanded.extend(_ABBREVIATIONS[w])
    return expanded


class _BuiltinBM25Okapi:
    """Exact BM25Okapi implementation as a zero-dependency fallback."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []

        # Count document frequencies for words
        df: dict[str, int] = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1

        # Calculate IDFs
        negative_idfs = []
        idf_sum = 0
        for word, freq in df.items():
            idf_val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf_val
            idf_sum += idf_val
            if idf_val < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(self.idf) if self.idf else 0
        eps = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        """Compute BM25 scores for a query across all documents."""
        scores = [0.0] * self.corpus_size
        doc_len = self.doc_len
        avgdl = self.avgdl
        k1 = self.k1
        b = self.b

        for q in query:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq > 0:
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * doc_len[i] / avgdl)
                    scores[i] += q_idf * (numerator / denominator)

        return scores


def _get_bm25_class():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError:
        logger.info("rank-bm25 not installed; using built-in exact BM25Okapi fallback.")
        return _BuiltinBM25Okapi


class BM25Store:
    """Lexical retrieval index using BM25."""

    def __init__(self):
        self._bm25 = None
        self._chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a BM25 index from document chunks.

        Args:
            chunks: List of chunk dicts with a ``text`` key.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks")

        self._chunks = list(chunks)
        self._tokenized_corpus = [_tokenize(c["text"]) for c in self._chunks]
        bm25_cls = _get_bm25_class()
        self._bm25 = bm25_cls(self._tokenized_corpus)

        logger.info("BM25 index built with %d documents", len(self._chunks))

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of chunk dicts with an added ``score`` key, sorted by
            descending BM25 score.
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty — returning no results")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: list[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path) -> None:
        """Save the BM25 index and chunks to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(dir_path / "bm25_store.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        logger.info("BM25 store saved to %s", dir_path)

    def load(self, dir_path: Path) -> None:
        """Load a previously saved BM25 index."""
        dir_path = Path(dir_path)
        pkl_path = dir_path / "bm25_store.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"BM25 store not found at {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        bm25_cls = _get_bm25_class()
        self._bm25 = bm25_cls(self._tokenized_corpus)

        logger.info("BM25 store loaded: %d documents", len(self._chunks))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    @property
    def total_documents(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[dict]:
        return list(self._chunks)
