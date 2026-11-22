"""
Hybrid retrieval module combining BM25 (lexical) and dense (FAISS) results.

Uses Reciprocal Rank Fusion (RRF) as the default fusion strategy.
The fusion weight (alpha) is configurable for experiments.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    alpha: float = 0.5,
    k_constant: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion.

    RRF score for each document:
        score = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_bm25))

    Args:
        bm25_results: Ranked results from BM25 (each dict has ``chunk_id``).
        dense_results: Ranked results from dense retrieval.
        alpha: Weight for dense retrieval (0 = BM25 only, 1 = dense only).
        k_constant: RRF smoothing constant (default 60, standard choice).
        top_k: Number of fused results to return.

    Returns:
        List of chunk dicts sorted by descending fused score, with
        ``score``, ``bm25_rank``, ``dense_rank``, and ``fusion_method`` keys.
    """
    # Build a combined chunk lookup keyed by chunk_id
    all_chunks: dict[str, dict] = {}
    bm25_ranks: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}

    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        bm25_ranks[cid] = rank
        if cid not in all_chunks:
            all_chunks[cid] = dict(chunk)

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        dense_ranks[cid] = rank
        if cid not in all_chunks:
            all_chunks[cid] = dict(chunk)

    # Calculate RRF scores
    scored: list[tuple[str, float]] = []
    for cid in all_chunks:
        rrf_dense = alpha * (1.0 / (k_constant + dense_ranks.get(cid, len(dense_results) + 1)))
        rrf_bm25 = (1 - alpha) * (1.0 / (k_constant + bm25_ranks.get(cid, len(bm25_results) + 1)))
        scored.append((cid, rrf_dense + rrf_bm25))

    # Sort by fused score descending and take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    results: list[dict] = []
    for cid, fused_score in scored[:top_k]:
        chunk = all_chunks[cid]
        chunk["score"] = fused_score
        chunk["bm25_rank"] = bm25_ranks.get(cid)
        chunk["dense_rank"] = dense_ranks.get(cid)
        chunk["fusion_method"] = "reciprocal_rank_fusion"
        results.append(chunk)

    return results


def hybrid_search(
    query: str,
    query_embedding,
    bm25_store,
    faiss_store,
    top_k: int = 5,
    alpha: float = 0.5,
    retrieval_k: Optional[int] = None,
) -> list[dict]:
    """
    Perform hybrid retrieval combining BM25 and dense search.

    Retrieves a larger set from each backend, then fuses with RRF.

    Args:
        query: The search query string.
        query_embedding: Pre-computed query embedding (NumPy array).
        bm25_store: A built BM25Store instance.
        faiss_store: A built FAISSStore instance.
        top_k: Number of final fused results.
        alpha: Weight for dense vs BM25 (0 = BM25-only, 1 = dense-only).
        retrieval_k: Per-backend retrieval depth (defaults to 2 × top_k).

    Returns:
        List of fused chunk dicts sorted by descending score.
    """
    if retrieval_k is None:
        retrieval_k = max(top_k * 2, 20)

    bm25_results = bm25_store.search(query, top_k=retrieval_k)
    dense_results = faiss_store.search(query_embedding, top_k=retrieval_k)

    logger.info(
        "Hybrid search: BM25 returned %d, Dense returned %d (alpha=%.2f)",
        len(bm25_results), len(dense_results), alpha,
    )

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        alpha=alpha,
        top_k=top_k,
    )

    logger.info("Hybrid fusion returned %d results", len(fused))
    return fused
