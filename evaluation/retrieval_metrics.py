"""
Retrieval quality metrics for the RAG system evaluation framework.

Measures retrieval performance including Recall@K, Precision@K,
Mean Reciprocal Rank (MRR), and context relevance metrics.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff (defaults to len(retrieved_ids)).

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Cutoff (defaults to len(retrieved_ids)).

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.

    Returns:
        1 / (rank of first relevant document), or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate NDCG@K (Normalised Discounted Cumulative Gain).

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff.

    Returns:
        NDCG@K score (0.0 to 1.0).
    """
    import math

    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not relevant_ids or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids, start=1)
        if doc_id in relevant_set
    )

    # Ideal DCG
    n_relevant_in_top = min(len(relevant_set), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant_in_top + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved_chunks: list[dict],
    relevant_ids: list[str],
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Compute a full set of retrieval metrics.

    Args:
        retrieved_chunks: Retrieved chunk dicts (must have ``chunk_id``).
        relevant_ids: Ground-truth relevant chunk IDs.
        k_values: List of K values to evaluate (default [1, 3, 5, 10]).

    Returns:
        Dict of metric name → value.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

    metrics: dict = {}
    for k in k_values:
        metrics[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)

    metrics["mrr"] = round(mean_reciprocal_rank(retrieved_ids, relevant_ids), 4)

    return metrics
