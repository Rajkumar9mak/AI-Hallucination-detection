"""
Targeted re-retrieval module for the correction pipeline.

Retrieves additional evidence specifically for problematic claims,
using the claim text as a focused query.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def re_retrieve_for_claim(
    claim_text: str,
    retriever_manager,
    method: str = "hybrid",
    top_k: int = 10,
    alpha: float = 0.5,
) -> list[dict]:
    """
    Perform targeted re-retrieval for a specific claim.

    Uses the claim text as a query to find additional supporting
    or contradicting evidence.

    Args:
        claim_text: The claim to find evidence for.
        retriever_manager: Initialised RetrieverManager.
        method: Retrieval method.
        top_k: Number of results.
        alpha: Hybrid fusion weight.

    Returns:
        List of chunk dicts with relevance scores.
    """
    if retriever_manager is None:
        logger.warning("No retriever available for re-retrieval")
        return []

    if not claim_text or not claim_text.strip():
        return []

    try:
        results = retriever_manager.retrieve(
            query=claim_text,
            method=method,
            top_k=top_k,
            alpha=alpha,
        )
        logger.info(
            "Re-retrieval for claim returned %d chunks (method=%s)",
            len(results), method,
        )
        return results
    except Exception as e:
        logger.exception("Re-retrieval failed: %s", e)
        return []
