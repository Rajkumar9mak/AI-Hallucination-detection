"""
Verdict mapper for the RAG system.

Maps raw NLI labels (entailment / contradiction / neutral) to the
project's three operational verdicts:
    - Supported
    - Contradicted
    - Insufficient Evidence
"""

import logging
import time
from typing import Optional

from config.settings import Verdict

logger = logging.getLogger(__name__)


def map_nli_to_verdict(
    nli_label: str,
    entailment_score: Optional[float] = None,
    contradiction_score: Optional[float] = None,
    entailment_threshold: float = 0.7,
    contradiction_threshold: float = 0.7,
) -> str:
    """
    Map an NLI label to an operational verdict.

    If confidence scores are provided, the thresholds are checked
    to avoid classifying with low-confidence predictions.

    Args:
        nli_label: Raw NLI label ("entailment", "contradiction", "neutral").
        entailment_score: Confidence score for entailment (optional).
        contradiction_score: Confidence score for contradiction (optional).
        entailment_threshold: Minimum confidence for Supported verdict.
        contradiction_threshold: Minimum confidence for Contradicted verdict.

    Returns:
        One of: Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT.
    """
    nli_label = nli_label.lower().strip()

    if nli_label == "entailment":
        if entailment_score is not None and entailment_score < entailment_threshold:
            return Verdict.INSUFFICIENT
        return Verdict.SUPPORTED

    if nli_label == "contradiction":
        if contradiction_score is not None and contradiction_score < contradiction_threshold:
            return Verdict.INSUFFICIENT
        return Verdict.CONTRADICTED

    # neutral, or anything else
    return Verdict.INSUFFICIENT


def create_claim_verdict(
    claim_data: dict,
    nli_result: dict,
    entailment_threshold: float = 0.7,
    contradiction_threshold: float = 0.7,
    verifier_model: str = "cross-encoder/nli-deberta-v3-base",
) -> dict:
    """
    Create a complete claim verification result.

    Combines claim data with NLI results and maps to operational verdict.

    Args:
        claim_data: Dict with claim_id, claim_text, source_answer.
        nli_result: Dict from verify_claim_against_multiple_evidence().
        entailment_threshold: Threshold for Supported verdict.
        contradiction_threshold: Threshold for Contradicted verdict.
        verifier_model: Identifier of the verifier model used.

    Returns:
        Claim verification result dict matching the architecture data contract.
    """
    verdict = map_nli_to_verdict(
        nli_label=nli_result.get("aggregate_label", "neutral"),
        entailment_score=nli_result.get("entailment_score"),
        contradiction_score=nli_result.get("contradiction_score"),
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )

    # Build evidence list for the verdict
    evidence_list: list[dict] = []
    for ev in nli_result.get("per_evidence_results", []):
        evidence_list.append({
            "chunk_id": ev.get("evidence_chunk_id", ""),
            "source_name": ev.get("source_name", "unknown"),
            "page_number": ev.get("page_number"),
            "text": ev.get("evidence_text", ""),
            "nli_label": ev.get("label", ""),
            "nli_scores": ev.get("scores", {}),
        })

    return {
        "claim_id": claim_data.get("claim_id", ""),
        "claim_text": claim_data.get("claim_text", ""),
        "verdict": verdict,
        "evidence": evidence_list,
        "verifier": verifier_model,
        "confidence": nli_result.get("entailment_score")
        if verdict == Verdict.SUPPORTED
        else nli_result.get("contradiction_score")
        if verdict == Verdict.CONTRADICTED
        else None,
        "aggregate_nli_label": nli_result.get("aggregate_label", ""),
    }


def verify_all_claims(
    claims: list[dict],
    evidence_chunks: list[dict],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    nli_device: str = "cpu",
    entailment_threshold: float = 0.7,
    contradiction_threshold: float = 0.7,
) -> dict:
    """
    Verify all extracted claims against evidence and produce verdicts.

    Args:
        claims: List of claim dicts from claim extraction.
        evidence_chunks: Retrieved evidence chunks.
        nli_model_name: NLI model identifier.
        nli_device: Device for NLI inference.
        entailment_threshold: Threshold for Supported.
        contradiction_threshold: Threshold for Contradicted.

    Returns:
        Dict with keys:
            verdicts (list), summary (dict), latency_ms (float).
    """
    from verification.nli_verifier import verify_claim_against_multiple_evidence

    start = time.time()
    verdicts: list[dict] = []

    for claim in claims:
        nli_result = verify_claim_against_multiple_evidence(
            claim=claim["claim_text"],
            evidence_chunks=evidence_chunks,
            model_name=nli_model_name,
            device=nli_device,
        )

        verdict = create_claim_verdict(
            claim_data=claim,
            nli_result=nli_result,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            verifier_model=nli_model_name,
        )
        verdicts.append(verdict)

    latency_ms = round((time.time() - start) * 1000, 1)

    # Summary counts
    summary = {
        "total_claims": len(verdicts),
        "supported": sum(1 for v in verdicts if v["verdict"] == Verdict.SUPPORTED),
        "contradicted": sum(1 for v in verdicts if v["verdict"] == Verdict.CONTRADICTED),
        "insufficient": sum(1 for v in verdicts if v["verdict"] == Verdict.INSUFFICIENT),
    }
    if summary["total_claims"] > 0:
        summary["grounding_rate"] = round(
            summary["supported"] / summary["total_claims"], 4
        )
        summary["contradiction_rate"] = round(
            summary["contradicted"] / summary["total_claims"], 4
        )
        summary["insufficient_rate"] = round(
            summary["insufficient"] / summary["total_claims"], 4
        )
    else:
        summary["grounding_rate"] = 0.0
        summary["contradiction_rate"] = 0.0
        summary["insufficient_rate"] = 0.0

    logger.info(
        "Verification complete: %d claims — %d supported, %d contradicted, "
        "%d insufficient (%.1fms)",
        summary["total_claims"], summary["supported"],
        summary["contradicted"], summary["insufficient"], latency_ms,
    )

    return {
        "verdicts": verdicts,
        "summary": summary,
        "latency_ms": latency_ms,
    }
