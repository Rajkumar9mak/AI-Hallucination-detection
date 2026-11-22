"""
NLI-based claim verification module for the RAG system.

Uses a cross-encoder NLI model to estimate entailment, contradiction,
or neutrality between a claim and its evidence.
Includes a heuristic lexical overlap fallback when sentence-transformers
or transformers models are not locally installed.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level model cache
_nli_model = None
_nli_model_name = None
_use_nli_fallback = False


class _FallbackNLIModel:
    """Heuristic NLI model when cross-encoder is not installed or offline."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        results = []
        for premise, hypothesis in pairs:
            import re
            premise_words = set(re.findall(r"\w+", premise.lower()))
            hypo_words = [w for w in re.findall(r"\w+", hypothesis.lower()) if len(w) > 1]

            if not hypo_words:
                results.append([0.1, 0.1, 0.8])  # mostly neutral
                continue

            matches = sum(1 for w in hypo_words if w in premise_words)
            overlap = matches / len(hypo_words)

            # Check negation or contradiction indicators (avoid treating 'no.' abbreviation as negation)
            negations = {"not", "never", "cannot", "neither", "none", "incorrect", "false"}
            premise_has_neg = bool(premise_words & negations)
            hypo_has_neg = bool(set(hypo_words) & negations)

            if premise_has_neg != hypo_has_neg and overlap > 0.5:
                # Contradiction signal
                results.append([0.85, 0.05, 0.10])
            elif overlap >= 0.7:
                # Strong entailment
                results.append([0.05, 0.90, 0.05])
            elif overlap >= 0.4:
                # Moderate entailment
                results.append([0.10, 0.70, 0.20])
            else:
                # Neutral / insufficient evidence
                results.append([0.10, 0.20, 0.70])

        return results


def _get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "cpu"):
    """Load and cache the NLI cross-encoder model (or heuristic fallback)."""
    global _nli_model, _nli_model_name, _use_nli_fallback
    if _nli_model is None or _nli_model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading NLI model: %s (device=%s)", model_name, device)
            _nli_model = CrossEncoder(model_name, device=device)
            _nli_model_name = model_name
            _use_nli_fallback = False
            logger.info("NLI model loaded: %s", model_name)
        except Exception:
            logger.info("CrossEncoder unavailable; using heuristic NLI verifier fallback.")
            _nli_model = _FallbackNLIModel()
            _nli_model_name = "heuristic-nli-fallback"
            _use_nli_fallback = True
    return _nli_model


# Standard NLI label ordering for cross-encoder/nli-deberta-v3-base:
# [contradiction, entailment, neutral]
_NLI_LABELS = ["contradiction", "entailment", "neutral"]


def verify_claim_against_evidence(
    claim: str,
    evidence_text: str,
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
) -> dict:
    """
    Verify a single claim against a single evidence passage using NLI.

    Args:
        claim: The atomic factual claim to verify.
        evidence_text: The evidence passage to check against.
        model_name: NLI model identifier.
        device: Device for inference.

    Returns:
        Dict with keys: label, scores, claim, evidence.
            label: "entailment", "contradiction", or "neutral"
            scores: dict of {label: probability}
    """
    model = _get_nli_model(model_name, device)

    # NLI format: (premise, hypothesis) = (evidence, claim)
    scores = model.predict([(evidence_text, claim)])[0]

    # Map scores to labels
    score_dict = {
        label: float(score)
        for label, score in zip(_NLI_LABELS, scores)
    }
    predicted_label = max(score_dict, key=score_dict.get)

    return {
        "label": predicted_label,
        "scores": score_dict,
        "claim": claim,
        "evidence": evidence_text,
    }


def verify_claim_against_multiple_evidence(
    claim: str,
    evidence_chunks: list[dict],
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    device: str = "cpu",
) -> dict:
    """
    Verify a claim against multiple evidence chunks and aggregate results.

    Strategy:
        - Run NLI against each evidence chunk.
        - If ANY chunk entails the claim → supported signal.
        - If ANY chunk contradicts the claim → contradiction signal.
        - Use the strongest signal as the aggregate result.

    Args:
        claim: The atomic factual claim.
        evidence_chunks: List of chunk dicts with ``text`` key.
        model_name: NLI model identifier.
        device: Device for inference.

    Returns:
        Dict with keys:
            aggregate_label, per_evidence_results, claim,
            best_supporting_evidence, best_contradicting_evidence.
    """
    if not evidence_chunks:
        return {
            "aggregate_label": "neutral",
            "per_evidence_results": [],
            "claim": claim,
            "best_supporting_evidence": None,
            "best_contradicting_evidence": None,
        }

    model = _get_nli_model(model_name, device)

    # Batch NLI inference
    pairs = [(chunk["text"], claim) for chunk in evidence_chunks]
    all_scores = model.predict(pairs)

    per_evidence: list[dict] = []
    best_entailment_score = -1.0
    best_contradiction_score = -1.0
    best_supporting = None
    best_contradicting = None

    for i, (scores, chunk) in enumerate(zip(all_scores, evidence_chunks)):
        score_dict = {
            label: float(score) for label, score in zip(_NLI_LABELS, scores)
        }
        predicted = max(score_dict, key=score_dict.get)

        result_entry = {
            "evidence_chunk_id": chunk.get("chunk_id", f"unk_{i}"),
            "source_name": chunk.get("source_name", "unknown"),
            "page_number": chunk.get("page_number"),
            "evidence_text": chunk["text"],
            "label": predicted,
            "scores": score_dict,
        }
        per_evidence.append(result_entry)

        if score_dict["entailment"] > best_entailment_score:
            best_entailment_score = score_dict["entailment"]
            best_supporting = result_entry

        if score_dict["contradiction"] > best_contradiction_score:
            best_contradiction_score = score_dict["contradiction"]
            best_contradicting = result_entry

    # Aggregate: strongest signal wins
    if best_entailment_score > best_contradiction_score:
        aggregate = "entailment" if best_entailment_score > 0.5 else "neutral"
    elif best_contradiction_score > best_entailment_score:
        aggregate = "contradiction" if best_contradiction_score > 0.5 else "neutral"
    else:
        aggregate = "neutral"

    return {
        "aggregate_label": aggregate,
        "per_evidence_results": per_evidence,
        "claim": claim,
        "best_supporting_evidence": best_supporting,
        "best_contradicting_evidence": best_contradicting,
        "entailment_score": best_entailment_score,
        "contradiction_score": best_contradiction_score,
    }
