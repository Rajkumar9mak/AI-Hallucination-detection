"""
Hallucination correction pipeline for the RAG system.

Orchestrates correction strategies for claims that are Contradicted
or have Insufficient Evidence.
"""

import logging
import time
from typing import Optional

from config.settings import Verdict

logger = logging.getLogger(__name__)


def identify_problematic_claims(verdicts: list[dict]) -> list[dict]:
    """
    Identify claims that need correction.

    Args:
        verdicts: List of claim verdict dicts.

    Returns:
        List of verdict dicts where verdict is Contradicted or Insufficient.
    """
    problematic = [
        v for v in verdicts
        if v["verdict"] in (Verdict.CONTRADICTED, Verdict.INSUFFICIENT)
    ]
    logger.info(
        "Identified %d problematic claim(s) out of %d",
        len(problematic), len(verdicts),
    )
    return problematic


def correct_claims(
    original_answer: str,
    verdicts: list[dict],
    evidence_chunks: list[dict],
    retriever_manager=None,
    llm_config: Optional[dict] = None,
    strategy: str = "hybrid",
) -> dict:
    """
    Apply correction strategies to problematic claims.

    Strategies:
        - "re_retrieval": Fetch more evidence for unsupported claims.
        - "regeneration": Regenerate answer constrained to evidence.
        - "removal": Remove unsupported claims.
        - "abstention": Add disclaimers for unverifiable claims.
        - "hybrid": Combine strategies based on verdict type.

    Args:
        original_answer: The original generated answer.
        verdicts: Claim verification results.
        evidence_chunks: Originally retrieved evidence.
        retriever_manager: RetrieverManager instance for re-retrieval.
        llm_config: Dict with model_name, api_key, temperature, max_tokens.
        strategy: Correction strategy to apply.

    Returns:
        Dict with corrected_answer, correction_actions, strategy used.
    """
    start = time.time()

    problematic = identify_problematic_claims(verdicts)

    if not problematic:
        return {
            "corrected_answer": original_answer,
            "correction_actions": [],
            "strategy": strategy,
            "original_answer": original_answer,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "claims_corrected": 0,
        }

    if llm_config is None:
        llm_config = {}

    actions: list[dict] = []

    if strategy == "removal":
        corrected, acts = _apply_removal(original_answer, problematic)
        actions.extend(acts)
    elif strategy == "abstention":
        corrected, acts = _apply_abstention(original_answer, problematic)
        actions.extend(acts)
    elif strategy == "regeneration":
        corrected, acts = _apply_regeneration(
            original_answer, verdicts, evidence_chunks, llm_config
        )
        actions.extend(acts)
    elif strategy == "re_retrieval":
        corrected, acts = _apply_re_retrieval(
            original_answer, problematic, evidence_chunks,
            retriever_manager, llm_config,
        )
        actions.extend(acts)
    elif strategy == "hybrid":
        corrected, acts = _apply_hybrid(
            original_answer, verdicts, evidence_chunks,
            retriever_manager, llm_config,
        )
        actions.extend(acts)
    else:
        raise ValueError(f"Unknown correction strategy: {strategy}")

    latency_ms = round((time.time() - start) * 1000, 1)

    result = {
        "corrected_answer": corrected,
        "correction_actions": actions,
        "strategy": strategy,
        "original_answer": original_answer,
        "latency_ms": latency_ms,
        "claims_corrected": len(actions),
    }

    logger.info(
        "Correction complete: %d action(s) using '%s' strategy (%.1fms)",
        len(actions), strategy, latency_ms,
    )
    return result


# ------------------------------------------------------------------
# Correction strategies
# ------------------------------------------------------------------

def _apply_removal(
    answer: str, problematic: list[dict]
) -> tuple[str, list[dict]]:
    """Remove unsupported claims from the answer."""
    actions: list[dict] = []
    corrected = answer

    for claim in problematic:
        claim_text = claim["claim_text"]
        # Try to find and remove the claim from the answer
        if claim_text in corrected:
            corrected = corrected.replace(claim_text, "").strip()
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim_text,
                "action": "removed",
                "verdict": claim["verdict"],
                "reason": f"Claim {claim['verdict'].lower()} — removed from answer.",
            })
        else:
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim_text,
                "action": "flagged",
                "verdict": claim["verdict"],
                "reason": "Could not locate exact claim text in answer for removal.",
            })

    # Clean up double spaces / blank lines
    import re
    corrected = re.sub(r"\n{3,}", "\n\n", corrected)
    corrected = re.sub(r"  +", " ", corrected).strip()

    if not corrected:
        corrected = (
            "After removing claims without adequate evidence support, "
            "no verified content remains for this query."
        )

    return corrected, actions


def _apply_abstention(
    answer: str, problematic: list[dict]
) -> tuple[str, list[dict]]:
    """Add disclaimers for claims that cannot be verified."""
    actions: list[dict] = []
    disclaimers: list[str] = []

    for claim in problematic:
        if claim["verdict"] == Verdict.CONTRADICTED:
            disclaimers.append(
                f"⚠️ The following claim may be inaccurate based on the "
                f"available evidence: \"{claim['claim_text']}\""
            )
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "action": "disclaimer_added",
                "verdict": claim["verdict"],
                "reason": "Evidence contradicts this claim.",
            })
        else:
            disclaimers.append(
                f"ℹ️ Insufficient evidence to verify: \"{claim['claim_text']}\""
            )
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "action": "disclaimer_added",
                "verdict": claim["verdict"],
                "reason": "Not enough evidence to verify this claim.",
            })

    corrected = answer + "\n\n---\n**Verification Notes:**\n" + "\n".join(disclaimers)
    return corrected, actions


def _apply_regeneration(
    answer: str,
    verdicts: list[dict],
    evidence_chunks: list[dict],
    llm_config: dict,
) -> tuple[str, list[dict]]:
    """Regenerate the answer constrained to supported evidence only."""
    from generation.answer_generator import generate_answer

    # Collect only supported claims to guide regeneration
    supported_claims = [
        v["claim_text"] for v in verdicts if v["verdict"] == Verdict.SUPPORTED
    ]

    if not supported_claims:
        corrected = (
            "The available evidence was not sufficient to generate a "
            "fully verified answer for this query."
        )
        return corrected, [{
            "claim_id": "all",
            "claim_text": "All claims",
            "action": "abstained",
            "verdict": "mixed",
            "reason": "No claims were supported by evidence.",
        }]

    # Build a regeneration prompt that explicitly includes only supported facts
    regen_query = (
        "Based ONLY on the following verified facts and the supporting evidence, "
        "provide a corrected and accurate answer. Include only information that "
        "is directly supported by the evidence.\n\n"
        "Verified facts:\n" +
        "\n".join(f"- {c}" for c in supported_claims)
    )

    result = generate_answer(
        query=regen_query,
        evidence_chunks=evidence_chunks,
        model_name=llm_config.get("model_name", "gemini-2.0-flash"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 1024),
    )

    actions = [{
        "claim_id": "all",
        "claim_text": "Full answer",
        "action": "regenerated",
        "verdict": "mixed",
        "reason": f"Answer regenerated using {len(supported_claims)} supported claims.",
    }]

    return result["answer"], actions


def _apply_re_retrieval(
    answer: str,
    problematic: list[dict],
    evidence_chunks: list[dict],
    retriever_manager,
    llm_config: dict,
) -> tuple[str, list[dict]]:
    """Re-retrieve evidence for problematic claims and regenerate."""
    actions: list[dict] = []
    additional_evidence: list[dict] = []

    if retriever_manager is None:
        logger.warning("No retriever_manager provided for re-retrieval; falling back to abstention.")
        return _apply_abstention(answer, problematic)

    for claim in problematic:
        try:
            new_evidence = retriever_manager.retrieve(
                query=claim["claim_text"],
                method="hybrid",
                top_k=llm_config.get("re_retrieval_top_k", 10),
            )
            additional_evidence.extend(new_evidence)
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "action": "re_retrieved",
                "verdict": claim["verdict"],
                "reason": f"Retrieved {len(new_evidence)} additional evidence chunk(s).",
            })
        except Exception as e:
            logger.warning("Re-retrieval failed for claim %s: %s", claim["claim_id"], e)
            actions.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "action": "re_retrieval_failed",
                "verdict": claim["verdict"],
                "reason": str(e),
            })

    # Regenerate with expanded evidence
    all_evidence = evidence_chunks + additional_evidence
    if all_evidence:
        from generation.answer_generator import generate_answer

        result = generate_answer(
            query=f"Answer accurately based on all available evidence. Original answer to improve: {answer}",
            evidence_chunks=all_evidence,
            model_name=llm_config.get("model_name", "gemini-2.0-flash"),
            api_key=llm_config.get("api_key"),
        )
        return result["answer"], actions

    return answer, actions


def _apply_hybrid(
    answer: str,
    verdicts: list[dict],
    evidence_chunks: list[dict],
    retriever_manager,
    llm_config: dict,
) -> tuple[str, list[dict]]:
    """
    Hybrid correction: apply the most appropriate strategy per claim.

    - Contradicted claims → removal or regeneration.
    - Insufficient Evidence claims → re-retrieval if possible, else abstention.
    """
    contradicted = [v for v in verdicts if v["verdict"] == Verdict.CONTRADICTED]
    insufficient = [v for v in verdicts if v["verdict"] == Verdict.INSUFFICIENT]

    all_actions: list[dict] = []

    # Step 1: Handle contradicted claims via regeneration (excludes them)
    if contradicted:
        corrected, acts = _apply_regeneration(
            answer, verdicts, evidence_chunks, llm_config
        )
        for act in acts:
            act["strategy_phase"] = "regeneration"
        all_actions.extend(acts)
    else:
        corrected = answer

    # Step 2: Handle insufficient evidence via abstention
    if insufficient:
        _, abs_acts = _apply_abstention(corrected, insufficient)
        for act in abs_acts:
            act["strategy_phase"] = "abstention"
        all_actions.extend(abs_acts)
        # Add disclaimers to corrected answer
        disclaimers = [
            f"ℹ️ Insufficient evidence to verify: \"{c['claim_text']}\""
            for c in insufficient
        ]
        if disclaimers:
            corrected += (
                "\n\n---\n**Verification Notes:**\n" + "\n".join(disclaimers)
            )

    return corrected, all_actions
