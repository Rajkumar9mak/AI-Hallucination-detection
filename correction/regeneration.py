"""
Evidence-constrained regeneration module for the correction pipeline.

Regenerates answer text for specific claims using only the available
evidence, without introducing unsupported information.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_REGEN_PROMPT = """You are correcting a specific claim that was found to be problematic.

ORIGINAL CLAIM: {claim}
VERDICT: {verdict}

AVAILABLE EVIDENCE:
{evidence}

Based ONLY on the evidence above:
1. If the evidence supports a corrected version of the claim, provide it.
2. If the evidence contradicts the claim, state what the evidence actually says.
3. If there is insufficient evidence, state that the claim cannot be verified.

Do NOT invent information. Only use what is in the evidence.

CORRECTED STATEMENT:"""


def regenerate_claim(
    claim_text: str,
    verdict: str,
    evidence_chunks: list[dict],
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> dict:
    """
    Regenerate a corrected statement for a problematic claim.

    Args:
        claim_text: The original claim text.
        verdict: The verdict (Contradicted / Insufficient Evidence).
        evidence_chunks: Available evidence for regeneration.
        model_name: LLM model identifier.
        api_key: API key.

    Returns:
        Dict with corrected_text, original_claim, verdict, evidence_used.
    """
    if not evidence_chunks:
        return {
            "corrected_text": f"[Cannot verify: {claim_text}]",
            "original_claim": claim_text,
            "verdict": verdict,
            "evidence_used": 0,
            "action": "abstained",
        }

    evidence_text = "\n\n".join(
        f"[{i+1}] (Source: {c.get('source_name', '?')}, "
        f"Page: {c.get('page_number', '?')})\n{c.get('text', '')}"
        for i, c in enumerate(evidence_chunks)
    )

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            prompt = _REGEN_PROMPT.format(
                claim=claim_text, verdict=verdict, evidence=evidence_text
            )
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 256},
            )
            corrected = response.text.strip()
            return {
                "corrected_text": corrected,
                "original_claim": claim_text,
                "verdict": verdict,
                "evidence_used": len(evidence_chunks),
                "action": "regenerated",
            }
        except Exception as e:
            logger.info("LLM regeneration failed (%s); using evidence-constrained fallback.", e)

    # Deterministic fallback when LLM is unavailable
    if verdict == "Contradicted":
        corrected = f"Available evidence does not substantiate '{claim_text}'."
    else:
        corrected = f"Insufficient evidence to confirm '{claim_text}'."

    return {
        "corrected_text": corrected,
        "original_claim": claim_text,
        "verdict": verdict,
        "evidence_used": len(evidence_chunks),
        "action": "corrected_extractive",
    }
