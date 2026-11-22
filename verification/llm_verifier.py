"""
Optional LLM-based claim verifier for the RAG system.

Provides a complementary verification approach using an LLM as judge,
in addition to the primary NLI-based verifier.
"""

import logging
import re
import time
from typing import Optional

from config.settings import Verdict

logger = logging.getLogger(__name__)


_LLM_VERIFY_PROMPT = """You are a claim verification system. Determine whether the following claim is supported, contradicted, or has insufficient evidence based ONLY on the provided evidence.

CLAIM: {claim}

EVIDENCE:
{evidence}

Classify the claim as one of:
- SUPPORTED: The evidence clearly supports this claim.
- CONTRADICTED: The evidence directly conflicts with this claim.
- INSUFFICIENT EVIDENCE: The evidence does not provide enough information to support or refute the claim. This does NOT mean the claim is false.

Respond with EXACTLY one of these three labels on the first line, followed by a brief explanation.

VERDICT:"""


def verify_claim_with_llm(
    claim: str,
    evidence_chunks: list[dict],
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> dict:
    """
    Verify a claim using an LLM as judge.

    Args:
        claim: The atomic factual claim.
        evidence_chunks: Retrieved evidence chunks.
        model_name: LLM model identifier.
        api_key: API key for the LLM provider.

    Returns:
        Dict with verdict, explanation, claim, evidence, model.
    """
    if not evidence_chunks:
        return {
            "verdict": Verdict.INSUFFICIENT,
            "explanation": "No evidence was available for verification.",
            "claim": claim,
            "evidence": [],
            "model": model_name,
        }

    # Format evidence
    evidence_text = "\n\n".join(
        f"[{i+1}] (Source: {c.get('source_name', '?')}, "
        f"Page: {c.get('page_number', '?')})\n{c.get('text', '')}"
        for i, c in enumerate(evidence_chunks)
    )

    if not api_key:
        return {
            "verdict": Verdict.INSUFFICIENT,
            "explanation": "No LLM API key provided; falling back to Insufficient Evidence.",
            "claim": claim,
            "evidence": evidence_chunks,
            "model": "no-api-key",
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = _LLM_VERIFY_PROMPT.format(claim=claim, evidence=evidence_text)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 256},
        )
        raw = response.text.strip()
    except Exception as e:
        logger.warning("LLM verification failed: %s", e)
        return {
            "verdict": Verdict.INSUFFICIENT,
            "explanation": f"Verification error: {e}",
            "claim": claim,
            "evidence": evidence_chunks,
            "model": model_name,
        }

    verdict, explanation = _parse_llm_verdict(raw)

    return {
        "verdict": verdict,
        "explanation": explanation,
        "claim": claim,
        "evidence": evidence_chunks,
        "model": model_name,
        "raw_response": raw,
    }


def _parse_llm_verdict(raw: str) -> tuple[str, str]:
    """Parse the LLM's verdict response into (verdict_label, explanation)."""
    lines = raw.strip().split("\n", 1)
    first_line = lines[0].strip().upper()
    explanation = lines[1].strip() if len(lines) > 1 else ""

    if "SUPPORTED" in first_line and "INSUFFICIENT" not in first_line:
        return Verdict.SUPPORTED, explanation
    if "CONTRADICTED" in first_line or "CONTRADICT" in first_line:
        return Verdict.CONTRADICTED, explanation
    if "INSUFFICIENT" in first_line:
        return Verdict.INSUFFICIENT, explanation

    logger.warning("Could not parse LLM verdict from: %s", first_line)
    return Verdict.INSUFFICIENT, raw
