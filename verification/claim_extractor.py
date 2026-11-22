"""
Atomic claim extraction module for the RAG system.

Decomposes a generated answer into individual atomic factual claims
that can each be independently verified against evidence.
"""

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


_CLAIM_EXTRACTION_PROMPT = """You are a claim extraction system. Your task is to decompose the following answer into individual atomic factual claims.

Rules:
1. Each claim must be a single, independent factual statement.
2. Break compound sentences into separate claims.
3. Preserve the original meaning — do not add information.
4. Skip opinions, questions, and hedging language (e.g., "I think", "probably").
5. Skip meta-statements about the answer itself (e.g., "Based on the evidence...").
6. Number each claim sequentially.
7. If the answer contains no factual claims, respond with "NO_CLAIMS".

ANSWER:
{answer}

Extract each atomic factual claim, one per line, numbered:
1. [first claim]
2. [second claim]
..."""


def extract_claims_with_llm(
    answer: str,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Extract atomic claims from an answer using an LLM (with rule-based fallback).

    Args:
        answer: The generated answer text.
        model_name: LLM model identifier.
        api_key: API key for the LLM provider.

    Returns:
        List of claim dicts with keys: claim_id, claim_text, source_answer.
    """
    if not answer or not answer.strip():
        return []

    if not api_key:
        return _fallback_extraction(answer)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = _CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 1024},
        )
        raw_text = response.text.strip()
    except Exception as e:
        logger.info("LLM claim extraction unavailable (%s); using sentence-level atomic extraction.", e)
        return _fallback_extraction(answer)

    if raw_text.upper() == "NO_CLAIMS":
        return []

    claims = _parse_numbered_claims(raw_text, answer)
    return claims if claims else _fallback_extraction(answer)


def _parse_numbered_claims(raw_text: str, source_answer: str) -> list[dict]:
    """Parse numbered lines from the LLM output into claim dicts."""
    claims: list[dict] = []
    # Match lines starting with a number followed by . or )
    pattern = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+)", re.MULTILINE)

    for match in pattern.finditer(raw_text):
        claim_text = match.group(2).strip()
        if claim_text:
            claims.append({
                "claim_id": f"claim_{len(claims) + 1:03d}",
                "claim_text": claim_text,
                "source_answer": source_answer,
            })

    if not claims:
        # If regex failed, try splitting by newlines
        for line in raw_text.split("\n"):
            line = line.strip()
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            if line and len(line) > 10:
                claims.append({
                    "claim_id": f"claim_{len(claims) + 1:03d}",
                    "claim_text": line,
                    "source_answer": source_answer,
                })

    return claims


def _fallback_extraction(answer: str) -> list[dict]:
    """
    Rule-based fallback: split on sentence and clause boundaries.

    Decomposes sentences into atomic factual statements.
    """
    # Split on sentence terminals
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims: list[dict] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Split compound clauses joined by coordinating conjunctions when clauses are long enough
        clauses = re.split(r";\s*|\s+(?:and|while|but)\s+(?=[A-Z0-9])", sentence)
        for clause in clauses:
            clause = clause.strip().rstrip(".")
            if len(clause) > 12:
                # Add terminal period if missing
                if not clause.endswith((".", "!", "?")):
                    clause += "."
                claims.append({
                    "claim_id": f"claim_{len(claims) + 1:03d}",
                    "claim_text": clause,
                    "source_answer": answer,
                })

    if not claims and answer.strip():
        claims.append({
            "claim_id": "claim_001",
            "claim_text": answer.strip(),
            "source_answer": answer,
        })

    return claims


def extract_claims(
    answer: str,
    method: str = "llm",
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> dict:
    """
    Extract atomic factual claims from an answer.

    Args:
        answer: The generated answer text.
        method: Extraction method — "llm" or "rule_based".
        model_name: LLM model identifier (for LLM method).
        api_key: API key (for LLM method).

    Returns:
        Dict with keys: claims (list), total_claims (int),
        original_answer (str), extraction_method (str), latency_ms (float).
    """
    start = time.time()

    # Strip bracketed tags and page citations before decomposing into atomic claims
    clean_answer = re.sub(r"\[(Supported|Unsupported|Irrelevant)\]", "", answer).strip()
    clean_answer = re.sub(r"\(Page \d+\)\.?", "", clean_answer).strip()

    # If it's an explicit refusal or irrelevance notification, return zero claims
    lower_ans = clean_answer.lower()
    if any(k in lower_ans for k in ["unrelated to the", "doesn't provide enough information", "does not provide enough information", "cannot provide an answer", "no relevant evidence"]):
        return {
            "claims": [],
            "total_claims": 0,
            "original_answer": answer,
            "extraction_method": method,
            "latency_ms": round((time.time() - start) * 1000, 1),
        }

    if method == "llm":
        claims = extract_claims_with_llm(clean_answer, model_name, api_key)
    elif method == "rule_based":
        claims = _fallback_extraction(clean_answer)
    else:
        raise ValueError(f"Unknown extraction method: {method}")

    latency_ms = round((time.time() - start) * 1000, 1)

    result = {
        "claims": claims,
        "total_claims": len(claims),
        "original_answer": answer,
        "extraction_method": method,
        "latency_ms": latency_ms,
    }

    logger.info(
        "Extracted %d claim(s) using %s method in %.1fms",
        len(claims), method, latency_ms,
    )
    return result
