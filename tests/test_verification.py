"""Unit tests for claim extraction and verification verdict mapping."""

import pytest
from config.settings import Verdict
from verification.claim_extractor import _parse_numbered_claims, _fallback_extraction
from verification.verdict_mapper import map_nli_to_verdict, create_claim_verdict


def test_parse_numbered_claims():
    raw_llm_output = (
        "1. Guido van Rossum created Python in 1991.\n"
        "2. Python is dynamically typed.\n"
        "3. It supports object-oriented programming.\n"
    )
    source_ans = "Guido van Rossum created Python in 1991. Python is dynamically typed and supports OOP."
    claims = _parse_numbered_claims(raw_llm_output, source_ans)
    assert len(claims) == 3
    assert claims[0]["claim_id"] == "claim_001"
    assert claims[0]["claim_text"] == "Guido van Rossum created Python in 1991."
    assert claims[1]["claim_id"] == "claim_002"
    assert claims[2]["claim_id"] == "claim_003"


def test_fallback_claim_extraction():
    ans = "Paris is the capital of France. It has a population of over two million people."
    claims = _fallback_extraction(ans)
    assert len(claims) >= 2
    assert "Paris is the capital of France." in [c["claim_text"] for c in claims]


def test_map_nli_to_verdict():
    # Entailment above threshold -> Supported
    assert map_nli_to_verdict("entailment", entailment_score=0.85, entailment_threshold=0.7) == Verdict.SUPPORTED

    # Entailment below threshold -> Insufficient Evidence
    assert map_nli_to_verdict("entailment", entailment_score=0.60, entailment_threshold=0.7) == Verdict.INSUFFICIENT

    # Contradiction above threshold -> Contradicted
    assert map_nli_to_verdict("contradiction", contradiction_score=0.90, contradiction_threshold=0.7) == Verdict.CONTRADICTED

    # Contradiction below threshold -> Insufficient Evidence
    assert map_nli_to_verdict("contradiction", contradiction_score=0.55, contradiction_threshold=0.7) == Verdict.INSUFFICIENT

    # Neutral -> Insufficient Evidence (CRITICAL: never treat neutral as false/contradicted)
    assert map_nli_to_verdict("neutral") == Verdict.INSUFFICIENT


def test_create_claim_verdict():
    claim_data = {
        "claim_id": "claim_001",
        "claim_text": "Python was created in 1991.",
        "source_answer": "Python was created in 1991.",
    }
    nli_result = {
        "aggregate_label": "entailment",
        "entailment_score": 0.95,
        "contradiction_score": 0.02,
        "per_evidence_results": [
            {
                "evidence_chunk_id": "chk_1",
                "source_name": "python_history.pdf",
                "page_number": 1,
                "evidence_text": "Python was released in 1991 by Guido van Rossum.",
                "label": "entailment",
                "scores": {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03},
            }
        ],
    }

    result = create_claim_verdict(claim_data, nli_result, entailment_threshold=0.7)
    assert result["verdict"] == Verdict.SUPPORTED
    assert result["confidence"] == 0.95
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["source_name"] == "python_history.pdf"
