"""Unit tests for hallucination correction pipeline."""

import pytest
from config.settings import Verdict
from correction.correction_pipeline import (
    identify_problematic_claims,
    _apply_removal,
    _apply_abstention,
)


def test_identify_problematic_claims():
    verdicts = [
        {"claim_id": "c1", "verdict": Verdict.SUPPORTED},
        {"claim_id": "c2", "verdict": Verdict.CONTRADICTED},
        {"claim_id": "c3", "verdict": Verdict.INSUFFICIENT},
    ]
    prob = identify_problematic_claims(verdicts)
    assert len(prob) == 2
    assert prob[0]["claim_id"] == "c2"
    assert prob[1]["claim_id"] == "c3"


def test_apply_removal_strategy():
    answer = "Python was invented in 1991. Guido was born in Canada. Python is easy to learn."
    problematic = [
        {"claim_id": "c2", "claim_text": "Guido was born in Canada.", "verdict": Verdict.CONTRADICTED}
    ]
    corrected, actions = _apply_removal(answer, problematic)
    assert "Guido was born in Canada." not in corrected
    assert "Python was invented in 1991." in corrected
    assert len(actions) == 1
    assert actions[0]["action"] == "removed"


def test_apply_abstention_strategy():
    answer = "The moon orbit is 27 days. The core temperature is 50000 degrees."
    problematic = [
        {"claim_id": "c2", "claim_text": "The core temperature is 50000 degrees.", "verdict": Verdict.INSUFFICIENT}
    ]
    corrected, actions = _apply_abstention(answer, problematic)
    assert "Insufficient evidence to verify" in corrected
    assert len(actions) == 1
    assert actions[0]["action"] == "disclaimer_added"
