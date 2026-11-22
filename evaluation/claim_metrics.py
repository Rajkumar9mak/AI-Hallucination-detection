"""
Claim-level evaluation metrics for the RAG system.

Measures claim verification performance including grounding rate,
contradiction rate, insufficient-evidence rate, and verifier P/R/F1.
"""

import logging
from typing import Optional

from config.settings import Verdict

logger = logging.getLogger(__name__)


def grounding_rate(verdicts: list[dict]) -> float:
    """Fraction of claims that are Supported."""
    if not verdicts:
        return 0.0
    supported = sum(1 for v in verdicts if v["verdict"] == Verdict.SUPPORTED)
    return round(supported / len(verdicts), 4)


def contradiction_rate(verdicts: list[dict]) -> float:
    """Fraction of claims that are Contradicted."""
    if not verdicts:
        return 0.0
    contradicted = sum(1 for v in verdicts if v["verdict"] == Verdict.CONTRADICTED)
    return round(contradicted / len(verdicts), 4)


def insufficient_rate(verdicts: list[dict]) -> float:
    """Fraction of claims with Insufficient Evidence."""
    if not verdicts:
        return 0.0
    insuf = sum(1 for v in verdicts if v["verdict"] == Verdict.INSUFFICIENT)
    return round(insuf / len(verdicts), 4)


def verdict_distribution(verdicts: list[dict]) -> dict:
    """Count of each verdict type."""
    dist = {
        Verdict.SUPPORTED: 0,
        Verdict.CONTRADICTED: 0,
        Verdict.INSUFFICIENT: 0,
    }
    for v in verdicts:
        label = v.get("verdict", Verdict.INSUFFICIENT)
        if label in dist:
            dist[label] += 1
    return dist


def verifier_precision_recall_f1(
    predicted_verdicts: list[str],
    ground_truth_labels: list[str],
    positive_label: str = Verdict.SUPPORTED,
) -> dict:
    """
    Calculate precision, recall, and F1 for the verifier.

    Uses a one-vs-rest scheme for the specified positive label.

    Args:
        predicted_verdicts: List of predicted verdict strings.
        ground_truth_labels: Corresponding ground truth labels.
        positive_label: The label to treat as "positive".

    Returns:
        Dict with precision, recall, f1 keys.
    """
    if len(predicted_verdicts) != len(ground_truth_labels):
        raise ValueError("Predicted and ground truth lists must have the same length")

    tp = fp = fn = 0
    for pred, gt in zip(predicted_verdicts, ground_truth_labels):
        if pred == positive_label and gt == positive_label:
            tp += 1
        elif pred == positive_label and gt != positive_label:
            fp += 1
        elif pred != positive_label and gt == positive_label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_f1(
    predicted_verdicts: list[str],
    ground_truth_labels: list[str],
) -> dict:
    """
    Calculate macro-averaged precision, recall, and F1 across all verdict labels.
    """
    per_label: dict = {}
    for label in Verdict.ALL:
        per_label[label] = verifier_precision_recall_f1(
            predicted_verdicts, ground_truth_labels, positive_label=label
        )

    n = len(Verdict.ALL)
    macro = {
        "macro_precision": round(sum(v["precision"] for v in per_label.values()) / n, 4),
        "macro_recall": round(sum(v["recall"] for v in per_label.values()) / n, 4),
        "macro_f1": round(sum(v["f1"] for v in per_label.values()) / n, 4),
        "per_label": per_label,
    }
    return macro


def compare_before_after(
    initial_verdicts: list[dict],
    final_verdicts: list[dict],
) -> dict:
    """
    Compare initial and final verification results.

    Returns:
        Dict with initial/final distributions and deltas.
    """
    initial_dist = verdict_distribution(initial_verdicts)
    final_dist = verdict_distribution(final_verdicts)

    return {
        "initial": initial_dist,
        "final": final_dist,
        "initial_grounding_rate": grounding_rate(initial_verdicts),
        "final_grounding_rate": grounding_rate(final_verdicts),
        "grounding_improvement": round(
            grounding_rate(final_verdicts) - grounding_rate(initial_verdicts), 4
        ),
        "initial_contradiction_rate": contradiction_rate(initial_verdicts),
        "final_contradiction_rate": contradiction_rate(final_verdicts),
        "contradiction_reduction": round(
            contradiction_rate(initial_verdicts) - contradiction_rate(final_verdicts), 4
        ),
    }
