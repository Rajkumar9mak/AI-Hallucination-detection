"""Unit tests for evaluation metrics (retrieval metrics and claim metrics)."""

import pytest
from config.settings import Verdict
from evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval,
)
from evaluation.claim_metrics import (
    grounding_rate,
    contradiction_rate,
    insufficient_rate,
    verdict_distribution,
    verifier_precision_recall_f1,
    compare_before_after,
)


def test_retrieval_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c2", "c5", "c9"]

    # Recall@3: in top 3, only "c2" found out of 3 relevant => 1/3 = 0.3333
    r3 = recall_at_k(retrieved, relevant, k=3)
    assert abs(r3 - 1 / 3) < 0.01

    # Precision@3: in top 3, 1 relevant out of 3 => 1/3 = 0.3333
    p3 = precision_at_k(retrieved, relevant, k=3)
    assert abs(p3 - 1 / 3) < 0.01

    # MRR: first relevant is c2 at rank 2 => 1/2 = 0.5
    mrr = mean_reciprocal_rank(retrieved, relevant)
    assert mrr == 0.5


def test_claim_metrics():
    verdicts = [
        {"verdict": Verdict.SUPPORTED},
        {"verdict": Verdict.SUPPORTED},
        {"verdict": Verdict.CONTRADICTED},
        {"verdict": Verdict.INSUFFICIENT},
    ]

    assert grounding_rate(verdicts) == 0.5
    assert contradiction_rate(verdicts) == 0.25
    assert insufficient_rate(verdicts) == 0.25

    dist = verdict_distribution(verdicts)
    assert dist[Verdict.SUPPORTED] == 2
    assert dist[Verdict.CONTRADICTED] == 1
    assert dist[Verdict.INSUFFICIENT] == 1


def test_verifier_precision_recall_f1():
    preds = [Verdict.SUPPORTED, Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT]
    truth = [Verdict.SUPPORTED, Verdict.INSUFFICIENT, Verdict.CONTRADICTED, Verdict.INSUFFICIENT]

    metrics = verifier_precision_recall_f1(preds, truth, positive_label=Verdict.SUPPORTED)
    # tp=1 (first), fp=1 (second), fn=0
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_compare_before_after():
    initial = [
        {"verdict": Verdict.SUPPORTED},
        {"verdict": Verdict.CONTRADICTED},
        {"verdict": Verdict.INSUFFICIENT},
    ]
    final = [
        {"verdict": Verdict.SUPPORTED},
        {"verdict": Verdict.SUPPORTED},
        {"verdict": Verdict.SUPPORTED},
    ]
    comp = compare_before_after(initial, final)
    assert comp["initial_grounding_rate"] == 0.3333
    assert comp["final_grounding_rate"] == 1.0
    assert comp["grounding_improvement"] > 0
    assert comp["contradiction_reduction"] > 0
