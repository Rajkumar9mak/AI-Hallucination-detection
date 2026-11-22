"""Unit tests for FEVER and RAGTruth benchmark runners."""

import pytest
from evaluation.benchmark_runner import run_fever_benchmark, run_ragtruth_benchmark


def test_fever_benchmark_execution():
    res = run_fever_benchmark()
    assert "error" not in res
    assert res["total_samples"] == 10
    assert "accuracy" in res
    assert res["accuracy"] >= 0.0
    assert "confusion_matrix" in res
    assert "macro_f1" in res
    assert len(res["sample_results"]) == 10


def test_ragtruth_benchmark_execution():
    res = run_ragtruth_benchmark()
    assert "error" not in res
    assert res["total_samples"] == 6
    assert "detection_accuracy" in res
    assert "detection_precision" in res
    assert "detection_recall" in res
    assert "detection_f1" in res
    assert res["total_claims_evaluated"] > 0
    assert len(res["sample_results"]) == 6
