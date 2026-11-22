"""
Benchmark execution runner for FEVER and RAGTruth datasets.

Provides empirical evaluation pipelines to benchmark claim verification
accuracy on FEVER and hallucination detection fidelity on RAGTruth.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from config.settings import Verdict, DATA_DIR
from evaluation.datasets import load_fever_subset, load_ragtruth_subset
from evaluation.claim_metrics import verifier_precision_recall_f1, macro_f1
from verification.verdict_mapper import verify_all_claims
from verification.claim_extractor import extract_claims

logger = logging.getLogger(__name__)

DATASETS_DIR = DATA_DIR / "datasets"


def run_fever_benchmark(
    samples: Optional[list[dict]] = None,
    dataset_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    nli_device: str = "cpu",
    entailment_threshold: float = 0.7,
    contradiction_threshold: float = 0.7,
) -> dict:
    """
    Run empirical claim verification benchmark on FEVER dataset.

    Args:
        samples: Preloaded list of FEVER sample dicts.
        dataset_path: Path to fever JSONL file if samples not provided.
        max_samples: Maximum samples to evaluate.
        nli_model_name: NLI verifier model name.
        nli_device: Device for NLI model.
        entailment_threshold: Supported threshold.
        contradiction_threshold: Contradicted threshold.

    Returns:
        Dict with overall accuracy, per-class metrics, macro F1,
        confusion matrix, and detailed sample results.
    """
    if samples is None:
        p = dataset_path or (DATASETS_DIR / "fever_subset.jsonl")
        samples = load_fever_subset(p, max_samples=max_samples)

    if not samples:
        return {"error": "No FEVER samples available for evaluation."}

    start_time = time.time()
    results = []
    y_true = []
    y_pred = []

    # Confusion matrix tracker: [actual][predicted]
    labels_order = [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT]
    conf_matrix = {
        actual: {pred: 0 for pred in labels_order}
        for actual in labels_order
    }

    for idx, sample in enumerate(samples):
        claim_text = sample["claim"]
        evidence_text = sample["evidence"]
        expected_label = sample["label"]

        # Package as chunk dict for the verifier
        ev_chunk = [{
            "chunk_id": f"ev_{idx}",
            "source_name": "fever_evidence",
            "page_number": 1,
            "text": evidence_text,
        }]

        verif = verify_all_claims(
            claims=[{"claim_id": f"c_{idx}", "claim_text": claim_text}],
            evidence_chunks=ev_chunk,
            nli_model_name=nli_model_name,
            nli_device=nli_device,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )

        pred_verdict = verif["verdicts"][0]["verdict"] if verif["verdicts"] else Verdict.INSUFFICIENT

        y_true.append(expected_label)
        y_pred.append(pred_verdict)
        conf_matrix[expected_label][pred_verdict] += 1

        is_correct = (expected_label == pred_verdict)
        results.append({
            "id": sample.get("id", f"sample_{idx}"),
            "claim": claim_text,
            "evidence": evidence_text,
            "ground_truth": expected_label,
            "predicted_verdict": pred_verdict,
            "correct": is_correct,
            "nli_label": verif["verdicts"][0].get("aggregate_nli_label", ""),
        })

    elapsed_s = time.time() - start_time
    total = len(y_true)
    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_count / total if total > 0 else 0.0

    # Calculate per-class metrics & macro F1
    macro_scores = macro_f1(y_pred, y_true)

    return {
        "dataset": "FEVER",
        "total_samples": total,
        "correct_predictions": correct_count,
        "accuracy": round(accuracy, 4),
        "macro_precision": macro_scores["macro_precision"],
        "macro_recall": macro_scores["macro_recall"],
        "macro_f1": macro_scores["macro_f1"],
        "per_class_metrics": macro_scores["per_label"],
        "confusion_matrix": conf_matrix,
        "labels_order": labels_order,
        "sample_results": results,
        "elapsed_seconds": round(elapsed_s, 2),
    }


def run_ragtruth_benchmark(
    samples: Optional[list[dict]] = None,
    dataset_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
    nli_device: str = "cpu",
) -> dict:
    """
    Run empirical hallucination detection benchmark on RAGTruth dataset.

    Evaluates RAGTruth responses by extracting atomic claims, verifying
    against retrieved source context, and evaluating hallucination detection
    accuracy against ground-truth hallucination annotations.

    Args:
        samples: Preloaded list of RAGTruth sample dicts.
        dataset_path: Path to ragtruth JSONL file if samples not provided.
        max_samples: Maximum samples to evaluate.
        nli_model_name: NLI verifier model name.
        nli_device: Device for NLI model.

    Returns:
        Dict with detection precision, recall, F1, grounding rates,
        task breakdowns, and detailed sample results.
    """
    if samples is None:
        p = dataset_path or (DATASETS_DIR / "ragtruth_subset.jsonl")
        samples = load_ragtruth_subset(p, max_samples=max_samples)

    if not samples:
        return {"error": "No RAGTruth samples available for evaluation."}

    start_time = time.time()
    results = []
    y_true_hallu = []
    y_pred_hallu = []

    total_claims_all = 0
    supported_claims_all = 0
    contradicted_claims_all = 0
    insufficient_claims_all = 0

    for idx, item in enumerate(samples):
        source_context = item.get("source_context", "")
        model_response = item.get("model_response", "")
        ground_truth_has_hallu = item.get("has_hallucination", False)
        task_name = item.get("task", "General")

        # 1. Extract claims from response
        extracted = extract_claims(model_response, method="rule_based")
        claims = extracted.get("claims", [])

        # 2. Package source context as evidence chunk
        ev_chunks = [{
            "chunk_id": f"ragtruth_ctx_{idx}",
            "source_name": f"ragtruth_{task_name}",
            "page_number": 1,
            "text": source_context,
        }]

        # 3. Verify all claims
        verif = verify_all_claims(
            claims=claims,
            evidence_chunks=ev_chunks,
            nli_model_name=nli_model_name,
            nli_device=nli_device,
        )

        claim_verdicts = verif.get("verdicts", [])
        num_claims = len(claim_verdicts)
        total_claims_all += num_claims

        # Count verdicts
        n_supp = sum(1 for v in claim_verdicts if v["verdict"] == Verdict.SUPPORTED)
        n_contra = sum(1 for v in claim_verdicts if v["verdict"] == Verdict.CONTRADICTED)
        n_insuf = sum(1 for v in claim_verdicts if v["verdict"] == Verdict.INSUFFICIENT)

        supported_claims_all += n_supp
        contradicted_claims_all += n_contra
        insufficient_claims_all += n_insuf

        # Predicted hallucination: response is flagged as hallucinated if ANY claim is Contradicted or Insufficient
        pred_has_hallu = (n_contra > 0 or n_insuf > 0)
        y_true_hallu.append(ground_truth_has_hallu)
        y_pred_hallu.append(pred_has_hallu)

        gr = (n_supp / num_claims) if num_claims > 0 else 0.0

        results.append({
            "id": item.get("id", f"ragtruth_{idx}"),
            "task": task_name,
            "question": item.get("question", ""),
            "model_response": model_response,
            "claims_count": num_claims,
            "supported_claims": n_supp,
            "contradicted_claims": n_contra,
            "insufficient_claims": n_insuf,
            "grounding_rate": round(gr, 4),
            "ground_truth_has_hallu": ground_truth_has_hallu,
            "predicted_has_hallu": pred_has_hallu,
            "detection_correct": (ground_truth_has_hallu == pred_has_hallu),
            "hallucinated_spans": item.get("hallucinated_spans", []),
            "claims_detail": claim_verdicts,
        })

    elapsed_s = time.time() - start_time
    total_samples = len(samples)

    # Hallucination detection classification metrics (Positive = has_hallucination)
    tp = sum(1 for yt, yp in zip(y_true_hallu, y_pred_hallu) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true_hallu, y_pred_hallu) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_hallu, y_pred_hallu) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true_hallu, y_pred_hallu) if not yt and not yp)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / total_samples if total_samples > 0 else 0.0

    return {
        "dataset": "RAGTruth",
        "total_samples": total_samples,
        "total_claims_evaluated": total_claims_all,
        "detection_accuracy": round(acc, 4),
        "detection_precision": round(prec, 4),
        "detection_recall": round(rec, 4),
        "detection_f1": round(f1, 4),
        "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "aggregate_grounding_rate": round(supported_claims_all / total_claims_all, 4) if total_claims_all > 0 else 0.0,
        "aggregate_contradiction_rate": round(contradicted_claims_all / total_claims_all, 4) if total_claims_all > 0 else 0.0,
        "aggregate_insufficient_rate": round(insufficient_claims_all / total_claims_all, 4) if total_claims_all > 0 else 0.0,
        "sample_results": results,
        "elapsed_seconds": round(elapsed_s, 2),
    }
