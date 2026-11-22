"""
Dataset loaders for evaluation (RAGTruth and FEVER subsets).

Provides utilities to load and format benchmark datasets
for verifier evaluation and end-to-end experiments.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_fever_subset(
    data_path: Path,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    Load a FEVER dataset subset for claim verification evaluation.

    Expected JSONL format per line:
        {"id": ..., "claim": "...", "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
         "evidence": "..."}

    Maps FEVER labels to project verdicts:
        SUPPORTS → Supported
        REFUTES → Contradicted
        NOT ENOUGH INFO → Insufficient Evidence

    Args:
        data_path: Path to the FEVER JSONL file.
        max_samples: Maximum number of samples to load.

    Returns:
        List of dicts with claim, label (project verdict), evidence, original_label.
    """
    from config.settings import Verdict

    label_map = {
        "SUPPORTS": Verdict.SUPPORTED,
        "REFUTES": Verdict.CONTRADICTED,
        "NOT ENOUGH INFO": Verdict.INSUFFICIENT,
    }

    data_path = Path(data_path)
    if not data_path.exists():
        logger.warning("FEVER data file not found: %s", data_path)
        return []

    samples: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            try:
                entry = json.loads(line.strip())
                original_label = entry.get("label", "").upper()
                mapped = label_map.get(original_label, Verdict.INSUFFICIENT)
                samples.append({
                    "claim": entry.get("claim", ""),
                    "label": mapped,
                    "evidence": entry.get("evidence", ""),
                    "original_label": original_label,
                    "id": entry.get("id"),
                })
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d FEVER sample(s) from %s", len(samples), data_path)
    return samples


def load_ragtruth_subset(
    data_path: Path,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    Load a RAGTruth dataset subset.

    RAGTruth provides word-level hallucination spans — this loader
    preserves the raw data for downstream adaptation. It does NOT
    assume direct atomic-claim labels.

    Args:
        data_path: Path to the RAGTruth JSON/JSONL file.
        max_samples: Maximum samples to load.

    Returns:
        List of raw RAGTruth entries.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        logger.warning("RAGTruth data file not found: %s", data_path)
        return []

    samples: list[dict] = []

    # Try JSONL first, then JSON array
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                for entry in data:
                    if max_samples and len(samples) >= max_samples:
                        break
                    samples.append(entry)
            else:
                for line in f:
                    if max_samples and len(samples) >= max_samples:
                        break
                    try:
                        samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.exception("Failed to load RAGTruth data: %s", e)

    logger.info("Loaded %d RAGTruth sample(s) from %s", len(samples), data_path)
    return samples


def create_controlled_query_set() -> list[dict]:
    """
    Return a small built-in controlled query set for end-to-end testing.

    These queries are designed to work with any general-knowledge
    document collection and test various verification scenarios.
    """
    return [
        {
            "query_id": "q_001",
            "query": "What is retrieval-augmented generation?",
            "category": "factual",
        },
        {
            "query_id": "q_002",
            "query": "How does BM25 work for text retrieval?",
            "category": "factual",
        },
        {
            "query_id": "q_003",
            "query": "What are the main causes of hallucination in language models?",
            "category": "analytical",
        },
        {
            "query_id": "q_004",
            "query": "Compare dense retrieval with lexical retrieval.",
            "category": "comparative",
        },
        {
            "query_id": "q_005",
            "query": "What is natural language inference and how is it used for fact checking?",
            "category": "factual",
        },
    ]
