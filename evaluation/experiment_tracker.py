"""
Experiment tracking module for the RAG system.

Records experiment configurations, metrics, and results to JSON files
for reproducibility and comparison.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Records and manages experiment results."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: list[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load previously saved experiments."""
        results_file = self.results_dir / "experiments.json"
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    self._experiments = json.load(f)
                logger.info("Loaded %d existing experiment(s)", len(self._experiments))
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load existing experiments; starting fresh")
                self._experiments = []

    def create_experiment(
        self,
        retrieval_method: str,
        top_k: int,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: str = "",
        verifier_model: str = "",
        correction_strategy: str = "",
        llm_model: str = "",
        notes: str = "",
    ) -> str:
        """
        Create a new experiment record.

        Returns:
            The experiment ID.
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding_model,
                "verifier_model": verifier_model,
                "correction_strategy": correction_strategy,
                "llm_model": llm_model,
            },
            "metrics": {},
            "latency_ms": None,
            "inference_cost": None,
            "notes": notes,
            "status": "created",
        }

        self._experiments.append(record)
        logger.info("Created experiment: %s", experiment_id)
        return experiment_id

    def update_metrics(self, experiment_id: str, metrics: dict) -> None:
        """Add or update metrics for an experiment."""
        record = self._find(experiment_id)
        if record:
            record["metrics"].update(metrics)
            record["status"] = "metrics_recorded"
            self._save()

    def update_latency(self, experiment_id: str, latency_ms: float) -> None:
        """Record latency for an experiment."""
        record = self._find(experiment_id)
        if record:
            record["latency_ms"] = latency_ms
            self._save()

    def update_cost(self, experiment_id: str, cost: Optional[dict]) -> None:
        """Record inference cost for an experiment."""
        record = self._find(experiment_id)
        if record:
            record["inference_cost"] = cost
            self._save()

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as complete."""
        record = self._find(experiment_id)
        if record:
            record["status"] = "completed"
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Retrieve a single experiment by ID."""
        return self._find(experiment_id)

    def get_all_experiments(self) -> list[dict]:
        """Return all recorded experiments."""
        return list(self._experiments)

    def get_comparison_table(self) -> list[dict]:
        """
        Return a simplified comparison table across all completed experiments.
        """
        rows = []
        for exp in self._experiments:
            if exp.get("status") != "completed":
                continue
            row = {
                "experiment_id": exp["experiment_id"],
                **exp.get("config", {}),
                **exp.get("metrics", {}),
                "latency_ms": exp.get("latency_ms"),
            }
            rows.append(row)
        return rows

    def _find(self, experiment_id: str) -> Optional[dict]:
        for exp in self._experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        logger.warning("Experiment not found: %s", experiment_id)
        return None

    def _save(self) -> None:
        results_file = self.results_dir / "experiments.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self._experiments, f, indent=2, default=str)
