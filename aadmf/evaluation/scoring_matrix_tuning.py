"""Scoring matrix weight tuning on real UCI gas sensor drift batches.

This module provides `ScoringMatrixTuner` to evaluate multiple planner weight
configurations using the full AADMF orchestrator pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type
import copy

import pandas as pd
import yaml

from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.streaming.uci_loader import UCIGasSensorLoader


WeightTriple = Tuple[float, float, float]


@dataclass(frozen=True)
class WeightPreset:
    """Named scoring-weight preset for planner tuning."""

    name: str
    drift: float
    accuracy: float
    cost: float


class ScoringMatrixTuner:
    """Run scoring-matrix tuning experiments on UCI batches.

    Supports using either the full `ManualOrchestrator` or another compatible
    orchestrator class (for example a LangGraph-based flow wrapper) as long as
    it implements:
        orchestrator = Orchestrator(config)
        results = orchestrator.run(streamer)
    where `results["results_df"]` is a DataFrame with at least:
    `batch_id`, `drift_score`, `drift_detected`, `algorithm`, `quality_score`.
    """

    PRESETS: List[WeightPreset] = [
        WeightPreset("Default", 0.4, 0.3, 0.3),
        WeightPreset("Drift-dominant", 0.7, 0.2, 0.1),
        WeightPreset("Accuracy-dominant", 0.2, 0.7, 0.1),
        WeightPreset("Cost-dominant", 0.1, 0.2, 0.7),
    ]

    def __init__(
        self,
        base_config: dict,
        orchestrator_cls: Type = ManualOrchestrator,
        output_csv: str = "experiments/results/scoring_matrix_tuning.csv",
        high_drift_threshold: float = 0.5,
    ) -> None:
        self.base_config = copy.deepcopy(base_config)
        self.orchestrator_cls = orchestrator_cls
        self.output_csv = Path(output_csv)
        self.high_drift_threshold = high_drift_threshold

    @classmethod
    def from_yaml(
        cls,
        config_path: str = "config.yaml",
        orchestrator_cls: Type = ManualOrchestrator,
        output_csv: str = "experiments/results/scoring_matrix_tuning.csv",
        high_drift_threshold: float = 0.5,
    ) -> "ScoringMatrixTuner":
        """Create tuner from YAML config file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(
            base_config=config,
            orchestrator_cls=orchestrator_cls,
            output_csv=output_csv,
            high_drift_threshold=high_drift_threshold,
        )

    def _build_config_for_weights(self, weights: WeightPreset) -> dict:
        cfg = copy.deepcopy(self.base_config)

        planner_cfg = cfg.setdefault("planner", {})
        planner_cfg["w_drift"] = weights.drift
        planner_cfg["w_accuracy"] = weights.accuracy
        planner_cfg["w_cost"] = weights.cost

        # Force real-data mode and all 10 batches.
        stream_cfg = cfg.setdefault("streaming", {})
        stream_cfg["dataset"] = "uci"
        stream_cfg["uci_batch_count"] = 10

        uci_cfg = cfg.setdefault("uci_loader", {})
        uci_cfg.setdefault("data_dir", "data/raw")
        uci_cfg["batch_numbers"] = list(range(1, 11))
        uci_cfg.setdefault("normalize", True)
        uci_cfg.setdefault("use_ucimlrepo", True)

        return cfg

    @staticmethod
    def _algorithm_distribution(results_df: pd.DataFrame) -> Dict[str, int]:
        counts = results_df["algorithm"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    @staticmethod
    def _drift_detection_latency(results_df: pd.DataFrame) -> Optional[int]:
        """Latency to first detected drift, measured from batch start (0-index).

        Returns None if drift was never detected.
        """
        detected = results_df[results_df["drift_detected"] == True]
        if detected.empty:
            return None
        first_batch = int(detected.iloc[0]["batch_id"])
        return first_batch

    def _high_drift_mean_quality(self, results_df: pd.DataFrame) -> float:
        high_mask = results_df["drift_score"] >= self.high_drift_threshold
        subset = results_df[high_mask]

        # If threshold yields no rows, fallback to top 30% drift-score batches.
        if subset.empty:
            n = max(1, int(round(len(results_df) * 0.3)))
            subset = results_df.sort_values("drift_score", ascending=False).head(n)

        return float(subset["quality_score"].mean())

    def _run_single(self, weights: WeightPreset) -> dict:
        cfg = self._build_config_for_weights(weights)

        print("\n" + "=" * 72)
        print(
            f"Running weight set: {weights.name} "
            f"(drift={weights.drift}, accuracy={weights.accuracy}, cost={weights.cost})"
        )

        uci_cfg = cfg.get("uci_loader", {})
        loader = UCIGasSensorLoader(**uci_cfg)
        orchestrator = self.orchestrator_cls(cfg)
        results = orchestrator.run(loader)

        results_df = results["results_df"].copy()
        mean_quality = float(results_df["quality_score"].mean())
        algo_dist = self._algorithm_distribution(results_df)
        latency = self._drift_detection_latency(results_df)
        high_drift_quality = self._high_drift_mean_quality(results_df)

        print(f"Mean quality score: {mean_quality:.4f}")
        print(f"Algorithm distribution: {algo_dist}")
        print(f"Drift detection latency (from batch 0): {latency}")
        print(f"High-drift mean quality score: {high_drift_quality:.4f}")

        return {
            "preset": weights.name,
            "w_drift": weights.drift,
            "w_accuracy": weights.accuracy,
            "w_cost": weights.cost,
            "mean_quality_score": round(mean_quality, 6),
            "high_drift_mean_quality_score": round(high_drift_quality, 6),
            "drift_detection_latency": latency,
            "algorithm_selection_distribution": str(algo_dist),
        }

    def run_all(self) -> pd.DataFrame:
        """Run all 4 weight presets and save CSV results."""
        rows: List[dict] = []
        for preset in self.PRESETS:
            rows.append(self._run_single(preset))

        df = pd.DataFrame(rows)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)

        print("\n" + "=" * 72)
        print(f"Saved scoring matrix tuning results to: {self.output_csv}")
        print(df.to_string(index=False))

        best = self.recommend_best(df)
        print("\nRecommended best configuration (high-drift criterion):")
        print(best.to_string())

        return df

    def recommend_best(self, results_df: pd.DataFrame) -> pd.Series:
        """Recommend best weights by highest high-drift mean quality score."""
        idx = results_df["high_drift_mean_quality_score"].idxmax()
        return results_df.loc[idx]


if __name__ == "__main__":
    tuner = ScoringMatrixTuner.from_yaml("config.yaml")
    tuner.run_all()
