"""Miner tuning experiments on real UCI Gas Sensor drift data.

This module tunes the three miners independently on all 10 UCI batches:
- IsolationForest (base_contamination + adaptive contamination formulas)
- DBSCAN (base_eps + adaptive eps formulas)
- StatisticalRules (corr_threshold)

Outputs:
- Per-batch quality scores for every configuration
- CSV saved to experiments/results/miner_tuning_results.csv
- Printed summary of best parameters for low-drift and high-drift batches
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from aadmf.drift.page_hinkley import PageHinkleyDriftDetector
from aadmf.streaming.uci_loader import UCIGasSensorLoader


@dataclass(frozen=True)
class IFConfig:
    base_contamination: float
    formula_name: str


@dataclass(frozen=True)
class DBSCANConfig:
    base_eps: float
    formula_name: str


@dataclass(frozen=True)
class StatRulesConfig:
    corr_threshold: float


class MinerTuningExperiments:
    """Run miner parameter sweeps on UCI batches and export results."""

    def __init__(
        self,
        config: dict,
        output_csv: str = "experiments/results/miner_tuning_results.csv",
        low_drift_threshold: float = 0.10,
        high_drift_threshold: float = 0.50,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.output_csv = Path(output_csv)
        self.low_drift_threshold = low_drift_threshold
        self.high_drift_threshold = high_drift_threshold

        self.if_formulas: Dict[str, Callable[[float, float], float]] = {
            "fixed": lambda base, drift: base,
            "linear_0.20": lambda base, drift: base + 0.20 * drift,
            "linear_0.30": lambda base, drift: base + 0.30 * drift,
            "linear_0.15": lambda base, drift: base + 0.15 * drift,
        }
        self.dbscan_formulas: Dict[str, Callable[[float, float], float]] = {
            "fixed": lambda base, drift: base,
            "shrink_0.30": lambda base, drift: base * (1.0 - 0.30 * drift),
            "shrink_0.50": lambda base, drift: base * (1.0 - 0.50 * drift),
            "inverse_0.50": lambda base, drift: base / (1.0 + 0.50 * drift),
        }

    @classmethod
    def from_yaml(
        cls,
        config_path: str = "config.yaml",
        output_csv: str = "experiments/results/miner_tuning_results.csv",
    ) -> "MinerTuningExperiments":
        """Build tuner from YAML config file."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg, output_csv=output_csv)

    def _load_uci_batches(self) -> Tuple[List[Tuple[pd.DataFrame, pd.Series]], List[float]]:
        """Load all 10 UCI batches and compute per-batch drift_score."""
        uci_cfg = dict(self.config.get("uci_loader", self.config.get("uci_streaming", {})))
        uci_cfg["batch_numbers"] = list(range(1, 11))
        loader = UCIGasSensorLoader(**uci_cfg)
        batches = loader.load_all_batches()

        drift_cfg = self.config.get("drift_detection", {})
        detector = PageHinkleyDriftDetector(**drift_cfg)
        drift_scores: List[float] = []
        for X, _ in batches:
            _, score = detector.update(X)
            drift_scores.append(float(score))

        return batches, drift_scores

    def _drift_band(self, drift_score: float) -> str:
        if drift_score <= self.low_drift_threshold:
            return "low"
        if drift_score >= self.high_drift_threshold:
            return "high"
        return "mid"

    @staticmethod
    def _run_isolation_forest(
        X: pd.DataFrame,
        contamination: float,
        seed: int,
    ) -> float:
        contamination = float(np.clip(contamination, 0.01, 0.49))
        X_scaled = StandardScaler().fit_transform(X)
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            max_samples="auto",
            random_state=seed,
            n_jobs=-1,
        )
        preds = model.fit_predict(X_scaled)
        anomaly_rate = float((preds == -1).mean())
        return 1.0 - anomaly_rate

    @staticmethod
    def _run_dbscan(
        X: pd.DataFrame,
        eps: float,
        min_samples: int,
    ) -> float:
        eps = max(0.05, float(eps))
        X_scaled = StandardScaler().fit_transform(X)
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return float(n_clusters / (n_clusters + 1))

    @staticmethod
    def _run_stat_rules(
        X: pd.DataFrame,
        corr_threshold: float,
        k_features: int,
    ) -> float:
        cols = X.columns[: min(k_features, X.shape[1])]
        X_bin = (X[cols] > X[cols].mean()).astype(int)

        rules = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, _ = pearsonr(X_bin.iloc[:, i], X_bin.iloc[:, j])
                if abs(r) > corr_threshold:
                    rules += 1

        return float(min(rules / 5.0, 1.0))

    def _if_grid(self) -> Iterable[IFConfig]:
        bases = [round(x, 2) for x in np.arange(0.05, 0.46, 0.05)]
        for base in bases:
            for formula_name in self.if_formulas:
                yield IFConfig(base_contamination=base, formula_name=formula_name)

    def _dbscan_grid(self) -> Iterable[DBSCANConfig]:
        bases = [round(x, 2) for x in np.arange(0.5, 2.51, 0.5)]
        for base in bases:
            for formula_name in self.dbscan_formulas:
                yield DBSCANConfig(base_eps=base, formula_name=formula_name)

    def _statrules_grid(self) -> Iterable[StatRulesConfig]:
        thresholds = [round(x, 2) for x in np.arange(0.2, 0.51, 0.05)]
        for thr in thresholds:
            yield StatRulesConfig(corr_threshold=thr)

    def run_all(self) -> pd.DataFrame:
        """Run all miner tuning experiments and save per-batch scores to CSV."""
        batches, drift_scores = self._load_uci_batches()
        seed = int(self.config.get("streaming", {}).get("seed", 42))
        min_samples = int(self.config.get("dbscan", {}).get("min_samples", 5))
        k_features = int(self.config.get("statistical_rules", {}).get("k_features", 4))

        rows: List[dict] = []

        print("\n" + "=" * 72)
        print("Running IsolationForest tuning on 10 UCI batches")
        for cfg in self._if_grid():
            for batch_idx, ((X, _), drift_score) in enumerate(zip(batches, drift_scores), start=1):
                contamination = self.if_formulas[cfg.formula_name](cfg.base_contamination, drift_score)
                quality = self._run_isolation_forest(X, contamination=contamination, seed=seed)
                rows.append(
                    {
                        "miner": "IsolationForest",
                        "config": f"base={cfg.base_contamination},formula={cfg.formula_name}",
                        "batch_id": batch_idx,
                        "drift_score": drift_score,
                        "drift_band": self._drift_band(drift_score),
                        "quality_score": quality,
                        "base_contamination": cfg.base_contamination,
                        "adaptive_formula": cfg.formula_name,
                        "effective_param": round(float(np.clip(contamination, 0.01, 0.49)), 6),
                    }
                )

        print("\n" + "=" * 72)
        print("Running DBSCAN tuning on 10 UCI batches")
        for cfg in self._dbscan_grid():
            for batch_idx, ((X, _), drift_score) in enumerate(zip(batches, drift_scores), start=1):
                eps = self.dbscan_formulas[cfg.formula_name](cfg.base_eps, drift_score)
                quality = self._run_dbscan(X, eps=eps, min_samples=min_samples)
                rows.append(
                    {
                        "miner": "DBSCAN",
                        "config": f"base={cfg.base_eps},formula={cfg.formula_name}",
                        "batch_id": batch_idx,
                        "drift_score": drift_score,
                        "drift_band": self._drift_band(drift_score),
                        "quality_score": quality,
                        "base_eps": cfg.base_eps,
                        "adaptive_formula": cfg.formula_name,
                        "effective_param": round(float(max(0.05, eps)), 6),
                    }
                )

        print("\n" + "=" * 72)
        print("Running StatisticalRules tuning on 10 UCI batches")
        for cfg in self._statrules_grid():
            for batch_idx, ((X, _), drift_score) in enumerate(zip(batches, drift_scores), start=1):
                quality = self._run_stat_rules(X, corr_threshold=cfg.corr_threshold, k_features=k_features)
                rows.append(
                    {
                        "miner": "StatisticalRules",
                        "config": f"corr_threshold={cfg.corr_threshold}",
                        "batch_id": batch_idx,
                        "drift_score": drift_score,
                        "drift_band": self._drift_band(drift_score),
                        "quality_score": quality,
                        "corr_threshold": cfg.corr_threshold,
                        "adaptive_formula": "none",
                        "effective_param": cfg.corr_threshold,
                    }
                )

        df = pd.DataFrame(rows)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)

        print("\n" + "=" * 72)
        print(f"Saved miner tuning results to: {self.output_csv}")

        summary = self.best_params_by_drift_band(df)
        print("\nBest parameter summary (low vs high drift):")
        print(summary.to_string(index=False))

        return df

    def best_params_by_drift_band(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return best config per miner for low-drift and high-drift batches."""
        work = df.copy()

        # Ensure high band exists; if not, map top 30% drift scores to high.
        if (work["drift_band"] == "high").sum() == 0:
            cutoff = float(work["drift_score"].quantile(0.70))
            work.loc[work["drift_score"] >= cutoff, "drift_band"] = "high"

        candidate = work[work["drift_band"].isin(["low", "high"])].copy()
        grouped = (
            candidate
            .groupby(["miner", "drift_band", "config"], as_index=False)["quality_score"]
            .mean()
            .rename(columns={"quality_score": "mean_quality_score"})
        )

        idx = grouped.groupby(["miner", "drift_band"])["mean_quality_score"].idxmax()
        best = grouped.loc[idx].sort_values(["miner", "drift_band"]).reset_index(drop=True)
        return best


if __name__ == "__main__":
    tuner = MinerTuningExperiments.from_yaml("config.yaml")
    tuner.run_all()
