"""
KMeans mining wrapper for compact cluster-based adaptation.

This miner provides a lightweight third option for planner selection.
It standardizes features, adapts the number of clusters to drift, and
returns a normalized quality score derived from within-cluster inertia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from aadmf.mining.base import BaseMiner


class KMeansMiner(BaseMiner):
    """KMeans-based miner with drift-aware cluster count."""

    def _cluster_count(self, n_samples: int, drift_score: float) -> int:
        base_clusters = int(self.config.get("base_clusters", 2))
        max_clusters = int(self.config.get("max_clusters", 5))
        drift_bonus = 1 if drift_score >= 0.5 else 0
        n_clusters = base_clusters + drift_bonus
        n_clusters = min(max(2, n_clusters), max(2, max_clusters))
        n_clusters = min(n_clusters, max(2, n_samples - 1))
        return n_clusters

    def mine(self, X: pd.DataFrame, drift_score: float) -> dict:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = self._cluster_count(len(X_scaled), drift_score)
        model = KMeans(
            n_clusters=n_clusters,
            n_init=int(self.config.get("n_init", 10)),
            random_state=int(self.config.get("seed", 42)),
        )
        labels = model.fit_predict(X_scaled)

        inertia_per_sample = float(model.inertia_) / max(len(X_scaled), 1)
        quality_score = 1.0 / (1.0 + inertia_per_sample)

        return {
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "inertia": float(model.inertia_),
            "quality_score": quality_score,
            "labels": labels.tolist(),
        }
