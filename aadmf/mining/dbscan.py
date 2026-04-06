"""
DBSCAN mining wrapper with adaptive eps.

Adaptive eps formula (YOUR LOGIC):
  eps = base_eps * (1.0 - 0.3 * drift_score)

Higher drift -> tighter eps -> more noise points identified as anomalies.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from aadmf.mining.base import BaseMiner


class DBSCANMiner(BaseMiner):
    """DBSCAN with drift-adaptive epsilon."""

    def mine(self, X: pd.DataFrame, drift_score: float) -> dict:
        """
        1. Compute adaptive eps from drift_score
        2. StandardScaler.fit_transform(X)
        3. DBSCAN.fit_predict(X_scaled)
        4. Count clusters (unique labels excluding -1)
        5. Count noise points (label == -1)
        6. Return dict: {algorithm, clusters, noise_points, quality_score}

        quality_score = clusters / (clusters + 1)
        """
        base_eps = self.config.get("base_eps", 1.5)
        min_samples = self.config.get("min_samples", 5)
        eps = base_eps * (1.0 - 0.3 * drift_score)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="auto",
            n_jobs=int(self.config.get("n_jobs", 1)),
        )
        labels = model.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = int((labels == -1).sum())
        quality_score = n_clusters / (n_clusters + 1)

        return {
            "algorithm": "DBSCAN",
            "eps": eps,
            "clusters": n_clusters,
            "noise_points": noise_points,
            "quality_score": quality_score,
        }
