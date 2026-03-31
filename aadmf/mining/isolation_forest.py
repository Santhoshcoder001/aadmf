"""
IsolationForest mining wrapper with adaptive contamination.

Adaptive contamination formula (YOUR LOGIC):
  contamination = max(0.05, min(0.45, 0.10 + 0.20 * drift_score))

Higher drift -> expect more anomalies -> increase contamination estimate.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from aadmf.mining.base import BaseMiner


class IFMiner(BaseMiner):
    """IsolationForest with drift-adaptive contamination."""

    def mine(self, X: pd.DataFrame, drift_score: float) -> dict:
        """
        1. Compute adaptive contamination from drift_score
        2. StandardScaler.fit_transform(X)
        3. IsolationForest.fit_predict(X_scaled)
        4. Count anomalies (label == -1)
        5. Return dict: {algorithm, anomalies, anomaly_rate, quality_score}

        quality_score = 1.0 - anomaly_rate
        """
        contamination = max(0.05, min(0.45, 0.10 + 0.20 * drift_score))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            max_samples="auto",
            random_state=self.config.get("seed", 42),
            n_jobs=-1,
        )
        preds = model.fit_predict(X_scaled)

        anomalies = int((preds == -1).sum())
        anomaly_rate = anomalies / len(X)
        quality_score = 1.0 - anomaly_rate

        return {
            "algorithm": "IsolationForest",
            "contamination": contamination,
            "anomalies": anomalies,
            "anomaly_rate": anomaly_rate,
            "quality_score": quality_score,
        }
