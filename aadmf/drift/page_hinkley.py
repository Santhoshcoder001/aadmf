"""
Page-Hinkley Drift Detector — EWMA-enhanced implementation.

Standard PH uses a running arithmetic mean.
Your enhancement: exponentially weighted moving mean (EWMA) via alpha parameter.
This makes detection more sensitive to recent observations without losing stability.

Patent Novelty Point 1: EWMA-enhanced Page-Hinkley for streaming sensor data.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd


class PageHinkleyDriftDetector:
    """
    EWMA-enhanced Page-Hinkley test for concept drift detection.

    YOUR ALGORITHM (from Algorithms & Methodology Section 2.3):
    ============================================================

    INPUTS:
      x_t   = mean of primary sensor column in batch t
      α     = 0.9999 (forgetting factor — tunes sensitivity)
      δ     = 0.005  (minimum detectable change)
      λ     = 50.0   (alarm threshold)

    INITIALISE:
      x̄ ← 0, M ← 0, M* ← 0

    FOR each batch t:
      x̄  ← α × x̄ + (1−α) × x_t
      M  ← M + x_t − x̄ − δ
      M* ← min(M*, M)
      PH ← M − M*

      drift_score    ← min(PH / λ, 1.0)    [normalised 0–1]
      drift_detected ← PH > λ

    Args:
        delta:     minimum detectable change (sensitivity). Default 0.005.
        threshold: alarm threshold λ. Default 50.0.
        alpha:     EWMA forgetting factor. Default 0.9999.
    """

    def __init__(self, delta: float = 0.005,
                 threshold: float = 50.0,
                 alpha: float = 0.9999):
        """
        Initialize the EWMA-enhanced Page-Hinkley detector.

        Args:
            delta: minimum detectable change (δ in algorithm)
            threshold: alarm threshold (λ in algorithm)
            alpha: EWMA forgetting factor (α in algorithm)
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        # Initialize state variables (YOUR ALGORITHM init step)
        self.x_mean = 0.0          # x̄
        self.M = 0.0               # M (cumulative deviation)
        self.M_min = 0.0           # M* (running minimum)

        self.logger = logging.getLogger(self.__class__.__name__)

    def update(self, X: pd.DataFrame) -> Tuple[bool, float]:
        """
        Process one batch using EWMA-enhanced Page-Hinkley test.

        YOUR ALGORITHM implementation (exact formula from Section 2.3):
        =============================================================

        FOR each batch t:
          x_t  = mean(X[:, primary_sensor])  (sensor_0)
          x̄  ← α × x̄ + (1−α) × x_t
          M  ← M + x_t − x̄ − δ
          M* ← min(M*, M)
          PH ← M − M*
          drift_score    ← min(PH / λ, 1.0)
          drift_detected ← PH > λ

        Args:
            X: pd.DataFrame of current batch features (shape: n_samples × n_features)

        Returns:
            Tuple[bool, float]: (drift_detected, drift_score)
                - drift_detected: True if PH > threshold
                - drift_score: normalized PH value in [0.0, 1.0]
        """
        # Step 1: Extract mean of primary sensor (sensor_0)
        if isinstance(X, pd.DataFrame):
            x_t = X.iloc[:, 0].mean()  # mean of first column (sensor_0)
        else:
            x_t = X[:, 0].mean()  # numpy array fallback

        # Step 2: Update EWMA mean
        # x̄ ← α × x̄ + (1−α) × x_t
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * x_t

        # Step 3: Update cumulative deviation
        # M ← M + (x_t − x̄ − δ)
        m = x_t - self.x_mean - self.delta
        self.M = self.M + m

        # Step 4: Track running minimum
        # M* ← min(M*, M)
        self.M_min = min(self.M_min, self.M)

        # Step 5: Compute Page-Hinkley statistic
        # PH ← M − M*
        PH = self.M - self.M_min

        # Step 6: Normalize drift score
        # drift_score ← min(PH / λ, 1.0)
        drift_score = min(PH / self.threshold, 1.0)

        # Step 7: Detect drift
        # drift_detected ← PH > λ
        drift_detected = PH > self.threshold

        self.logger.debug(
            f"PH update: x_t={x_t:.4f}, x̄={self.x_mean:.4f}, M={self.M:.4f}, "
            f"M*={self.M_min:.4f}, PH={PH:.4f}, drift_score={drift_score:.4f}, "
            f"drift_detected={drift_detected}"
        )

        return drift_detected, drift_score

    def reset(self):
        """
        Reset internal state (call at start of new experiment run).
        Reinitializes x̄, M, M* to 0.
        """
        self.x_mean = 0.0
        self.M = 0.0
        self.M_min = 0.0
        self.logger.info("PageHinkleyDriftDetector state reset")
