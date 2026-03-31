"""Streaming data simulator for AADMF PoC.

This module provides the `StreamingSimulator` class, which generates
batch-wise synthetic sensor data and injects concept drift after a
configured batch index.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class StreamingSimulator:
    """Generate streaming sensor batches with optional injected drift.

    The simulator mimics a 16-sensor gas dataset style stream where labels are
    integer gas classes in the range 1-6.

    Drift behavior (kept exactly as in PoC):
    - Drift starts when `current_batch > drift_after`.
    - A progressive mean shift is applied to sensors 0-7.
    - A progressive variance scaling is applied to sensors 0-3.
    """

    def __init__(
        self,
        n_batches: int = 10,
        batch_size: int = 100,
        n_features: int = 16,
        drift_after: int = 5,
        seed: int = 42,
    ) -> None:
        """Initialize the simulator configuration.

        Args:
            n_batches: Total number of batches available in the stream.
            batch_size: Number of rows generated per batch.
            n_features: Number of sensor features per row.
            drift_after: Batch index after which drift is injected.
            seed: Random seed for deterministic generation.
        """
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_features = n_features
        self.drift_after = drift_after
        self.rng = np.random.default_rng(seed)
        self.current_batch = 0

    def next_batch(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Return the next batch of features and labels.

        Returns:
            A tuple `(X, y)` where:
            - `X` is a DataFrame of shape `(batch_size, n_features)`
            - `y` is a Series named `gas_class` with integer labels in `[1, 6]`

            If the stream is exhausted, returns `(None, None)`.
        """
        if self.current_batch >= self.n_batches:
            return None, None
        drift = self.current_batch > self.drift_after
        # Base distribution: sensors ~ N(0,1)
        X = self.rng.normal(0, 1, (self.batch_size, self.n_features))
        if drift:
            shift = 1.5 * (self.current_batch - self.drift_after) / 5
            # Only sensors 0-7 drift (simulates real partial drift)
            X[:, :8] += shift
            # Add variance increase too
            X[:, :4] *= (1 + 0.2 * shift)
        y = self.rng.integers(1, 7, self.batch_size)
        self.current_batch += 1
        cols = [f"sensor_{i}" for i in range(self.n_features)]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="gas_class")
