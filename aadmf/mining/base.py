"""Base class for all mining algorithm wrappers."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseMiner(ABC):
    """All mining algorithm wrappers implement this interface."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def mine(self, X: pd.DataFrame, drift_score: float) -> dict:
        """
        Run mining algorithm on current batch.

        Args:
            X:           pd.DataFrame - current batch (already scaled internally)
            drift_score: float - used for adaptive parameter tuning

        Returns:
            dict with at minimum: {"algorithm": str, "quality_score": float}
            Plus algorithm-specific keys.
        """
        pass
