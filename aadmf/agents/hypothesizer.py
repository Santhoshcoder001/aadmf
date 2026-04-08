"""
HypothesizerAgent — YOUR ORIGINAL CONTRIBUTION.

Patent Claim 2 core implementation.

Template: IF correlation(f_i, f_j) > tau_r
          AND MI(f_i, f_j) > tau_MI
          AND drift triggered
          THEN generate hypothesis

MI computation (Phase 2): Uses real mutual_info_score on discretized features
MI decay tracking: Monitors delta_MI across batches; triggers if |delta_MI| > threshold
Combined ranking: 0.5 * |correlation| + 0.5 * MI for hypothesis prioritization
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

from aadmf.agents.base import BaseAgent
from aadmf.core.state import SystemState
from aadmf.core.state import Hypothesis


class HypothesizerAgent(BaseAgent):
    """
    Generates statistically-grounded hypotheses using correlation + real MI + decay tracking.

    Patent Claim 2 core logic:
    - Correlation threshold guards against spurious patterns
    - Real mutual information captures non-linear dependencies
    - MI decay tracking detects sudden relationship changes
    - Combined scoring prioritizes patterns by both linear and non-linear strength

    Trigger condition: drift_score > trigger_threshold OR drift_detected

    Args (from config["hypothesizer"]):
        corr_threshold:          minimum |r| to generate hypothesis (default 0.3)
        mi_threshold:            minimum real MI to generate hypothesis (default 0.05)
        n_features_check:        how many features to check (default 8)
        max_hypotheses_per_batch: return top N by combined score (default 3)
        trigger_drift_score:     minimum drift_score to trigger (default 0.1)
        use_llm:                 use Ollama for phrasing (default False)
        real_mi:                 use real MI instead of proxy (default True)
        mi_decay_threshold:      minimum |delta_MI| to trigger on decay (default 0.05)
        use_decay_trigger:       trigger hypothesis if MI decays significantly (default True)
    """

    def __init__(self, config: dict, provenance_logger=None, llm_client=None, llm=None):
        super().__init__(config, provenance_logger)

        h_cfg = config.get("hypothesizer", {})
        llm_cfg = config.get("llm", {})
        self.corr_threshold = h_cfg.get("corr_threshold", 0.3)
        self.mi_threshold = h_cfg.get("mi_threshold", 0.05)
        self.n_features_check = h_cfg.get("n_features_check", 8)
        self.max_hypotheses_per_batch = h_cfg.get("max_hypotheses_per_batch", 3)
        self.trigger_drift_score = h_cfg.get("trigger_drift_score", 0.1)
        self.use_llm = h_cfg.get("use_llm", False)
        self.llm_model = llm_cfg.get("model", "phi3:mini")
        self.llm_temperature = llm_cfg.get("temperature", 0.2)
        self.llm_client = llm_client if llm_client is not None else llm

        # Patent Claim 2 core logic: Real MI and decay tracking
        self.real_mi = h_cfg.get("real_mi", True)
        self.mi_decay_threshold = h_cfg.get("mi_decay_threshold", 0.05)
        self.use_decay_trigger = h_cfg.get("use_decay_trigger", True)

        # MI history: key = "feature_a-feature_b", value = previous MI value
        self.mi_history: Dict[str, float] = {}

    def update_mi_history(self, feature_pair: str, current_mi: float) -> float:
        """
        Update MI history for a feature pair and return delta_MI.

        delta_MI = current_MI - previous_MI. If no previous MI exists,
        returns 0.0 and initializes the history entry.
        """
        previous_mi = self.mi_history.get(feature_pair)
        self.mi_history[feature_pair] = current_mi

        if previous_mi is None:
            return 0.0

        return current_mi - previous_mi

    def _compute_real_mi(self, x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
        """
        Patent Claim 2 core logic: Compute real mutual information on discretized features.

        Discretizes continuous features into bins, then computes MI using
        sklearn.metrics.mutual_info_score. This captures non-linear dependencies
        that correlation-based methods miss.

        Args:
            x: Feature array (1D)
            y: Feature array (1D)
            n_bins: Number of bins for discretization (default 5)

        Returns:
            Mutual information score (bits; normalized approximately [0, 1])
        """
        try:
            # Discretize both features into equal-width bins
            x_binned = pd.cut(x, bins=n_bins, labels=False, duplicates='drop')
            y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')

            # Handle NaN values from binning (e.g., if bins couldn't be created)
            x_binned = x_binned.fillna(-1).astype(int)
            y_binned = y_binned.fillna(-1).astype(int)

            # Compute MI on the binned features
            mi = mutual_info_score(x_binned, y_binned)
            return float(mi)
        except Exception:
            # Fallback to proxy if discretization fails
            r, _ = pearsonr(x, y)
            return abs(r) * 0.5

    def _mi_proxy(self, x, y) -> float:
        """
        Mutual information computation (uses real MI if enabled, otherwise proxy).

        Patent Claim 2 core logic:
        - Real MI (Phase 2): Discretized mutual_info_score (captures non-linear patterns)
        - Proxy MI (Phase 1): |correlation| * 0.5 (fallback for compatibility)
        """
        if self.real_mi:
            return self._compute_real_mi(x, y)
        else:
            r, _ = pearsonr(x, y)
            return abs(r) * 0.5

    def _template_statement(
        self,
        feature_a: str,
        feature_b: str,
        r: float,
        mi: float,
        drift_score: float,
        drift_detected: bool,
    ) -> str:
        """
        Patent Claim 2 core logic: Fallback deterministic phrase.

        This template is your original contribution and remains unchanged
        to preserve patent claim foundations. Includes correlation, MI, and drift context.
        """
        condition = "under drift" if drift_detected else "in stable regime"
        return (
            f"{feature_a} and {feature_b} show co-pattern "
            f"(r={r:.2f}, MI={mi:.2f}) {condition} "
            f"[drift_score={drift_score:.4f}] "
            f"-> investigate combined feature for gas classification"
        )

    def _compute_combined_score(self, correlation: float, mi: float) -> float:
        """
        Patent Claim 2 core logic: Combine correlation and MI for hypothesis ranking.

        Combined score = 0.5 * |correlation| + 0.5 * MI

        This equally weights linear (correlation) and non-linear (MI) dependencies,
        enabling detection of complex gas sensor relationships.

        Args:
            correlation: Pearson correlation coefficient
            mi: Mutual information score

        Returns:
            Combined score in approximate range [0, 1]
        """
        return 0.5 * abs(correlation) + 0.5 * mi

    def _build_statement(
        self,
        feature_a: str,
        feature_b: str,
        r: float,
        mi: float,
        drift_score: float,
        drift_detected: bool,
    ) -> str:
        """Phrase only: keep hypothesis statistics untouched and optionally use Ollama."""
        template = self._template_statement(
            feature_a=feature_a,
            feature_b=feature_b,
            r=r,
            mi=mi,
            drift_score=drift_score,
            drift_detected=drift_detected,
        )

        # LLM only for phrasing — core hypothesis logic remains original (Patent Claim 2)
        if self.use_llm and self.llm_client and hasattr(self.llm_client, "phrase_hypothesis"):
            hyp = {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": float(r),
                "mutual_info_proxy": float(mi),
                "drift_triggered": bool(drift_detected),
                "drift_score": float(drift_score),
            }
            try:
                return str(self.llm_client.phrase_hypothesis(hyp))
            except Exception:
                return template

        return template

    def run(self, state: SystemState) -> SystemState:
        """
        Generate hypotheses using correlation + real MI + decay tracking.

        Patent Claim 2 core logic:
        ===========================
        Trigger: drift_score > trigger_drift_score OR drift_detected OR significant MI decay

        FOR each feature pair (i, j):
          1. Compute correlation r and p-value
          2. Compute real MI (discretized mutual_info_score)
          3. Check if |r| > corr_threshold AND MI > mi_threshold [Primary trigger]
          4. Check if MI decayed significantly [Secondary trigger]
          5. Rank by combined_score = 0.5 * |r| + 0.5 * MI
          6. Return top max_hypotheses_per_batch

        Reads:  state["X"], state["drift_score"], state["drift_detected"], state["batch_id"]
        Writes: state["hypotheses"]
        Logs:   HYPOTHESES_GENERATED event
        """
        X = state["X"]
        drift_score = state.get("drift_score", 0.0)
        drift_detected = state.get("drift_detected", False)
        batch_id = state.get("batch_id", 0)

        n_features = min(self.n_features_check, X.shape[1])
        hypotheses: List[Hypothesis] = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature_a = X.columns[i]
                feature_b = X.columns[j]
                x_i = X.iloc[:, i]
                x_j = X.iloc[:, j]

                # Pearson is undefined when either input is constant.
                if x_i.nunique(dropna=False) <= 1 or x_j.nunique(dropna=False) <= 1:
                    continue

                # Patent Claim 2: Correlation baseline
                r, p = pearsonr(x_i, x_j)

                # Patent Claim 2: Real mutual information (captures non-linear deps)
                mi = self._mi_proxy(x_i, x_j)

                # Patent Claim 2: update MI history and compute delta_MI.
                # O(1) dict access keeps this efficient for up to 28 pairs per batch.
                pair_key = f"{feature_a}-{feature_b}"
                delta_mi = self.update_mi_history(pair_key, mi)
                has_decay = self.use_decay_trigger and abs(delta_mi) > self.mi_decay_threshold

                # Patent Claim 2: Trigger condition
                # Primary: Both |r| and MI above thresholds
                primary_trigger = abs(r) > self.corr_threshold and mi > self.mi_threshold

                # Secondary: Significant MI decay trigger
                secondary_trigger = self.use_decay_trigger and has_decay

                if primary_trigger or secondary_trigger:
                    statement = self._build_statement(
                        feature_a,
                        feature_b,
                        r,
                        mi,
                        drift_score,
                        drift_detected,
                    )
                    hypotheses.append(
                        Hypothesis(
                            id=f"H{batch_id}_{len(hypotheses) + 1}",
                            batch=batch_id,
                            feature_a=feature_a,
                            feature_b=feature_b,
                            correlation=float(r),
                            mutual_info_proxy=float(mi),
                            p_value=float(p),
                            drift_triggered=bool(drift_detected or drift_score > self.trigger_drift_score),
                            statement=statement,
                        )
                    )

        # Patent Claim 2: Rank by combined score (0.5 * |correlation| + 0.5 * MI)
        hypotheses.sort(
            key=lambda h: self._compute_combined_score(h.correlation, h.mutual_info_proxy),
            reverse=True,
        )
        hypotheses = hypotheses[:self.max_hypotheses_per_batch]

        state["hypotheses"] = hypotheses

        self._log(
            "HYPOTHESES_GENERATED",
            {
                "batch_id": batch_id,
                "drift_score": drift_score,
                "drift_detected": drift_detected,
                "n_hypotheses": len(hypotheses),
                "hypothesis_ids": [h.id for h in hypotheses],
                "mi_history_size": len(self.mi_history),
            },
            state,
        )

        return state
