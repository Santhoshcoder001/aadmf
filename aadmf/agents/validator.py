"""
ValidatorAgent — validates hypotheses via chi-square + synthetic augmentation.

YOUR NOVELTY in validation: double-validation via synthetic perturbation.
A hypothesis must pass chi-square test on BOTH real data AND noisy copy.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from aadmf.agents.base import BaseAgent
from aadmf.core.state import SystemState, Hypothesis
from typing import List


class ValidatorAgent(BaseAgent):
    """
    Validates hypotheses using:
    1. Chi-square test on real data (primary)
    2. Chi-square test on Gaussian-noisy copy (robustness check)

    Args (from config["validator"]):
        p_threshold_real:  max p-value for real data (default 0.05)
        p_threshold_noisy: max p-value for synthetic data (default 0.10)
        noise_pct:         noise amplitude as fraction (default 0.05 = +/-5%)
    """

    def __init__(self, config: dict, provenance_logger=None):
        super().__init__(config, provenance_logger)
        v_cfg = config.get("validator", {})
        self.p_threshold_real = v_cfg.get("p_threshold_real", 0.05)
        self.p_threshold_noisy = v_cfg.get("p_threshold_noisy", 0.10)
        self.noise_pct = v_cfg.get("noise_pct", 0.05)

    def _validate_one(self, hyp: Hypothesis, X: pd.DataFrame) -> Hypothesis:
        """
        Validate a single hypothesis.

        YOUR ALGORITHM (see Algorithms Document Section 6):
          1. Binarise feature_a at median -> a_bin
          2. Binarise feature_b at median -> b_bin
          3. chi2, p_real = chi2_contingency(crosstab(a_bin, b_bin))
          4. noise_factor ~ Uniform(1-noise_pct, 1+noise_pct)
          5. a_noisy = (feature_a * noise_factor > median) as int
          6. b_noisy = (feature_b * noise_factor > median) as int
          7. _, p_noisy = chi2_contingency(crosstab(a_noisy, b_noisy))
          8. valid = (p_real < p_threshold_real) AND (p_noisy < p_threshold_noisy)
          9. confidence = "HIGH" / "MEDIUM" / "LOW"
        """
        a = X[hyp.feature_a]
        b = X[hyp.feature_b]

        # 1-2. Binarize on real data medians
        a_bin = (a > a.median()).astype(int)
        b_bin = (b > b.median()).astype(int)

        # 3. Chi-square on real data
        contingency_real = pd.crosstab(a_bin, b_bin)
        chi2, p_real, _, _ = chi2_contingency(contingency_real)

        # 4. Synthetic perturbation via multiplicative noise
        noise_factor = np.random.uniform(1 - self.noise_pct, 1 + self.noise_pct, size=len(X))

        # 5-6. Build noisy binary features
        a_noisy_vals = a.to_numpy() * noise_factor
        b_noisy_vals = b.to_numpy() * noise_factor
        a_noisy = (a_noisy_vals > np.median(a_noisy_vals)).astype(int)
        b_noisy = (b_noisy_vals > np.median(b_noisy_vals)).astype(int)

        # 7. Chi-square on noisy data
        contingency_noisy = pd.crosstab(a_noisy, b_noisy)
        _, p_noisy, _, _ = chi2_contingency(contingency_noisy)

        # 8. Final validity decision
        valid = (p_real < self.p_threshold_real) and (p_noisy < self.p_threshold_noisy)

        # 9. Confidence from p_real
        if p_real < 0.01:
            confidence = "HIGH"
        elif p_real < 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        hyp.valid = valid
        hyp.confidence = confidence
        hyp.p_value_chi2 = float(p_real)
        hyp.p_value_synthetic = float(p_noisy)
        hyp.chi2 = float(chi2)
        return hyp

    def run(self, state: SystemState) -> SystemState:
        """
        Validate all hypotheses in state["hypotheses"].

        Reads:  state["hypotheses"], state["X"]
        Writes: state["validated_hypotheses"]
        Logs:   HYPOTHESIS_VALIDATED event per hypothesis
        """
        hypotheses = state.get("hypotheses", [])
        X = state.get("X")

        validated: List[Hypothesis] = []
        for hyp in hypotheses:
            validated_hyp = self._validate_one(hyp, X)
            validated.append(validated_hyp)

            self._log(
                "HYPOTHESIS_VALIDATED",
                {
                    "batch_id": state.get("batch_id", -1),
                    "hypothesis_id": validated_hyp.id,
                    "feature_a": validated_hyp.feature_a,
                    "feature_b": validated_hyp.feature_b,
                    "statement": validated_hyp.statement,
                    "p_value_chi2": validated_hyp.p_value_chi2,
                    "p_value_synthetic": validated_hyp.p_value_synthetic,
                    "chi2": validated_hyp.chi2,
                    "valid": validated_hyp.valid,
                    "confidence": validated_hyp.confidence,
                },
                state,
            )

        state["validated_hypotheses"] = validated
        return state
