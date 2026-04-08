"""
StatisticalRules miner - hand-coded correlation-based association rules.

Algorithm:
  FOR each pair (f_i, f_j) in first K features:
    Binarise at column mean
    Compute Pearson r
    IF |r| > corr_threshold: generate rule
"""

import pandas as pd
from scipy.stats import pearsonr

from aadmf.mining.base import BaseMiner


class StatRulesMiner(BaseMiner):
    """Fast correlation-based rule miner for stable data regimes."""

    def mine(self, X: pd.DataFrame, drift_score: float) -> dict:
        """
        1. Get first K features (K from config, default 4)
        2. Binarise each at column mean
        3. Compute Pearson r for all pairs
        4. Collect pairs where |r| > corr_threshold
        5. Return dict: {algorithm, rules_found, top_rules, quality_score}

        quality_score = min(rules_found / 5.0, 1.0)
        top_rules = list of (feat_a, feat_b, r) tuples
        """
        k_features = self.config.get("k_features", 4)
        corr_threshold = self.config.get("corr_threshold", 0.3)

        cols = X.columns[: min(k_features, X.shape[1])]
        X_sub = X[cols]
        X_bin = (X_sub > X_sub.mean()).astype(int)

        rules = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                feat_a = cols[i]
                feat_b = cols[j]
                x_i = X_bin.iloc[:, i]
                x_j = X_bin.iloc[:, j]

                # Pearson is undefined for constant vectors; skip these pairs.
                if x_i.nunique(dropna=False) <= 1 or x_j.nunique(dropna=False) <= 1:
                    continue

                r, _ = pearsonr(x_i, x_j)
                if abs(r) > corr_threshold:
                    rules.append((feat_a, feat_b, float(r)))

        rules_sorted = sorted(rules, key=lambda t: abs(t[2]), reverse=True)
        quality_score = min(len(rules_sorted) / 5.0, 1.0)

        return {
            "algorithm": "StatisticalRules",
            "rules_found": len(rules_sorted),
            "top_rules": rules_sorted[:3],
            "quality_score": quality_score,
        }
