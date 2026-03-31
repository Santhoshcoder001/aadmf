"""
PlannerAgent — YOUR original scoring matrix + EMA accuracy feedback.

Patent Claim 1 core implementation.
Selects mining algorithm based on:
  score = 0.4 * drift_weight * drift_score
        + 0.3 * accuracy_history
        + 0.3 * (1 - cost)
"""

from aadmf.agents.base import BaseAgent
from aadmf.core.state import SystemState
from typing import Dict


class PlannerAgent(BaseAgent):
    """
    Autonomous algorithm selector using drift-weighted scoring matrix.

    YOUR ORIGINAL CONTRIBUTION — see Algorithms Document Section 3
    for full mathematical derivation.
    """

    # Algorithm registry — YOUR design choices
    # From Algorithms & Methodology Document 2, Section 3.3
    ALGORITHM_REGISTRY = {
        "IsolationForest": {
            "drift_weight": 0.9,
            "cost": 0.3,
        },
        "DBSCAN": {
            "drift_weight": 0.6,
            "cost": 0.4,
        },
        "StatisticalRules": {
            "drift_weight": 0.4,
            "cost": 0.1,
        },
    }

    def __init__(self, config: dict, provenance_logger=None):
        """
        Initialize PlannerAgent with weight parameters and accuracy history.

        Args:
            config: Configuration dict with optional keys:
                - "planner": { "w_drift": 0.4, "w_accuracy": 0.3, "w_cost": 0.3,
                               "alpha_ema": 0.7 }
            provenance_logger: DictChainLogger for audit trail
        """
        super().__init__(config, provenance_logger)

        planner_cfg = config.get("planner", {})

        # Weight parameters (must sum to 1.0)
        # Supports both:
        #   planner.scoring_weights.{drift,accuracy,cost}
        # and legacy keys:
        #   planner.w_drift, planner.w_accuracy, planner.w_cost
        scoring_weights = planner_cfg.get("scoring_weights", {})
        self.w_drift = scoring_weights.get("drift", planner_cfg.get("w_drift", 0.4))
        self.w_accuracy = scoring_weights.get("accuracy", planner_cfg.get("w_accuracy", 0.3))
        self.w_cost = scoring_weights.get("cost", planner_cfg.get("w_cost", 0.3))

        # EMA smoothing factor for accuracy history
        # From Algorithms & Methodology Document 2, Section 3.2
        self.alpha_ema = planner_cfg.get("alpha_ema", 0.7)

        # Initialize accuracy history for each algorithm
        self.accuracy_history: Dict[str, float] = {
            algo: 0.5 for algo in self.ALGORITHM_REGISTRY.keys()
        }

    def set_scoring_weights(self, new_weights_dict: Dict[str, float]) -> None:
        """Dynamically update scoring weights during experiments.

        Expected keys: ``drift``, ``accuracy``, ``cost``.
        Missing keys keep their current values.
        """
        self.w_drift = float(new_weights_dict.get("drift", self.w_drift))
        self.w_accuracy = float(new_weights_dict.get("accuracy", self.w_accuracy))
        self.w_cost = float(new_weights_dict.get("cost", self.w_cost))

    def _compute_score(self, algo: str, drift_score: float) -> float:
        """
        Compute selection score for one algorithm using scoring matrix.

        YOUR ALGORITHM (from Algorithms & Methodology Section 3.1):
        ===========================================================

        score(a_i, t) = w_d × drift_weight(a_i) × drift_score(t)
                      + w_a × accuracy_history(a_i, t)
                      + w_c × (1 − cost(a_i))

        where: w_d = 0.4, w_a = 0.3, w_c = 0.3
               w_d + w_a + w_c = 1.0

        Args:
            algo: Algorithm name (key in ALGORITHM_REGISTRY)
            drift_score: Current drift score in [0.0, 1.0]

        Returns:
            float: Composite score for this algorithm
        """
        if algo not in self.ALGORITHM_REGISTRY:
            return 0.0

        registry = self.ALGORITHM_REGISTRY[algo]
        drift_weight = registry["drift_weight"]
        cost = registry["cost"]
        acc_hist = self.accuracy_history[algo]

        # YOUR EXACT FORMULA
        score = (
            self.w_drift * drift_weight * drift_score
            + self.w_accuracy * acc_hist
            + self.w_cost * (1.0 - cost)
        )

        return score

    def run(self, state: SystemState) -> SystemState:
        """
        Select best algorithm based on current drift score.

        Algorithm:
          1. For each algorithm in registry: compute_score(algo, drift_score)
          2. Choose algorithm with maximum score
          3. Update state with choice and individual scores
          4. Log decision to provenance

        Args:
            state: SystemState with drift_score populated

        Returns:
            Updated state with chosen_algorithm and algorithm_scores set
        """
        drift_score = state.get("drift_score", 0.0)

        # Compute scores for all algorithms
        scores: Dict[str, float] = {
            algo: self._compute_score(algo, drift_score)
            for algo in self.ALGORITHM_REGISTRY.keys()
        }

        # Select algorithm with highest score
        chosen_algorithm = max(scores, key=scores.get)

        # Update state
        state["chosen_algorithm"] = chosen_algorithm
        state["algorithm_scores"] = scores

        # Log decision
        self._log(
            "ALGO_SELECTED",
            {
                "batch_id": state.get("batch_id", -1),
                "drift_score": drift_score,
                "algorithm_scores": scores,
                "chosen_algorithm": chosen_algorithm,
                "scoring_weights": {
                    "drift": self.w_drift,
                    "accuracy": self.w_accuracy,
                    "cost": self.w_cost,
                },
            },
            state,
        )

        self.logger.info(
            f"Batch {state.get('batch_id')}: Selected {chosen_algorithm} "
            f"(score={scores[chosen_algorithm]:.4f}, drift={drift_score:.4f})"
        )

        return state

    def update_accuracy(self, algo: str, quality_score: float) -> None:
        """
        Update accuracy history for an algorithm using EMA.

        YOUR ALGORITHM (from Algorithms & Methodology Section 3.2):
        ==========================================================

        accuracy_history(a, t+1) = α_ema × accuracy_history(a, t)
                                 + (1 − α_ema) × quality_score(t)

        where α_ema = 0.7 (configurable)

        This ensures the planner learns from experience: algorithms that
        consistently produce high quality scores get a higher accuracy
        history, biasing future selections toward them.

        Args:
            algo: Algorithm name (key in ALGORITHM_REGISTRY)
            quality_score: Quality score from this batch's mining output [0.0, 1.0]
        """
        if algo not in self.accuracy_history:
            self.accuracy_history[algo] = 0.5

        # YOUR EXACT EMA FORMULA
        old_acc = self.accuracy_history[algo]
        new_acc = (
            self.alpha_ema * old_acc
            + (1.0 - self.alpha_ema) * quality_score
        )
        self.accuracy_history[algo] = new_acc

        self.logger.debug(
            f"Updated {algo} accuracy: {old_acc:.4f} → {new_acc:.4f} "
            f"(quality_score={quality_score:.4f}, α_ema={self.alpha_ema})"
        )
