"""
MinerAgent — dispatches to correct mining algorithm based on Planner's choice.
Feeds quality_score back to PlannerAgent after execution.
"""

import pandas as pd

from aadmf.agents.base import BaseAgent
from aadmf.core.state import SystemState


class MinerAgent(BaseAgent):
    """Dispatches to IFMiner, DBSCANMiner, or StatRulesMiner."""

    def __init__(self, config: dict, provenance_logger=None):
        super().__init__(config, provenance_logger)

        # Import and instantiate mining wrappers
        from aadmf.mining.isolation_forest import IFMiner
        from aadmf.mining.dbscan import DBSCANMiner
        from aadmf.mining.statistical_rules import StatRulesMiner

        self._miners = {
            "IsolationForest": IFMiner(config.get("isolation_forest", {})),
            "DBSCAN": DBSCANMiner(config.get("dbscan", {})),
            "StatisticalRules": StatRulesMiner(config.get("statistical_rules", {})),
        }

    def run(self, state: SystemState) -> SystemState:
        """
        Execute chosen algorithm on current batch.

        Reads:  state["chosen_algorithm"], state["X"], state["drift_score"]
        Writes: state["mining_result"]
        Logs:   MINING_RESULT event
        """
        # 1. Get chosen_algorithm from state
        chosen_algorithm = state.get("chosen_algorithm", "")

        # 2. Dispatch to correct miner
        X = state.get("X")
        drift_score = state.get("drift_score", 0.0)
        result = self._miners[chosen_algorithm].mine(X, drift_score)

        # 3. Store result in state
        state["mining_result"] = result

        # 4. Log MINING_RESULT event
        self._log(
            "MINING_RESULT",
            {
                "batch_id": state.get("batch_id", -1),
                "chosen_algorithm": chosen_algorithm,
                "drift_score": drift_score,
                "quality_score": result.get("quality_score"),
                "result": result,
            },
            state,
        )

        # 5. Return updated state
        return state
