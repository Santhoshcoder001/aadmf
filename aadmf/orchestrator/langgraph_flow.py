"""LangGraph-based orchestrator flow for AADMF.

This module adds a graph-oriented orchestrator while preserving existing
agent logic. It reuses the current agents and state schema, and only defines
how control flows between nodes.
"""

from __future__ import annotations

import time
import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)

from langgraph.graph import StateGraph, END

from aadmf.core.state import SystemState, BatchResult
from aadmf.drift.page_hinkley import PageHinkleyDriftDetector
from aadmf.agents.planner import PlannerAgent
from aadmf.agents.miner import MinerAgent
from aadmf.agents.hypothesizer import HypothesizerAgent
from aadmf.agents.validator import ValidatorAgent
from aadmf.provenance.dict_chain import DictChainLogger


def _create_provenance_logger(config: dict):
    """Factory function to create the configured provenance logger.

    Attempts to instantiate the configured backend (Neo4j or DictChain).
    If Neo4j connection fails, falls back to DictChainLogger.
    Logs a SYSTEM message when backend is instantiated or switched.

    Args:
        config: configuration dict with 'provenance' settings

    Returns:
        Provenance logger instance (Neo4jLogger or DictChainLogger)
    """
    prov_cfg = config.get("provenance", {})
    backend = prov_cfg.get("backend", "dict").lower()
    neo4j_cfg = prov_cfg.get("neo4j", {})
    enabled = neo4j_cfg.get("enabled", True)

    if backend == "neo4j" and enabled:
        try:
            from aadmf.provenance.neo4j_graph import Neo4jLogger

            logger = Neo4jLogger(
                password=neo4j_cfg.get("password", ""),
                uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
                user=neo4j_cfg.get("user", "neo4j"),
            )
            logger.log("SYSTEM", {"message": "Provenance backend: Neo4j"})
            return logger
        except Exception as e:
            # Fallback to DictChainLogger if Neo4j connection fails
            print(f"[WARN] Neo4j connection failed: {e}. Falling back to DictChainLogger.")
            logger = DictChainLogger()
            logger.log("SYSTEM", {"message": f"Provenance backend switch: Neo4j failed, using DictChainLogger. Error: {str(e)}"})
            return logger
    else:
        # Default to DictChainLogger
        logger = DictChainLogger()
        backend_msg = "DictChainLogger" if backend != "neo4j" else "Neo4j disabled (using DictChainLogger fallback)"
        logger.log("SYSTEM", {"message": f"Provenance backend: {backend_msg}"})
        return logger


class LangGraphOrchestrator:
    """StateGraph orchestrator that wires the existing AADMF agents.

    Design notes:
    - Reuses the current ``SystemState`` TypedDict without adding fields.
    - Reuses the exact agent classes and their ``run`` methods as graph nodes.
    - Uses conditional branching after mining to avoid expensive hypothesis
      generation/validation when drift signal is weak.
    """

    def __init__(self, config: dict, provenance_logger=None):
        self.config = config
        self.prov = provenance_logger if provenance_logger is not None else _create_provenance_logger(config)

        # Keep original agent implementations untouched; only orchestration is new.
        self.detector = PageHinkleyDriftDetector(**config.get("drift_detection", {}))
        self.planner = PlannerAgent(config, self.prov)
        self.miner = MinerAgent(config, self.prov)
        self.hypothesizer = HypothesizerAgent(config, self.prov)
        self.validator = ValidatorAgent(config, self.prov)

        self._compiled_graph = None

    def _drift_detect_node(self, state: SystemState) -> SystemState:
        """Update drift flags/scores in state using Page-Hinkley detector."""
        drift_detected, drift_score = self.detector.update(state["X"])
        state["drift_detected"] = drift_detected
        state["drift_score"] = drift_score
        return state

    def _mine_node(self, state: SystemState) -> SystemState:
        """Run miner and update planner EMA accuracy with resulting quality score."""
        state = self.miner.run(state)
        chosen_algo = state["chosen_algorithm"]
        quality_score = float(state["mining_result"].get("quality_score", 0.0))
        self.planner.update_accuracy(chosen_algo, quality_score)
        return state

    def _log_batch_node(self, state: SystemState) -> SystemState:
        """Terminal bookkeeping node; no state mutation required here."""
        return state

    def _build_graph(self):
        """Build the StateGraph topology and return compiled graph.

        Graph structure:
        1. drift_detect -> 2. plan -> 3. mine
        4. Conditional branch from mine:
           - If drift_score > 0.1 OR drift_detected is True:
             mine -> hypothesize -> validate -> log_batch -> END
           - Otherwise:
             mine -> log_batch -> END

        Why conditional edges:
        - Hypothesis generation and validation are drift-focused operations.
        - Gating these nodes prevents unnecessary compute when drift evidence is
          weak, while still preserving all existing agent behavior when invoked.
        """
        graph = StateGraph(SystemState)

        # Node mapping directly wraps existing components.
        graph.add_node("drift_detect", self._drift_detect_node)
        graph.add_node("plan", self.planner.run)
        graph.add_node("mine", self._mine_node)
        graph.add_node("hypothesize", self.hypothesizer.run)
        graph.add_node("validate", self.validator.run)
        graph.add_node("log_batch", self._log_batch_node)

        graph.set_entry_point("drift_detect")
        graph.add_edge("drift_detect", "plan")
        graph.add_edge("plan", "mine")

        # Drift-triggered conditional branch after mining.
        graph.add_conditional_edges(
            "mine",
            lambda s: "hypothesize" if (s["drift_score"] > 0.1 or s["drift_detected"] is True) else "log_batch",
            {"hypothesize": "hypothesize", "log_batch": "log_batch"},
        )

        # validate is only reached when hypothesis generation was executed.
        graph.add_edge("hypothesize", "validate")
        graph.add_edge("validate", "log_batch")
        graph.add_edge("log_batch", END)

        return graph.compile()

    def compile(self):
        """Compile and cache the LangGraph flow; return compiled graph object."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()
        return self._compiled_graph

    def run_graph(self, streamer) -> dict:
        """Process all batches from streamer through the compiled LangGraph.

        Args:
            streamer: object exposing ``next_batch() -> (X, y)`` and then
                ``(None, None)`` when exhausted.

        Returns:
            Dict with keys:
            - ``results_df``: per-batch run summary DataFrame
            - ``provenance_summary``: hash-chain summary
            - ``all_hypotheses``: all generated hypotheses
            - ``valid_hypotheses``: validated hypotheses where ``valid=True``
        """
        graph = self.compile()

        history = []
        all_hypotheses = []
        valid_hypotheses = []

        self.prov.log("SYSTEM_START", {"component": "LangGraphOrchestrator"})

        batch_id = 0
        while True:
            start = time.perf_counter()

            X, y = streamer.next_batch()
            if X is None:
                break

            self.prov.log("DATA_INGESTED", {"batch": batch_id, "rows": len(X), "cols": X.shape[1]})

            state: SystemState = {
                "batch_id": batch_id,
                "X": X,
                "y": y,
                "drift_score": 0.0,
                "drift_detected": False,
                "chosen_algorithm": "",
                "algorithm_scores": {},
                "mining_result": {},
                "hypotheses": [],
                "validated_hypotheses": [],
                "provenance_hash": "",
                "history": [],
                "error": None,
            }

            state = graph.invoke(state)

            quality_score = float(state["mining_result"].get("quality_score", 0.0))
            runtime_ms = (time.perf_counter() - start) * 1000.0

            batch_result = BatchResult(
                batch_id=batch_id,
                drift_score=float(state["drift_score"]),
                drift_detected=bool(state["drift_detected"]),
                algorithm=state["chosen_algorithm"],
                algo_scores=state["algorithm_scores"],
                quality_score=quality_score,
                n_hypotheses=len(state["hypotheses"]),
                n_valid_hypotheses=sum(1 for h in state["validated_hypotheses"] if h.valid),
                runtime_ms=runtime_ms,
                provenance_hash=self.prov.prev_hash,
            )

            history.append(batch_result)
            all_hypotheses.extend(state["hypotheses"])
            valid_hypotheses.extend([h for h in state["validated_hypotheses"] if h.valid])

            batch_id += 1

        rows = []
        for r in history:
            rows.append(
                {
                    "batch_id": r.batch_id,
                    "drift_score": r.drift_score,
                    "drift_detected": r.drift_detected,
                    "algorithm": r.algorithm,
                    "quality_score": r.quality_score,
                    "n_hypotheses": r.n_hypotheses,
                    "n_valid_hypotheses": r.n_valid_hypotheses,
                    "runtime_ms": round(r.runtime_ms, 2),
                }
            )

        return {
            "results_df": pd.DataFrame(rows),
            "provenance_summary": self.prov.summary(),
            "all_hypotheses": all_hypotheses,
            "valid_hypotheses": valid_hypotheses,
        }
