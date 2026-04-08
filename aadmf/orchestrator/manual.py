"""
ManualOrchestrator - Phase 1 orchestrator.
Runs the full agent pipeline as a simple Python loop.
Uses LangGraph StateGraph for agent orchestration.
"""

import time
import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)

try:
    from langgraph.graph import StateGraph, END
    _HAS_LANGGRAPH = True
except Exception:
    StateGraph = None
    END = None
    _HAS_LANGGRAPH = False

from aadmf.core.state import SystemState, BatchResult
from aadmf.drift.page_hinkley import PageHinkleyDriftDetector
from aadmf.agents.planner import PlannerAgent
from aadmf.agents.miner import MinerAgent
from aadmf.agents.hypothesizer import HypothesizerAgent
from aadmf.agents.validator import ValidatorAgent
from aadmf.provenance.dict_chain import DictChainLogger
from aadmf.streaming.simulator import StreamingSimulator
from aadmf.streaming.uci_loader import UCIGasSensorLoader


def _resolve_drift_config(config: dict) -> dict:
    """Resolve drift config with backward-compatible key support.

    Supports both:
    - ``drift_detection`` (legacy, currently used across repo)
    - ``drift_detector`` (newer config naming)
    """
    legacy = dict(config.get("drift_detection", {}))
    modern = dict(config.get("drift_detector", {}))

    if modern:
        method = str(modern.get("method", "page_hinkley")).lower()
        if method != "page_hinkley":
            print(f"[WARN] Unsupported drift_detector.method='{method}', using page_hinkley.")
        merged = {
            "delta": modern.get("delta", legacy.get("delta", 0.005)),
            "threshold": modern.get("threshold", legacy.get("threshold", 50.0)),
            "alpha": modern.get("alpha", legacy.get("alpha", 0.9999)),
            "min_batch_size": modern.get("min_batch_size", legacy.get("min_batch_size", 1)),
            "use_relative_change": modern.get("use_relative_change", legacy.get("use_relative_change", False)),
        }
    else:
        merged = {
            "delta": legacy.get("delta", 0.005),
            "threshold": legacy.get("threshold", 50.0),
            "alpha": legacy.get("alpha", 0.9999),
            "min_batch_size": legacy.get("min_batch_size", 1),
            "use_relative_change": legacy.get("use_relative_change", False),
        }

    return merged


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
            err_text = str(e).rstrip(".")
            # Fallback to DictChainLogger if Neo4j connection fails
            print(
                "[WARN] Neo4j unavailable. Install 'neo4j' and start Neo4j Desktop or a Docker "
                f"container, then check the Bolt URI. Error: {err_text}. Falling back to DictChainLogger."
            )
            logger = DictChainLogger()
            logger.log(
                "SYSTEM",
                {
                    "message": (
                        "Provenance backend switch: Neo4j failed, using DictChainLogger. "
                        f"Install hint: pip install neo4j; start Neo4j Desktop/Docker; error: {err_text}"
                    )
                },
            )
            return logger
    else:
        # Default to DictChainLogger
        logger = DictChainLogger()
        backend_msg = "DictChainLogger" if backend != "neo4j" else "Neo4j disabled (using DictChainLogger fallback)"
        logger.log("SYSTEM", {"message": f"Provenance backend: {backend_msg}"})
        return logger


def build_streamer_from_config(config: dict):
    """Create the configured streamer using ``config['streaming']['dataset']``.

    Supported datasets:
    - ``synthetic`` -> ``StreamingSimulator``
    - ``uci`` -> ``UCIGasSensorLoader``
    """
    stream_cfg = config.get("streaming", {})
    dataset = str(stream_cfg.get("dataset", "synthetic")).lower()

    if dataset == "uci":
        uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))

        # Allow quick control from streaming section while keeping uci_loader optional.
        uci_batch_count = int(stream_cfg.get("uci_batch_count", 10))
        if uci_cfg.get("batch_numbers") is None:
            uci_cfg["batch_numbers"] = list(range(1, max(1, uci_batch_count) + 1))

        return UCIGasSensorLoader(**uci_cfg)

    simulator_cfg = {
        "n_batches": stream_cfg.get("n_batches", 10),
        "batch_size": stream_cfg.get("batch_size", 100),
        "n_features": stream_cfg.get("n_features", 16),
        "drift_after": stream_cfg.get("drift_after", 5),
        "seed": stream_cfg.get("seed", 42),
    }
    return StreamingSimulator(**simulator_cfg)


class ManualOrchestrator:
    """
    Connects all agents in the correct order.
    Manages SystemState across batches.
    """

    def __init__(self, config: dict, use_langgraph: bool = False):
        self.config = config
        self.use_langgraph = use_langgraph
        self.prov = _create_provenance_logger(config)
        self._langgraph_flow = None

        # Instantiate all agents with the provenance logger
        self.detector = PageHinkleyDriftDetector(**_resolve_drift_config(config))
        self.planner = PlannerAgent(config, self.prov)
        self.miner = MinerAgent(config, self.prov)
        self.hypothesizer = HypothesizerAgent(config, self.prov)
        self.validator = ValidatorAgent(config, self.prov)

        self.graph = self._build_graph() if _HAS_LANGGRAPH else None

        if not _HAS_LANGGRAPH and self.use_langgraph:
            print("[WARN] langgraph is not installed. Falling back to manual orchestrator mode.")
            self.use_langgraph = False

        if self.use_langgraph:
            # Lazy import keeps backward compatibility for environments that only
            # use the original manual loop.
            from aadmf.orchestrator.langgraph_flow import LangGraphOrchestrator

            self._langgraph_flow = LangGraphOrchestrator(config, provenance_logger=self.prov)

    def _build_graph(self):
        """Build LangGraph StateGraph using Section 5.2 node/edge design."""
        if not _HAS_LANGGRAPH:
            return None

        graph = StateGraph(SystemState)

        # Add nodes
        graph.add_node("drift_detect", self._drift_detect_node)
        graph.add_node("plan", self.planner.run)
        graph.add_node("mine", self._mine_node)
        graph.add_node("hypothesize", self.hypothesizer.run)
        graph.add_node("validate", self.validator.run)
        graph.add_node("log_batch", self._log_batch_node)

        # Define edges
        graph.set_entry_point("drift_detect")
        graph.add_edge("drift_detect", "plan")
        graph.add_edge("plan", "mine")

        # Week 7: always run hypothesizer so MI-decay can track relationship
        # changes across consecutive batches even when detector drift_score is low.
        graph.add_edge("mine", "hypothesize")
        graph.add_edge("hypothesize", "validate")
        graph.add_edge("validate", "log_batch")
        graph.add_edge("log_batch", END)

        return graph.compile()

    def _drift_detect_node(self, state: SystemState) -> SystemState:
        drift_detected, drift_score = self.detector.update(state["X"])
        state["drift_detected"] = drift_detected
        state["drift_score"] = drift_score
        return state

    def _mine_node(self, state: SystemState) -> SystemState:
        state = self.miner.run(state)
        chosen_algo = state["chosen_algorithm"]
        quality_score = float(state["mining_result"].get("quality_score", 0.0))
        self.planner.update_accuracy(chosen_algo, quality_score)
        return state

    def _log_batch_node(self, state: SystemState) -> SystemState:
        return state

    def _run_pipeline_stepwise(self, state: SystemState) -> SystemState:
        """Sequential fallback when langgraph dependency is unavailable."""
        state = self._drift_detect_node(state)
        state = self.planner.run(state)
        state = self._mine_node(state)
        state = self.hypothesizer.run(state)
        state = self.validator.run(state)
        state = self._log_batch_node(state)
        return state

    def run(self, streamer) -> dict:
        """
        Main loop. Processes batches until streamer is exhausted.

        Args:
            streamer: any object with next_batch() returning (X, y) or (None, None)

        Returns:
            dict with: results_df, provenance_summary, all_hypotheses, valid_hypotheses
        """
        if self.use_langgraph:
            print("Using LangGraph Agent Orchestration")
            return self._langgraph_flow.run_graph(streamer)

        # Fallback manual loop is preserved for debugging/backward compatibility.
        history = []
        all_hypotheses = []
        valid_hypotheses = []

        self.prov.log("SYSTEM_START", {"component": "ManualOrchestrator"})

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

            if self.graph is not None:
                state = self.graph.invoke(state)
            else:
                state = self._run_pipeline_stepwise(state)

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

    def print_results(self, results: dict) -> None:
        """Print formatted results table and provenance summary."""
        df = results["results_df"]
        prov = results["provenance_summary"]
        all_h = results["all_hypotheses"]
        valid_h = results["valid_hypotheses"]

        print("\n" + "=" * 72)
        print("AADMF PHASE 1 RESULTS")
        print("=" * 72)
        if not df.empty:
            print(df.to_string(index=False))
            print("\nAlgorithm Distribution:")
            print(df["algorithm"].value_counts().to_string())
            print(f"\nAverage quality_score: {df['quality_score'].mean():.4f}")

        hvr = (len(valid_h) / len(all_h)) if all_h else 0.0
        print(f"Total hypotheses: {len(all_h)}")
        print(f"Valid hypotheses: {len(valid_h)}")
        print(f"HVR (hypothesis validity rate): {hvr:.4f}")

        print("\nProvenance Summary:")
        print(prov)
        intact, broken_at = self.prov.verify_integrity()
        print(f"Integrity check: intact={intact}, broken_at={broken_at}")
