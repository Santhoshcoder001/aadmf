"""LangGraph parity and routing test script.

This script validates three things:
1. UCI data can be loaded for 10 batches.
2. LangGraph orchestrator produces the same per-batch outcomes as the
   legacy manual orchestrator when both follow the same execution path.
3. Conditional routing behaves correctly in LangGraph:
   - low drift: hypothesizer is skipped
   - high drift: hypothesizer runs

Run:
    python test_langgraph.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aadmf.agents.hypothesizer import HypothesizerAgent
from aadmf.agents.miner import MinerAgent
from aadmf.agents.planner import PlannerAgent
from aadmf.agents.validator import ValidatorAgent
from aadmf.core.state import SystemState
from aadmf.drift.page_hinkley import PageHinkleyDriftDetector
from aadmf.orchestrator.langgraph_flow import LangGraphOrchestrator
from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.provenance.dict_chain import DictChainLogger
from aadmf.streaming.uci_loader import UCIGasSensorLoader


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure local UCI files are used in this repo.
    config.setdefault("streaming", {})
    config.setdefault("uci_loader", {})
    config.setdefault("uci_streaming", {})

    config["streaming"]["dataset"] = "uci"
    config["streaming"]["uci_batch_count"] = 10

    data_dir = "dataset1"
    config["uci_loader"]["data_dir"] = data_dir
    config["uci_loader"]["batch_numbers"] = list(range(1, 11))
    config["uci_loader"]["use_ucimlrepo"] = False

    # Keep alias in sync for any path that uses legacy key names.
    config["uci_streaming"]["data_dir"] = data_dir
    config["uci_streaming"]["batch_numbers"] = list(range(1, 11))
    config["uci_streaming"]["use_ucimlrepo"] = False

    # Keep this parity test independent from external Neo4j availability.
    config.setdefault("provenance", {})
    config["provenance"]["backend"] = "dict"
    config["provenance"].setdefault("neo4j", {})
    config["provenance"]["neo4j"]["enabled"] = False

    return config


def _make_loader(config: dict) -> UCIGasSensorLoader:
    uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))
    return UCIGasSensorLoader(**uci_cfg)


def _create_all_agents(config: dict) -> None:
    """Instantiate all core agents and components requested in the prompt."""
    prov = DictChainLogger()

    detector = PageHinkleyDriftDetector(**config.get("drift_detection", {}))
    planner = PlannerAgent(config, prov)
    miner = MinerAgent(config, prov)
    hypothesizer = HypothesizerAgent(config, prov)
    validator = ValidatorAgent(config, prov)

    # Lightweight sanity assertions to guarantee construction succeeded.
    assert detector is not None
    assert planner is not None
    assert miner is not None
    assert hypothesizer is not None
    assert validator is not None


def _build_state(batch_id: int, X: pd.DataFrame, y: pd.Series) -> SystemState:
    return {
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


def _results_keyframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize result columns for equality checks (exclude runtime)."""
    cols = [
        "batch_id",
        "drift_score",
        "drift_detected",
        "algorithm",
        "quality_score",
        "n_hypotheses",
        "n_valid_hypotheses",
    ]
    out = df[cols].copy()
    out["drift_score"] = out["drift_score"].round(6)
    out["quality_score"] = out["quality_score"].round(6)
    return out.reset_index(drop=True)


def _run_parity_test(config: dict) -> None:
    """Compare manual orchestrator vs LangGraph over 10 UCI batches."""
    manual = ManualOrchestrator(copy.deepcopy(config), use_langgraph=False)
    langgraph = LangGraphOrchestrator(copy.deepcopy(config))

    # Force both paths to execute hypothesizer/validator for parity with
    # legacy manual flow that always traverses those nodes.
    manual.detector.update = lambda X: (True, 1.0)
    langgraph.detector.update = lambda X: (True, 1.0)

    manual_loader = _make_loader(config)
    graph_loader = _make_loader(config)

    try:
        # Validator uses random noise; reseed each run for reproducibility.
        np.random.seed(12345)
        manual_results = manual.run(manual_loader)

        np.random.seed(12345)
        graph_results = langgraph.run_graph(graph_loader)

        manual_df = manual_results["results_df"]
        graph_df = graph_results["results_df"]

        assert len(manual_df) == 10, f"Expected 10 manual batches, got {len(manual_df)}"
        assert len(graph_df) == 10, f"Expected 10 graph batches, got {len(graph_df)}"

        manual_key = _results_keyframe(manual_df)
        graph_key = _results_keyframe(graph_df)

        if not manual_key.equals(graph_key):
            merged = manual_key.merge(
                graph_key,
                on="batch_id",
                suffixes=("_manual", "_langgraph"),
                how="outer",
            )
            raise AssertionError(
                "Manual and LangGraph results differ on key metrics.\n"
                f"Manual:\n{manual_key.to_string(index=False)}\n\n"
                f"LangGraph:\n{graph_key.to_string(index=False)}\n\n"
                f"Merged diff view:\n{merged.to_string(index=False)}"
            )
    finally:
        if hasattr(manual.prov, "close"):
            manual.prov.close()
        if hasattr(langgraph.prov, "close") and langgraph.prov is not manual.prov:
            langgraph.prov.close()


def _run_conditional_routing_tests(config: dict) -> None:
    """Verify low-drift skip and high-drift execution of hypothesizer node."""
    loader = _make_loader(config)
    X, y = loader.next_batch()
    if X is None or y is None:
        raise RuntimeError("Could not load first batch for conditional routing test")

    # Low drift case: branch should skip hypothesizer/validator.
    low_graph = LangGraphOrchestrator(copy.deepcopy(config))
    low_calls = {"hyp": 0, "val": 0}

    orig_low_hyp = low_graph.hypothesizer.run
    orig_low_val = low_graph.validator.run

    def low_hyp_wrapper(state: SystemState) -> SystemState:
        low_calls["hyp"] += 1
        return orig_low_hyp(state)

    def low_val_wrapper(state: SystemState) -> SystemState:
        low_calls["val"] += 1
        return orig_low_val(state)

    low_graph.hypothesizer.run = low_hyp_wrapper
    low_graph.validator.run = low_val_wrapper
    low_graph.detector.update = lambda X_: (False, 0.0)

    try:
        low_compiled = low_graph.compile()
        low_state = low_compiled.invoke(_build_state(0, X, y))

        assert low_calls["hyp"] == 0, "Low drift should skip hypothesizer"
        assert low_calls["val"] == 0, "Low drift should skip validator"
        assert low_state["hypotheses"] == [], "Low drift path should keep hypotheses empty"
    finally:
        if hasattr(low_graph.prov, "close"):
            low_graph.prov.close()

    # High drift case: branch should run hypothesizer/validator.
    high_graph = LangGraphOrchestrator(copy.deepcopy(config))
    high_calls = {"hyp": 0, "val": 0}

    orig_high_hyp = high_graph.hypothesizer.run
    orig_high_val = high_graph.validator.run

    def high_hyp_wrapper(state: SystemState) -> SystemState:
        high_calls["hyp"] += 1
        return orig_high_hyp(state)

    def high_val_wrapper(state: SystemState) -> SystemState:
        high_calls["val"] += 1
        return orig_high_val(state)

    high_graph.hypothesizer.run = high_hyp_wrapper
    high_graph.validator.run = high_val_wrapper
    high_graph.detector.update = lambda X_: (True, 1.0)

    try:
        high_compiled = high_graph.compile()
        _ = high_compiled.invoke(_build_state(0, X, y))

        assert high_calls["hyp"] > 0, "High drift should run hypothesizer"
        assert high_calls["val"] > 0, "High drift should run validator"
    finally:
        if hasattr(high_graph.prov, "close"):
            high_graph.prov.close()


def main() -> None:
    if not Path("config.yaml").exists():
        raise FileNotFoundError("config.yaml not found in current working directory")

    config = _load_config("config.yaml")

    # Prompt requirement: ensure all agents are created.
    _create_all_agents(config)

    # Prompt requirement: build/run LangGraph and compare with manual.
    _run_parity_test(config)

    # Prompt requirement: verify conditional routing behavior.
    _run_conditional_routing_tests(config)

    print("LangGraph produces same results as manual orchestrator")


if __name__ == "__main__":
    main()
