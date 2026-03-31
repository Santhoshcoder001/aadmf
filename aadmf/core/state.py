"""
SystemState — shared state passed between all agents.
BatchResult — stores result of one complete batch processing cycle.
Hypothesis — dataclass for a single generated hypothesis.
"""

from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    """Represents one generated hypothesis from the HypothesizerAgent."""
    id: str
    batch: int
    feature_a: str
    feature_b: str
    correlation: float
    mutual_info_proxy: float
    p_value: float
    drift_triggered: bool
    statement: str
    # Set by ValidatorAgent
    valid: Optional[bool] = None
    confidence: Optional[str] = None     # "HIGH" / "MEDIUM" / "LOW"
    p_value_chi2: Optional[float] = None
    p_value_synthetic: Optional[float] = None
    chi2: Optional[float] = None


@dataclass
class BatchResult:
    """Summary of one batch processing cycle. Stored in history."""
    batch_id: int
    drift_score: float
    drift_detected: bool
    algorithm: str
    algo_scores: Dict[str, float]
    quality_score: float
    n_hypotheses: int = 0
    n_valid_hypotheses: int = 0
    runtime_ms: float = 0.0
    provenance_hash: str = ""


class SystemState(TypedDict):
    """
    Shared mutable state passed between all agents.
    In Phase 1: passed as plain dict through manual orchestrator.
    In Phase 2: used as LangGraph TypedDict state.
    """
    # Current batch data
    batch_id: int
    X: Any                           # pd.DataFrame — current batch features
    y: Any                           # pd.Series — current batch labels

    # Drift detection output
    drift_score: float               # 0.0 to 1.0
    drift_detected: bool

    # Planner output
    chosen_algorithm: str
    algorithm_scores: Dict[str, float]

    # Miner output
    mining_result: Dict[str, Any]    # {algorithm, quality_score, ...algo-specific}

    # Hypothesizer output
    hypotheses: List[Hypothesis]

    # Validator output
    validated_hypotheses: List[Hypothesis]

    # Provenance
    provenance_hash: str             # hash of last logged event

    # History
    history: List[BatchResult]

    # Error handling
    error: Optional[str]
