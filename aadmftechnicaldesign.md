# AADMF — Technical Design Document (TDD)
## Agentic Adaptive Data Mining Framework

**Version:** 1.0 | **Date:** March 31, 2026
**Document:** 1 of 5 — Technical Design & Architecture

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Decisions & Justifications](#2-technology-decisions--justifications)
3. [Complete Folder & File Structure](#3-complete-folder--file-structure)
4. [Class Design — All Agents](#4-class-design--all-agents)
5. [Data Flow & State Management](#5-data-flow--state-management)
6. [Database Schema — Provenance Graph](#6-database-schema--provenance-graph)
7. [Configuration Management](#7-configuration-management)
8. [Error Handling Strategy](#8-error-handling-strategy)

---

## 1. System Overview

### 1.1 Architecture Philosophy

AADMF is built on three design principles:

- **Agent Isolation** — each agent is a self-contained Python class. No agent directly calls another. All communication goes through a shared `SystemState` object managed by the Orchestrator.
- **Layered Complexity** — PoC (Month 1) uses pure Python; Full system (Month 3+) adds LangGraph and Neo4j on top of the same base classes. You never rewrite — you extend.
- **Reproducibility First** — every random operation uses a seeded RNG. Every decision is logged before it executes, not after.

### 1.2 Two-Phase Build Strategy

```
PHASE 1 (Months 1–2): Pure Python — No external agent framework
─────────────────────────────────────────────────────────────────
StreamingSimulator → DriftDetector → PlannerAgent → MinerAgent
       → HypothesizerAgent → ValidatorAgent → ProvenanceLogger
       → Orchestrator (manual loop) → CLI output

PHASE 2 (Months 3–5): LangGraph + Neo4j layered on top
─────────────────────────────────────────────────────────────────
Same agents, now wrapped as LangGraph nodes
Provenance dict-chain replaced by Neo4j graph
Streamlit dashboard added on top
LLM (Ollama/Phi-3) added to Hypothesizer only
```

---

## 2. Technology Decisions & Justifications

### 2.1 Programming Language — Python 3.11+

| Criterion | Decision | Reason |
|---|---|---|
| Language | Python 3.11+ | Best ML ecosystem; all target libraries support it |
| Version minimum | 3.11 | Required by LangGraph stable; type hints performance |
| Package manager | pip + venv | Simpler than conda for reproducibility |

### 2.2 Drift Detection — Page-Hinkley Test (Hand-coded)

**Why Page-Hinkley over alternatives:**

| Algorithm | Complexity | Detects | Why Not Chosen |
|---|---|---|---|
| ADWIN | High | Variance + mean shifts | Requires `scikit-multiflow`; complex to tune |
| DDM (Drift Detection Method) | Medium | Error-rate drift | Needs labelled stream; we are unsupervised |
| KSWIN | High | Distribution shift | Computationally heavy for real-time |
| **Page-Hinkley** | **Low** | **Mean shift** | **Pure Python, hand-coded, patentable as custom impl** |
| Kolmogorov-Smirnov | Medium | Distribution | Batch-only, not sequential |

**Your custom enhancement over standard PH:**
Standard PH uses a fixed mean. Your version uses an exponentially weighted moving mean (`alpha=0.9999`), making it more robust to gradual drift. This is a documented novelty.

### 2.3 Agent Framework

**Phase 1: Manual Orchestrator (pure Python)**
- Zero dependencies
- Full control over state
- Easier to debug
- Sufficient for PoC and patent evidence

**Phase 2: LangGraph**

| Framework | Why Considered | Why LangGraph Won |
|---|---|---|
| LangChain AgentExecutor | Simple but no graph state | Cannot model complex agent dependencies |
| AutoGen | Multi-agent but heavyweight | Too much overhead; hard to control |
| CrewAI | Easy setup | Less control over state; newer/less stable |
| **LangGraph** | **Graph-based state machine** | **Explicit state, cyclable nodes, built-in memory** |

**LangGraph key concepts used:**
- `StateGraph` — defines agent nodes and edges
- `TypedDict` — typed system state passed between nodes
- `ToolNode` — wraps miner algorithms as tools
- Conditional edges — Planner → different Miner nodes based on score

### 2.4 Mining Algorithms

| Algorithm | Library | Use Case | Parameters You Tune |
|---|---|---|---|
| IsolationForest | scikit-learn | Anomaly detection under drift | `contamination` (adaptive) |
| DBSCAN | scikit-learn | Density clustering in shifted data | `eps` (adaptive to drift score) |
| Apriori (custom) | mlxtend or hand-coded | Association rules in binarised data | `min_support`, `min_confidence` |
| StatisticalRules | hand-coded (scipy) | Fast correlation rules (PoC) | `corr_threshold`, `mi_threshold` |

**Why not deep learning:** Adds GPU dependency and training time. Your novelty is the *agent orchestration*, not the mining algorithm itself. Shallow algorithms make ablation studies cleaner.

### 2.5 Hypothesis Validation — scipy.stats

| Test | Purpose | When Used |
|---|---|---|
| `pearsonr` | Linear correlation between two sensors | Hypothesis generation trigger |
| `chi2_contingency` | Independence test on binarised features | Primary hypothesis validation |
| `spearmanr` | Non-linear rank correlation | Secondary validation (Phase 2) |
| `mutual_info_score` | True MI calculation (Phase 2) | Replaces MI proxy in full version |

### 2.6 Provenance Storage

| Phase | Technology | Reason |
|---|---|---|
| Phase 1 (PoC) | Python dict + hashlib SHA-256 | Zero dependencies; proves concept |
| Phase 2 (Full) | Neo4j Community Edition | Graph queries; visual provenance; industry standard |

**Neo4j driver:** `neo4j` Python driver (official)
**Neo4j version:** 5.x Community (free, local install)

### 2.7 LLM Integration (Phase 2 only)

| Option | RAM Required | Speed | Why |
|---|---|---|---|
| Ollama + Phi-3 Mini (3.8B) | ~4GB | Fast | Best quality/RAM tradeoff on laptop |
| Ollama + TinyLlama (1.1B) | ~2GB | Fastest | Fallback if RAM constrained |
| Grok API | 0 (cloud) | Moderate | Alternative if local model too slow |
| GPT-4o | 0 (cloud) | Fast | Paid; not required for patent |

**LLM role is NARROW:** Only used to convert hypothesis dict → natural language sentence. The statistical logic remains yours. This is important for patent claims.

### 2.8 Dashboard — Streamlit

- `streamlit` — main framework
- `plotly` — interactive charts (drift score line chart, algorithm distribution)
- `networkx` + `pyvis` — provenance graph visualisation
- `streamlit-autorefresh` — live polling for new batches

---

## 3. Complete Folder & File Structure

```
aadmf/
│
├── README.md                    # Project overview + quick start
├── requirements.txt             # PoC dependencies
├── requirements_full.txt        # Full system dependencies
├── config.yaml                  # All tunable parameters
├── .env.example                 # Neo4j / API keys template
│
├── poc.py                       # Month 1: single-file PoC runner
│
├── aadmf/                       # Main package
│   ├── __init__.py
│   │
│   ├── core/                    # Core data structures
│   │   ├── __init__.py
│   │   ├── state.py             # SystemState TypedDict
│   │   ├── hypothesis.py        # Hypothesis dataclass
│   │   └── event.py             # ProvenanceEvent dataclass
│   │
│   ├── streaming/               # Layer 1: Data ingestion
│   │   ├── __init__.py
│   │   ├── simulator.py         # StreamingSimulator class
│   │   ├── uci_loader.py        # Real Gas Sensor dataset loader
│   │   └── base.py              # BaseStreamer abstract class
│   │
│   ├── agents/                  # Layer 2: All agents
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent abstract class
│   │   ├── planner.py           # PlannerAgent + scoring matrix
│   │   ├── miner.py             # MinerAgent (all 3 algorithms)
│   │   ├── hypothesizer.py      # HypothesizerAgent + your template
│   │   ├── validator.py         # ValidatorAgent
│   │   └── provenance.py        # ProvenanceLogger
│   │
│   ├── mining/                  # Layer 3: Mining algorithm wrappers
│   │   ├── __init__.py
│   │   ├── base.py              # BaseMiner abstract class
│   │   ├── isolation_forest.py  # IsolationForest wrapper
│   │   ├── dbscan.py            # DBSCAN wrapper
│   │   ├── statistical_rules.py # StatisticalRules (hand-coded)
│   │   └── apriori.py           # Apriori wrapper (Phase 2)
│   │
│   ├── drift/                   # Drift detection module
│   │   ├── __init__.py
│   │   ├── page_hinkley.py      # Your custom PH implementation
│   │   └── base.py              # BaseDriftDetector
│   │
│   ├── provenance/              # Layer 6: Provenance backends
│   │   ├── __init__.py
│   │   ├── dict_chain.py        # Phase 1: dict + SHA-256
│   │   └── neo4j_graph.py       # Phase 2: Neo4j backend
│   │
│   ├── orchestrator/            # Agent orchestration
│   │   ├── __init__.py
│   │   ├── manual.py            # Phase 1: manual Python loop
│   │   └── langgraph_flow.py    # Phase 2: LangGraph StateGraph
│   │
│   ├── dashboard/               # Layer 7: Streamlit UI
│   │   ├── __init__.py
│   │   ├── app.py               # Main Streamlit app entry point
│   │   ├── charts.py            # Plotly chart builders
│   │   └── graph_viz.py         # Provenance graph visualiser
│   │
│   └── evaluation/              # Experiment runner
│       ├── __init__.py
│       ├── ablation.py          # Ablation study runner
│       ├── metrics.py           # All metric calculations
│       └── baseline.py          # Static pipeline baseline
│
├── data/
│   ├── raw/                     # UCI Gas Sensor batches (gitignored)
│   ├── processed/               # Pre-processed batches
│   └── synthetic/               # Generated synthetic datasets
│
├── experiments/
│   ├── configs/                 # Experiment config YAML files
│   ├── results/                 # Saved experiment results (JSON/CSV)
│   └── notebooks/               # Jupyter analysis notebooks
│
└── tests/
    ├── unit/                    # Unit tests per module
    ├── integration/             # End-to-end integration tests
    └── fixtures/                # Shared test data
```

---

## 4. Class Design — All Agents

### 4.1 SystemState (shared between all agents)

```python
# aadmf/core/state.py

from typing import TypedDict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Hypothesis:
    id: str
    batch: int
    feature_a: str
    feature_b: str
    correlation: float
    mutual_info_proxy: float
    p_value: float
    drift_triggered: bool
    statement: str
    valid: Optional[bool] = None
    confidence: Optional[str] = None
    p_value_chi2: Optional[float] = None

@dataclass
class BatchResult:
    batch_id: int
    drift_score: float
    drift_detected: bool
    algorithm: str
    algo_scores: dict
    quality_score: float
    hypotheses: List[Hypothesis] = field(default_factory=list)
    validated: List[Hypothesis] = field(default_factory=list)

class SystemState(TypedDict):
    """Shared state passed between all agents in LangGraph (Phase 2)."""
    batch_id: int
    X: object                    # current DataFrame
    y: object                    # current labels
    drift_score: float
    drift_detected: bool
    chosen_algorithm: str
    algorithm_scores: dict
    mining_result: dict
    hypotheses: List[Hypothesis]
    validated_hypotheses: List[Hypothesis]
    provenance_hash: str
    history: List[BatchResult]
    error: Optional[str]
```

### 4.2 BaseAgent

```python
# aadmf/agents/base.py

from abc import ABC, abstractmethod
from aadmf.core.state import SystemState

class BaseAgent(ABC):
    """
    All agents inherit from BaseAgent.
    Enforces: single responsibility, config-driven thresholds,
    logging to provenance before returning.
    """

    def __init__(self, config: dict, provenance_logger=None):
        self.config = config
        self.prov = provenance_logger

    @abstractmethod
    def run(self, state: SystemState) -> SystemState:
        """
        Takes current state, performs single responsibility,
        returns updated state. Never mutates state in place.
        """
        pass

    def _log(self, event_type: str, details: dict, state: SystemState):
        if self.prov:
            h = self.prov.log(event_type, details)
            state["provenance_hash"] = h
```

### 4.3 PlannerAgent — Full Implementation Design

```python
# aadmf/agents/planner.py

class PlannerAgent(BaseAgent):
    """
    YOUR ORIGINAL LOGIC: drift-weighted scoring matrix
    with exponential moving average accuracy feedback.

    Patent Claim 1 core implementation.
    """

    ALGORITHM_REGISTRY = {
        "IsolationForest":  {"drift_weight": 0.9, "cost": 0.3},
        "DBSCAN":           {"drift_weight": 0.6, "cost": 0.4},
        "StatisticalRules": {"drift_weight": 0.4, "cost": 0.1},
        "Apriori":          {"drift_weight": 0.3, "cost": 0.5},  # Phase 2
    }

    def __init__(self, config, provenance_logger=None):
        super().__init__(config, provenance_logger)
        self.alpha = config.get("ema_alpha", 0.7)
        self.weights = config.get("scoring_weights", {
            "drift": 0.4, "accuracy": 0.3, "cost": 0.3
        })
        # Initialised from base_accuracy in registry
        self.accuracy_history = {
            k: 0.7 for k in self.ALGORITHM_REGISTRY
        }
        self.selection_log = []

    def _score(self, algo: str, drift_score: float) -> float:
        props = self.ALGORITHM_REGISTRY[algo]
        acc   = self.accuracy_history[algo]
        return (
            self.weights["drift"]    * props["drift_weight"] * drift_score
          + self.weights["accuracy"] * acc
          + self.weights["cost"]     * (1.0 - props["cost"])
        )

    def run(self, state: SystemState) -> SystemState:
        drift = state["drift_score"]
        scores = {
            algo: round(self._score(algo, drift), 4)
            for algo in self.ALGORITHM_REGISTRY
        }
        chosen = max(scores, key=scores.__getitem__)
        state["chosen_algorithm"] = chosen
        state["algorithm_scores"] = scores
        self._log("ALGO_SELECTED", {
            "batch": state["batch_id"],
            "algorithm": chosen,
            "scores": scores,
            "drift_score": drift
        }, state)
        return state

    def update_accuracy(self, algo: str, quality_score: float):
        old = self.accuracy_history[algo]
        self.accuracy_history[algo] = round(
            self.alpha * old + (1 - self.alpha) * quality_score, 4
        )
```

### 4.4 MinerAgent — Full Implementation Design

```python
# aadmf/agents/miner.py

class MinerAgent(BaseAgent):
    """
    Dispatches to the correct mining algorithm wrapper.
    Returns quality_score for feedback to PlannerAgent.
    """

    def __init__(self, config, provenance_logger=None):
        super().__init__(config, provenance_logger)
        # Import mining wrappers
        from aadmf.mining.isolation_forest import IFMiner
        from aadmf.mining.dbscan import DBSCANMiner
        from aadmf.mining.statistical_rules import StatRulesMiner

        self._miners = {
            "IsolationForest":  IFMiner(config),
            "DBSCAN":           DBSCANMiner(config),
            "StatisticalRules": StatRulesMiner(config),
        }

    def run(self, state: SystemState) -> SystemState:
        algo   = state["chosen_algorithm"]
        miner  = self._miners[algo]
        result = miner.mine(state["X"], state["drift_score"])
        state["mining_result"] = result
        self._log("MINING_RESULT", {
            "batch": state["batch_id"], **result
        }, state)
        return state
```

### 4.5 HypothesizerAgent — Full Implementation Design

```python
# aadmf/agents/hypothesizer.py

class HypothesizerAgent(BaseAgent):
    """
    YOUR ORIGINAL LOGIC — Patent Claim 2 core.

    Template: IF correlation(i,j) > threshold
              AND MI_proxy > threshold
              AND drift triggered
              THEN generate hypothesis statement.

    MI proxy = |r| × 0.5  (Phase 1)
    Real MI  = mutual_info_score(binarised_i, binarised_j)  (Phase 2)
    """

    def __init__(self, config, provenance_logger=None, llm=None):
        super().__init__(config, provenance_logger)
        self.corr_threshold = config.get("corr_threshold", 0.3)
        self.mi_threshold   = config.get("mi_threshold", 0.05)
        self.n_features_check = config.get("n_features_check", 8)
        self.max_per_batch  = config.get("max_hypotheses_per_batch", 3)
        self.llm = llm  # Optional: Ollama Phi-3 for Phase 2

    def _mi_proxy(self, x, y) -> float:
        """Phase 1: approximation. Phase 2: replace with real MI."""
        from scipy.stats import pearsonr
        r, _ = pearsonr(x, y)
        return abs(r) * 0.5

    def _phrase(self, h: dict) -> str:
        """
        Phase 1: template string.
        Phase 2: pass to LLM for natural language phrasing.
        """
        if self.llm:
            return self.llm.phrase_hypothesis(h)
        condition = "under drift" if h["drift_triggered"] else "in stable regime"
        return (
            f"{h['feature_a']} and {h['feature_b']} show co-pattern "
            f"(r={h['correlation']:.2f}, MI≈{h['mutual_info_proxy']:.2f}) "
            f"{condition} [drift_score={h['drift_score']:.4f}] "
            f"→ investigate combined feature for gas classification"
        )

    def run(self, state: SystemState) -> SystemState:
        from scipy.stats import pearsonr
        X = state["X"]
        cols = X.columns.tolist()
        n = min(self.n_features_check, len(cols))
        drift_score = state["drift_score"]
        drift_detected = state["drift_detected"]

        # Trigger condition
        if drift_score <= 0.1 and not drift_detected:
            state["hypotheses"] = []
            return state

        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = X.iloc[:, i].values, X.iloc[:, j].values
                r, p = pearsonr(xi, xj)
                mi = self._mi_proxy(xi, xj)
                if abs(r) > self.corr_threshold and mi > self.mi_threshold:
                    h = {
                        "id": f"H{state['batch_id']}_{i}_{j}",
                        "batch": state["batch_id"],
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(r), 3),
                        "mutual_info_proxy": round(float(mi), 3),
                        "p_value": round(float(p), 4),
                        "drift_triggered": drift_detected,
                        "drift_score": drift_score,
                    }
                    h["statement"] = self._phrase(h)
                    candidates.append(h)

        # Rank by |r|, take top N
        candidates.sort(key=lambda h: abs(h["correlation"]), reverse=True)
        top = candidates[:self.max_per_batch]
        state["hypotheses"] = top

        self._log("HYPOTHESES_GENERATED", {
            "batch": state["batch_id"],
            "count": len(top),
            "ids": [h["id"] for h in top]
        }, state)
        return state
```

### 4.6 ValidatorAgent — Full Implementation Design

```python
# aadmf/agents/validator.py

class ValidatorAgent(BaseAgent):
    """
    Validates hypotheses via:
    1. Chi-square contingency test on real data
    2. Repeat on Gaussian-noisy synthetic copy (±noise_pct)
    3. Assigns confidence: HIGH / MEDIUM / LOW
    """

    def __init__(self, config, provenance_logger=None):
        super().__init__(config, provenance_logger)
        self.p_threshold_real  = config.get("p_threshold_real", 0.05)
        self.p_threshold_noisy = config.get("p_threshold_noisy", 0.10)
        self.noise_pct         = config.get("noise_pct", 0.05)

    def _validate_one(self, hyp: dict, X) -> dict:
        from scipy.stats import chi2_contingency
        import pandas as pd, numpy as np

        fa, fb = hyp["feature_a"], hyp["feature_b"]
        if fa not in X.columns or fb not in X.columns:
            return {**hyp, "valid": False, "reason": "feature_missing"}

        a = (X[fa] > X[fa].median()).astype(int)
        b = (X[fb] > X[fb].median()).astype(int)

        # Real data chi-square
        ct = pd.crosstab(a, b)
        chi2, p_real, _, _ = chi2_contingency(ct)

        # Synthetic augmentation
        noise = np.random.uniform(
            1 - self.noise_pct, 1 + self.noise_pct, len(X)
        )
        a_n = (X[fa] * noise > X[fa].median()).astype(int)
        b_n = (X[fb] * noise > X[fb].median()).astype(int)
        _, p_noisy, _, _ = chi2_contingency(pd.crosstab(a_n, b_n))

        valid = (p_real < self.p_threshold_real and
                 p_noisy < self.p_threshold_noisy)
        conf  = ("HIGH"   if p_real < 0.01 else
                 "MEDIUM" if p_real < 0.05 else "LOW")

        return {
            **hyp,
            "chi2": round(float(chi2), 3),
            "p_value_chi2": round(float(p_real), 4),
            "p_value_synthetic": round(float(p_noisy), 4),
            "valid": valid,
            "confidence": conf,
        }

    def run(self, state: SystemState) -> SystemState:
        validated = []
        for hyp in state.get("hypotheses", []):
            v = self._validate_one(hyp, state["X"])
            validated.append(v)
            self._log("HYPOTHESIS_VALIDATED", {
                "batch": state["batch_id"],
                "id": v["id"],
                "valid": v["valid"],
                "p": v.get("p_value_chi2"),
                "confidence": v.get("confidence")
            }, state)
        state["validated_hypotheses"] = validated
        return state
```

### 4.7 ProvenanceLogger — Full Implementation Design

```python
# aadmf/provenance/dict_chain.py  (Phase 1)

class DictChainLogger:
    """
    YOUR ORIGINAL LOGIC — Patent Claim 3 core.
    SHA-256 hash-chained event ledger.
    Zero external dependencies.
    """

    def __init__(self):
        self.chain = []
        self.prev_hash = "GENESIS"
        self._index = {}  # type_str → [seq_ids]

    def _compute_hash(self, event: dict) -> str:
        import hashlib, json
        payload = json.dumps(
            {k: v for k, v in event.items() if k != "hash"},
            sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def log(self, event_type: str, details: dict) -> str:
        import time
        event = {
            "seq":       len(self.chain),
            "type":      event_type,
            "ts":        time.time(),
            "details":   details,
            "prev_hash": self.prev_hash,
        }
        h = self._compute_hash(event)
        event["hash"] = h
        self.chain.append(event)
        self.prev_hash = h
        self._index.setdefault(event_type, []).append(event["seq"])
        return h

    def verify_integrity(self) -> tuple:
        prev = "GENESIS"
        for i, event in enumerate(self.chain):
            if event["prev_hash"] != prev:
                return False, i
            if self._compute_hash(event) != event["hash"]:
                return False, i
            prev = event["hash"]
        return True, -1

    def query_by_type(self, event_type: str) -> list:
        seqs = self._index.get(event_type, [])
        return [self.chain[s] for s in seqs]

    def export_json(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.chain, f, indent=2, default=str)

    def summary(self) -> dict:
        by_type = {k: len(v) for k, v in self._index.items()}
        return {
            "total_events": len(self.chain),
            "by_type": by_type,
            "first_hash": self.chain[0]["hash"] if self.chain else None,
            "last_hash": self.chain[-1]["hash"] if self.chain else None,
        }
```

---

## 5. Data Flow & State Management

### 5.1 Phase 1 — Manual Orchestrator Flow

```
START
  │
  ▼
StreamingSimulator.next_batch()
  │  returns (X: DataFrame, y: Series)
  ▼
DriftDetector.update(X)
  │  returns (drift_detected: bool, drift_score: float)
  ▼
PlannerAgent.run(state)
  │  reads: drift_score
  │  writes: chosen_algorithm, algorithm_scores
  │  logs: ALGO_SELECTED event
  ▼
MinerAgent.run(state)
  │  reads: chosen_algorithm, X
  │  writes: mining_result (quality_score + algo details)
  │  logs: MINING_RESULT event
  ▼
PlannerAgent.update_accuracy(algo, quality_score)
  │  updates: accuracy_history EMA
  ▼
[IF drift_score > 0.1 OR drift_detected]
  │
  ▼
HypothesizerAgent.run(state)
  │  reads: X, drift_score, drift_detected
  │  writes: hypotheses list
  │  logs: HYPOTHESES_GENERATED event
  ▼
ValidatorAgent.run(state)
  │  reads: hypotheses, X
  │  writes: validated_hypotheses
  │  logs: HYPOTHESIS_VALIDATED per hypothesis
  ▼
Append BatchResult to history
  │
  ▼
[NEXT BATCH or END]
```

### 5.2 Phase 2 — LangGraph State Graph

```python
# aadmf/orchestrator/langgraph_flow.py

from langgraph.graph import StateGraph, END
from aadmf.core.state import SystemState

def build_graph(agents: dict) -> StateGraph:
    graph = StateGraph(SystemState)

    # Add nodes
    graph.add_node("drift_detect",  agents["drift"].run)
    graph.add_node("plan",          agents["planner"].run)
    graph.add_node("mine",          agents["miner"].run)
    graph.add_node("hypothesize",   agents["hypothesizer"].run)
    graph.add_node("validate",      agents["validator"].run)
    graph.add_node("log_batch",     agents["batch_logger"].run)

    # Define edges
    graph.set_entry_point("drift_detect")
    graph.add_edge("drift_detect", "plan")
    graph.add_edge("plan", "mine")

    # Conditional: only hypothesize if drift > threshold
    graph.add_conditional_edges(
        "mine",
        lambda s: "hypothesize" if s["drift_score"] > 0.1 else "log_batch",
        {"hypothesize": "hypothesize", "log_batch": "log_batch"}
    )
    graph.add_edge("hypothesize", "validate")
    graph.add_edge("validate", "log_batch")
    graph.add_edge("log_batch", END)

    return graph.compile()
```

### 5.3 State Transition Table

| Agent | Reads from State | Writes to State |
|---|---|---|
| DriftDetector | `X` | `drift_score`, `drift_detected` |
| PlannerAgent | `drift_score`, `accuracy_history` (internal) | `chosen_algorithm`, `algorithm_scores` |
| MinerAgent | `chosen_algorithm`, `X`, `drift_score` | `mining_result` |
| HypothesizerAgent | `X`, `drift_score`, `drift_detected`, `batch_id` | `hypotheses` |
| ValidatorAgent | `hypotheses`, `X` | `validated_hypotheses` |
| ProvenanceLogger | All (called inside each agent) | `provenance_hash` |

---

## 6. Database Schema — Provenance Graph

### 6.1 Phase 1: Dict Chain Node Schema

```json
{
  "seq":       0,
  "type":      "ALGO_SELECTED",
  "ts":        1743394800.123,
  "details": {
    "batch":      5,
    "algorithm":  "IsolationForest",
    "scores":     {"IsolationForest": 0.51, "DBSCAN": 0.38, "StatisticalRules": 0.39},
    "drift_score": 0.42
  },
  "prev_hash": "4d6a8c909bdbb7ca",
  "hash":      "a1b2c3d4e5f6a7b8"
}
```

### 6.2 Phase 2: Neo4j Graph Schema

**Node types:**

```cypher
// Batch node
CREATE (b:Batch {
  id: 5,
  rows: 150,
  drift_score: 0.42,
  drift_detected: true,
  timestamp: datetime()
})

// Decision node
CREATE (d:Decision {
  type: "ALGO_SELECTED",
  algorithm: "IsolationForest",
  score: 0.51,
  hash: "a1b2c3d4e5f6a7b8",
  prev_hash: "4d6a8c909bdbb7ca"
})

// Hypothesis node
CREATE (h:Hypothesis {
  id: "H5_0_3",
  feature_a: "sensor_0",
  feature_b: "sensor_3",
  correlation: 0.42,
  valid: true,
  confidence: "HIGH"
})

// MiningResult node
CREATE (m:MiningResult {
  algorithm: "IsolationForest",
  anomalies: 12,
  quality_score: 0.92
})
```

**Relationship types:**

```cypher
// Batch → caused → Decision
MATCH (b:Batch {id: 5}), (d:Decision {type: "ALGO_SELECTED"})
CREATE (b)-[:TRIGGERED]->(d)

// Decision → produced → MiningResult
MATCH (d:Decision), (m:MiningResult)
CREATE (d)-[:PRODUCED]->(m)

// Batch → generated → Hypothesis
MATCH (b:Batch {id: 5}), (h:Hypothesis {id: "H5_0_3"})
CREATE (b)-[:GENERATED]->(h)

// Chain link (tamper-proof)
MATCH (e1:Decision {hash: "prev"}), (e2:Decision {prev_hash: "prev"})
CREATE (e1)-[:PRECEDES]->(e2)
```

---

## 7. Configuration Management

```yaml
# config.yaml — single source of truth for all parameters

streaming:
  n_batches: 10
  batch_size: 150
  n_features: 16
  drift_after: 4
  seed: 42

drift_detection:
  algorithm: "page_hinkley"
  delta: 0.005
  threshold: 50.0
  alpha: 0.9999

planner:
  scoring_weights:
    drift: 0.4
    accuracy: 0.3
    cost: 0.3
  ema_alpha: 0.7

hypothesizer:
  corr_threshold: 0.3
  mi_threshold: 0.05
  n_features_check: 8
  max_hypotheses_per_batch: 3
  trigger_drift_score: 0.1
  use_llm: false          # set true in Phase 2

validator:
  p_threshold_real: 0.05
  p_threshold_noisy: 0.10
  noise_pct: 0.05

mining:
  isolation_forest:
    base_contamination: 0.1
    adaptive: true        # contamination = max(0.05, 0.15 * drift_score)
    random_state: 42
  dbscan:
    eps: 1.5
    min_samples: 5
    adaptive_eps: true
  statistical_rules:
    corr_threshold: 0.3
    top_n_features: 4

provenance:
  backend: "dict"         # "dict" (Phase 1) or "neo4j" (Phase 2)
  export_json: true
  export_path: "experiments/results/provenance.json"

neo4j:                    # Phase 2 only
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"

llm:                      # Phase 2 only
  provider: "ollama"
  model: "phi3:mini"
  temperature: 0.2

evaluation:
  n_runs: 5
  drift_levels: [0, 2, 4, 6, 8]
  ablation_variants:
    - "full"
    - "no_scoring_matrix"
    - "no_hypothesizer"
    - "no_provenance"
    - "static_baseline"
```

---

## 8. Error Handling Strategy

### 8.1 Agent-Level Errors

Each agent wraps its `run()` in a try-except. On error:
- Logs `AGENT_ERROR` event to provenance
- Sets `state["error"]` message
- Returns state (does not crash the pipeline)
- Orchestrator checks `state["error"]` after each agent

### 8.2 Graceful Degradation Table

| Failure | System Response |
|---|---|
| Empty batch from simulator | Skip batch; log `BATCH_SKIPPED` event |
| Chi-square fails (degenerate contingency) | Mark hypothesis as `valid=False, reason="degenerate_table"` |
| LLM timeout (Phase 2) | Fall back to template-string phrasing |
| Neo4j connection error (Phase 2) | Fall back to dict-chain logger; log warning |
| IsolationForest contamination error | Clamp to [0.01, 0.49] |

### 8.3 Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aadmf.log")
    ]
)
```

All agents use `self.logger = logging.getLogger(self.__class__.__name__)`.

---

*AADMF — Technical Design Document v1.0 | Document 1 of 5*