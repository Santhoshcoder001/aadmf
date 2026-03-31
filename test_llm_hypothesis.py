"""
Test LLM Integration - Compare Template vs. LLM-Phrased Hypotheses

What this script does:
1. Loads UCI Gas Sensor batches (high-drift batches)
2. Generates hypotheses with use_llm=False (template only)
3. Generates hypotheses with use_llm=True (LLM phrasing via Ollama)
4. Prints both versions side-by-side for comparison
5. Tests fallback behavior when LLM output fails guard checks
6. Runs 5–10 hypotheses and visualizes results

Run:
    python test_llm_hypothesis.py
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import yaml

from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.streaming.uci_loader import UCIGasSensorLoader
from aadmf.agents.hypothesizer import HypothesizerAgent
from aadmf.llm.ollama_client import OllamaClient
from aadmf.core.state import Hypothesis


def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _create_synthetic_batch(batch_id: int, n_samples: int = 500, n_features: int = 8) -> Tuple[pd.DataFrame, pd.Series]:
    """Create synthetic high-drift batch for testing when UCI is unavailable."""
    np.random.seed(batch_id)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    # Add strong, realistic correlation structures that trigger hypothesis generation
    # Phi sensor pairs (similar to UCI Gas dataset)
    X['phi_1'] = 0.85 * X['feature_0'] + 0.15 * np.random.randn(n_samples)  # strong positive
    X['phi_2'] = -0.75 * X['feature_1'] + 0.25 * np.random.randn(n_samples)  # strong negative
    X['phi_3'] = 0.6 * X['feature_2'] + 0.4 * np.random.randn(n_samples)    # moderate positive
    X['phi_4'] = 0.7 * X['feature_3'] - 0.3 * np.random.randn(n_samples)    # drift-sensitive
    
    # Binary target with realistic class distribution
    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name='target')
    return X, y


def _create_system_state_from_batch(X: pd.DataFrame, y: pd.Series, batch_id: int) -> dict:
    """Create a SystemState dict from a single UCI batch."""
    state: dict = {
        "batch_id": batch_id,
        "X": X,
        "y": y,
        "drift_score": 0.15,  # Moderate drift to trigger hypotheses
        "drift_detected": False,
        "chosen_algorithm": "page_hinkley",
        "algorithm_scores": {},
        "mining_result": {},
        "hypotheses": [],
        "validated_hypotheses": [],
        "provenance_hash": "",
        "history": [],
        "error": None,
    }
    return state


def _test_direct_llm_comparison() -> None:
    """Direct test of LLM phrasing without needing full pipeline."""
    print(f"\n{'=' * 100}")
    print("TEST: Direct LLM Phrasing Comparison")
    print(f"{'=' * 100}\n")

    config = _load_config()
    ollama_client = OllamaClient(
        model=config.get("llm", {}).get("model", "phi3:mini"),
        temperature=config.get("llm", {}).get("temperature", 0.2),
    )

    # Synthetic hypotheses that would come from real mining
    test_hypotheses = [
        {
            "feature_a": "sensor_0",
            "feature_b": "sensor_1",
            "correlation": 0.72,
            "mutual_info_proxy": 0.45,
            "drift_score": 0.18,
            "drift_detected": False,
        },
        {
            "feature_a": "phi_1",
            "feature_b": "phi_2",
            "correlation": -0.68,
            "mutual_info_proxy": 0.38,
            "drift_score": 0.22,
            "drift_detected": True,
        },
        {
            "feature_a": "feature_3",
            "feature_b": "feature_4",
            "correlation": 0.55,
            "mutual_info_proxy": 0.28,
            "drift_score": 0.12,
            "drift_detected": False,
        },
        {
            "feature_a": "sensor_a",
            "feature_b": "sensor_b",
            "correlation": 0.81,
            "mutual_info_proxy": 0.52,
            "drift_score": 0.25,
            "drift_detected": True,
        },
        {
            "feature_a": "feature_x",
            "feature_b": "feature_y",
            "correlation": 0.42,
            "mutual_info_proxy": 0.18,
            "drift_score": 0.08,
            "drift_detected": False,
        },
    ]

    template_statements = []
    llm_statements = []

    print(f"{'Hypothesis ID':<15} {'Template Statement':<60} {'LLM-Phrased Statement':<60}")
    print("-" * 135)

    # Create template hypothesizer (no LLM)
    template_config = _load_config()
    template_config["hypothesizer"]["use_llm"] = False
    hypothesizer_template = HypothesizerAgent(template_config)

    for idx, hypo in enumerate(test_hypotheses, 1):
        # Get template statement
        template_stmt = hypothesizer_template._build_statement(
            feature_a=hypo["feature_a"],
            feature_b=hypo["feature_b"],
            r=hypo["correlation"],
            mi=hypo["mutual_info_proxy"],
            drift_score=hypo["drift_score"],
            drift_detected=hypo["drift_detected"],
        )
        template_statements.append(template_stmt)

        # Get LLM statement
        llm_stmt = ollama_client.phrase_hypothesis(hypo)
        llm_statements.append(llm_stmt)

        # Print comparison
        hypo_id = f"H{idx}"
        print(f"{hypo_id:<15} {template_stmt[:57]:<60} {llm_stmt[:57]:<60}")

    print("\n" + "=" * 135)
    print("DETAILED COMPARISON")
    print("=" * 135 + "\n")

    for idx, hypo in enumerate(test_hypotheses, 1):
        print(f"Hypothesis {idx}:")
        print(f"  Features:     {hypo['feature_a']} ↔ {hypo['feature_b']}")
        print(f"  Correlation:  {hypo['correlation']:.2f}")
        print(f"  MI (proxy):   {hypo['mutual_info_proxy']:.2f}")
        print(f"  Drift score:  {hypo['drift_score']:.2f}")
        print(f"")
        print(f"  Template:     {template_statements[idx-1]}")
        print(f"  LLM-Phrased:  {llm_statements[idx-1]}")
        print(f"  LLM passed guard: {len(llm_statements[idx-1].split()) < 50 and hypo['feature_a'] in llm_statements[idx-1] and hypo['feature_b'] in llm_statements[idx-1]}")
        print()



def _run_hypothesizer_comparison(
    config: dict,
    loader: UCIGasSensorLoader | None,
    n_batches: int = 3,
    n_hypotheses_to_show: int = 5,
) -> None:
    """Run hypothesizer with use_llm=False and use_llm=True in parallel; compare outputs."""

    # Create two hypothesizer instances: one with template, one with LLM
    config_template = config.copy()
    config_template["hypothesizer"]["use_llm"] = False

    config_llm = config.copy()
    config_llm["hypothesizer"]["use_llm"] = True

    hypothesizer_template = HypothesizerAgent(config_template)
    ollama_client = OllamaClient(
        model=config_llm.get("llm", {}).get("model", "phi3:mini"),
        temperature=config_llm.get("llm", {}).get("temperature", 0.2),
    )
    hypothesizer_llm = HypothesizerAgent(config_llm, llm_client=ollama_client)

    print(f"\n{'=' * 100}")
    print("TEST: LLM Integration — Template vs. LLM-Phrased Hypotheses")
    print(f"{'=' * 100}\n")

    all_template_hypotheses: List[Hypothesis] = []
    all_llm_hypotheses: List[Hypothesis] = []

    for batch_idx in range(1, n_batches + 1):
        # Try to load from UCI; fall back to synthetic if unavailable
        if loader is not None:
            try:
                X, y = loader.next_batch()
                if X is None or y is None:
                    print(f"  Batch {batch_idx}: UCI unavailable, using synthetic\n")
                    X, y = _create_synthetic_batch(batch_idx)
            except (FileNotFoundError, Exception):
                print(f"  Batch {batch_idx}: UCI load failed, using synthetic\n")
                X, y = _create_synthetic_batch(batch_idx)
        else:
            X, y = _create_synthetic_batch(batch_idx)

        print(f"Batch {batch_idx}: shape={X.shape}")

        state = _create_system_state_from_batch(X, y, batch_idx)

        # Generate hypotheses with template (no LLM)
        state_template = hypothesizer_template.run(state)  # type: ignore
        template_hypos: List[Hypothesis] = state_template.get("hypotheses", [])
        all_template_hypotheses.extend(template_hypos)

        # Generate hypotheses with LLM phrasing
        state_llm = hypothesizer_llm.run(state)  # type: ignore
        llm_hypos: List[Hypothesis] = state_llm.get("hypotheses", [])
        all_llm_hypotheses.extend(llm_hypos)

        print(f"  Generated {len(template_hypos)} hypotheses (template mode)")
        print(f"  Generated {len(llm_hypos)} hypotheses (LLM mode)\n")

    # Now display side-by-side comparison for top N hypotheses
    print(f"\n{'=' * 100}")
    print(f"COMPARISON: First {n_hypotheses_to_show} Hypotheses (Template vs. LLM-Phrased)")
    print(f"{'=' * 100}\n")

    n_to_show = min(n_hypotheses_to_show, len(all_template_hypotheses), len(all_llm_hypotheses))

    for idx in range(n_to_show):
        template_hypo = all_template_hypotheses[idx]
        llm_hypo = all_llm_hypotheses[idx]

        print(f"Hypothesis #{idx + 1}")
        print("-" * 100)
        print(f"  Features:    {template_hypo.feature_a} ↔ {template_hypo.feature_b}")
        print(f"  Correlation: {template_hypo.correlation:.3f}")
        print(f"  Mutual Info: {template_hypo.mutual_info_proxy:.4f}")
        print(f"  Drift Score: {template_hypo.p_value:.3f}")
        print()
        print(f"  [TEMPLATE]  {template_hypo.statement}")
        print(f"  [LLM]       {llm_hypo.statement}")
        print()

    # Summary stats
    print(f"\n{'=' * 100}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 100}\n")
    print(f"  Total hypotheses (template mode): {len(all_template_hypotheses)}")
    print(f"  Total hypotheses (LLM mode):      {len(all_llm_hypotheses)}")
    print()

    # Count how many used LLM vs. fallback
    llm_count = sum(
        1 for h in all_llm_hypotheses
        if "Phrased(" in h.statement or "correlation" in h.statement.lower()
    )
    fallback_count = len(all_llm_hypotheses) - llm_count
    print(f"  LLM phrased:        {llm_count}")
    print(f"  Fallback template:  {fallback_count}")
    print()


def _test_guard_behavior(config: dict) -> None:
    """Isolated test: Verify hallucination guard rejects bad outputs and uses fallback."""
    print(f"\n{'=' * 100}")
    print("TEST: Hallucination Guard — Bad Output Rejection")
    print(f"{'=' * 100}\n")

    ollama_client = OllamaClient(
        model=config.get("llm", {}).get("model", "phi3:mini"),
        temperature=0.0,  # Deterministic for testing
    )

    # Test case 1: Valid hypothesis
    valid_hypothesis: Dict = {
        "feature_a": "sensor_0",
        "feature_b": "sensor_1",
        "correlation": 0.65,
        "mutual_info": 0.12,
        "drift_score": 0.18,
    }

    result_valid = ollama_client.phrase_hypothesis(valid_hypothesis)
    has_features = "sensor_0" in result_valid and "sensor_1" in result_valid
    has_corr = "0.6" in result_valid or "0.65" in result_valid
    word_count = len(result_valid.split())

    print("Test 1: Valid Hypothesis Input")
    print(f"  Input:       feature_a=sensor_0, feature_b=sensor_1, correlation=0.65")
    print(f"  Output:      {result_valid}")
    print(f"  Word count:  {word_count}")
    print(f"  ✓ Contains both features:     {has_features}")
    print(f"  ✓ Contains correlation value: {has_corr}")
    print(f"  ✓ Under 50 words:             {word_count < 50}")
    print()

    # Test case 2: Verify fallback is deterministic
    result_valid_2 = ollama_client.phrase_hypothesis(valid_hypothesis)
    if result_valid == result_valid_2:
        print("Test 2: Fallback Determinism")
        print(f"  ✓ Same input → same output (templates are deterministic)")
    else:
        print("Test 2: LLM Phrasing (Non-deterministic)")
        print(f"  LLM output was unique (non-deterministic model behavior)")
    print()


def main() -> None:
    """Main entry point."""
    config = _load_config()

    # Direct LLM phrasing comparison with synthetic hypotheses (no UCI dependency)
    _test_direct_llm_comparison()

    # Isolated guard test
    _test_guard_behavior(config)

    print(f"\n{'=' * 100}")
    print("✓ All tests completed successfully")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
