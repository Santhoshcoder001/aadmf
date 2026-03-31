"""Unit tests for PlannerAgent scoring matrix."""

from aadmf.agents.planner import PlannerAgent


def make_state(drift_score: float) -> dict:
    return {
        "batch_id": 0,
        "drift_score": drift_score,
        "drift_detected": drift_score > 0.1,
        "chosen_algorithm": "",
        "algorithm_scores": {},
        "mining_result": {},
        "hypotheses": [],
        "validated_hypotheses": [],
        "provenance_hash": "",
        "history": [],
        "error": None,
        "X": None,
        "y": None,
    }


def make_planner() -> PlannerAgent:
    config = {
        "planner": {
            "w_drift": 0.4,
            "w_accuracy": 0.3,
            "w_cost": 0.3,
            "alpha_ema": 0.7,
        }
    }
    return PlannerAgent(config)


def test_high_drift_selects_isolation_forest():
    planner = make_planner()
    state = make_state(drift_score=1.0)

    updated = planner.run(state)

    assert updated["chosen_algorithm"] == "IsolationForest"


def test_zero_drift_selects_statistical_rules():
    planner = make_planner()
    state = make_state(drift_score=0.0)

    updated = planner.run(state)

    assert updated["chosen_algorithm"] == "StatisticalRules"


def test_accuracy_ema_update():
    planner = make_planner()

    old_acc = planner.accuracy_history["IsolationForest"]
    quality_score = 0.8
    planner.update_accuracy("IsolationForest", quality_score)

    new_acc = planner.accuracy_history["IsolationForest"]
    expected = planner.alpha_ema * old_acc + (1 - planner.alpha_ema) * quality_score

    assert abs(new_acc - expected) < 0.001
