"""Week 6 combined tuning runner.

This script:
1. Loads real UCI Gas Sensor batches with UCIGasSensorLoader
2. Runs ScoringMatrixTuner (4 weight presets)
3. Runs MinerTuningExperiments
4. Prints final recommendations
5. Saves a combined markdown report to
   experiments/results/week6_final_recommendation.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import copy

import pandas as pd
import yaml

from aadmf.evaluation.scoring_matrix_tuning import ScoringMatrixTuner
from aadmf.evaluation.miner_tuning import MinerTuningExperiments
from aadmf.streaming.uci_loader import UCIGasSensorLoader


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_uci_mode(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    stream_cfg = cfg.setdefault("streaming", {})
    stream_cfg["dataset"] = "uci"
    stream_cfg["uci_batch_count"] = 10

    uci_cfg = cfg.setdefault("uci_loader", cfg.get("uci_streaming", {}))
    uci_cfg.setdefault("data_dir", "data/raw")
    uci_cfg["batch_numbers"] = list(range(1, 11))
    uci_cfg.setdefault("normalize", True)
    uci_cfg.setdefault("use_ucimlrepo", True)

    detected_dir = _detect_uci_data_dir(uci_cfg.get("data_dir", "data/raw"))
    if detected_dir is not None:
        uci_cfg["data_dir"] = detected_dir
    return cfg


def _detect_uci_data_dir(preferred_dir: str) -> Optional[str]:
    """Find a local directory containing UCI batch*.dat files.

    Search order: preferred dir -> data/raw -> dataset1 -> dataset2.
    Returns directory path string if found, else None.
    """
    candidates = [preferred_dir, "data/raw", "dataset1", "dataset2"]
    for candidate in candidates:
        p = Path(candidate)
        if (p / "batch1.dat").exists() and (p / "batch10.dat").exists():
            return str(p)
    return None


def _verify_uci_loading(config: dict) -> Tuple[int, int]:
    """Load UCI data once to confirm batch/row availability."""
    uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))
    loader = UCIGasSensorLoader(**uci_cfg)
    batches = loader.load_all_batches()
    n_batches = len(batches)
    n_rows = sum(len(X) for X, _ in batches)
    return n_batches, n_rows


def _best_miner_param(
    miner_df: pd.DataFrame,
    miner_name: str,
    field: str,
    fallback_field: str,
) -> Tuple[Optional[float], Optional[str], float]:
    """Pick best high-drift parameter for a miner.

    If no explicit high-drift rows exist, falls back to top 30% drift-score rows.
    Returns: (parameter_value, adaptive_formula, mean_quality_score)
    """
    df = miner_df[miner_df["miner"] == miner_name].copy()
    high_df = df[df["drift_band"] == "high"]

    if high_df.empty:
        cutoff = float(df["drift_score"].quantile(0.70))
        high_df = df[df["drift_score"] >= cutoff]

    grp_cols = [field, "adaptive_formula"] if field in high_df.columns else [fallback_field, "adaptive_formula"]
    grouped = (
        high_df.groupby(grp_cols, as_index=False)["quality_score"]
        .mean()
        .rename(columns={"quality_score": "mean_quality_score"})
        .sort_values("mean_quality_score", ascending=False)
    )
    if grouped.empty:
        return None, None, float("nan")

    row = grouped.iloc[0]
    param_key = field if field in row.index else fallback_field
    return float(row[param_key]), str(row["adaptive_formula"]), float(row["mean_quality_score"])


def _write_report(
    report_path: Path,
    n_batches: int,
    n_rows: int,
    scoring_df: pd.DataFrame,
    miner_df: pd.DataFrame,
    best_weights: pd.Series,
    if_best: Tuple[Optional[float], Optional[str], float],
    db_best: Tuple[Optional[float], Optional[str], float],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if_val, if_formula, if_q = if_best
    db_val, db_formula, db_q = db_best

    try:
        scoring_table = scoring_df.to_markdown(index=False)
    except Exception:
        scoring_table = scoring_df.to_string(index=False)

    lines = [
        "# Week 6 Final Recommendation",
        "",
        "## Dataset Verification",
        f"- UCI batches loaded: {n_batches}",
        f"- Total rows processed: {n_rows}",
        "",
        "## Best Scoring Matrix Recommendation",
        f"- Preset: {best_weights['preset']}",
        f"- Weights: drift={best_weights['w_drift']}, accuracy={best_weights['w_accuracy']}, cost={best_weights['w_cost']}",
        f"- Mean quality score: {best_weights['mean_quality_score']}",
        f"- High-drift mean quality score: {best_weights['high_drift_mean_quality_score']}",
        "",
        "## Best Miner Parameter Recommendations (High Drift)",
        f"- IsolationForest base_contamination: {if_val} (formula={if_formula}, mean_quality={if_q:.4f})",
        f"- DBSCAN base_eps: {db_val} (formula={db_formula}, mean_quality={db_q:.4f})",
        "",
        "## Saved Artifacts",
        "- experiments/results/scoring_matrix_tuning.csv",
        "- experiments/results/miner_tuning_results.csv",
        "- experiments/results/week6_final_recommendation.md",
        "",
        "## Novelty Log Entry Preparation",
        "Use the following points for NOVELTY_LOG.md (Week 6):",
        "- Decision: Tuned scoring weights and miner hyperparameters on real UCI drift batches.",
        "- Evidence: Comparative quality-score results across four scoring presets and miner parameter grids.",
        "- Rationale: Real drift behavior yields more reliable operating points than synthetic-only tuning.",
        "- Patent relevance: Strengthens adaptive selection and real-world robustness claims.",
        "",
        "## Scoring Matrix Results Table",
        scoring_table,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = _ensure_uci_mode(_load_config())

    print("=" * 72)
    print("Week 6 Combined Tuning Runner")

    n_batches, n_rows = _verify_uci_loading(config)
    print(f"Loaded UCI data successfully: batches={n_batches}, total_rows={n_rows}")

    scoring_tuner = ScoringMatrixTuner(
        base_config=config,
        output_csv="experiments/results/scoring_matrix_tuning.csv",
    )
    scoring_df = scoring_tuner.run_all()
    best_weights = scoring_tuner.recommend_best(scoring_df)

    miner_tuner = MinerTuningExperiments(
        config=config,
        output_csv="experiments/results/miner_tuning_results.csv",
    )
    miner_df = miner_tuner.run_all()

    if_best = _best_miner_param(
        miner_df,
        miner_name="IsolationForest",
        field="base_contamination",
        fallback_field="effective_param",
    )
    db_best = _best_miner_param(
        miner_df,
        miner_name="DBSCAN",
        field="base_eps",
        fallback_field="effective_param",
    )

    print("\n" + "=" * 72)
    print(
        "Best scoring weights: "
        f"drift={best_weights['w_drift']}, accuracy={best_weights['w_accuracy']}, cost={best_weights['w_cost']} "
        f"({best_weights['preset']})"
    )
    print(
        "Best IsolationForest contamination: "
        f"{if_best[0]} (formula={if_best[1]})"
    )
    print(
        "Best DBSCAN eps: "
        f"{db_best[0]} (formula={db_best[1]})"
    )

    report_path = Path("experiments/results/week6_final_recommendation.md")
    _write_report(
        report_path=report_path,
        n_batches=n_batches,
        n_rows=n_rows,
        scoring_df=scoring_df,
        miner_df=miner_df,
        best_weights=best_weights,
        if_best=if_best,
        db_best=db_best,
    )
    print(f"\nSaved combined report to: {report_path}")


if __name__ == "__main__":
    main()
