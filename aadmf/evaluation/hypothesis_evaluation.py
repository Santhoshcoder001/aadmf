"""Week 7 hypothesis evaluation on real UCI gas sensor drift data.

This script:
1. Loads UCI Gas Sensor data via UCIGasSensorLoader (10 batches)
2. Runs full ManualOrchestrator (Hypothesizer v2)
3. Collects all generated hypotheses
4. Computes HVR (Hypothesis Validity Rate)
5. Breaks down HVR by drift level: low / medium / high
6. Compares HVR for real MI vs proxy MI (if both runs succeed)
7. Saves results to experiments/results/hypothesis_evaluation_week7.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

import pandas as pd
import yaml

from aadmf.core.state import Hypothesis
from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.streaming.uci_loader import UCIGasSensorLoader


LOW_DRIFT_THRESHOLD = 0.10
HIGH_DRIFT_THRESHOLD = 0.50


@dataclass
class HVRSummary:
    mode: str
    total_hypotheses: int
    valid_hypotheses: int
    hvr: float
    band_stats: Dict[str, Dict[str, float]]
    top_hypotheses: List[Hypothesis]


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _detect_uci_data_dir(preferred_dir: str) -> Optional[str]:
    """Find local UCI batch directory, if available."""
    candidates = [preferred_dir, "data/raw", "dataset1", "dataset2"]
    for candidate in candidates:
        p = Path(candidate)
        if (p / "batch1.dat").exists() and (p / "batch10.dat").exists():
            return str(p)
    return None


def _prepare_config(base_config: dict, real_mi: bool) -> dict:
    """Force UCI mode and set hypothesizer MI mode."""
    cfg = copy.deepcopy(base_config)

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

    hyp_cfg = cfg.setdefault("hypothesizer", {})
    hyp_cfg["real_mi"] = bool(real_mi)
    hyp_cfg.setdefault("mi_decay_threshold", 0.05)
    hyp_cfg.setdefault("use_decay_trigger", True)

    return cfg


def _drift_band(drift_score: float) -> str:
    if drift_score <= LOW_DRIFT_THRESHOLD:
        return "low"
    if drift_score >= HIGH_DRIFT_THRESHOLD:
        return "high"
    return "medium"


def _batch_to_drift_band(results_df: pd.DataFrame) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for _, row in results_df.iterrows():
        mapping[int(row["batch_id"])] = _drift_band(float(row["drift_score"]))
    return mapping


def _confidence_rank(value: Optional[str]) -> int:
    ranks = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    return ranks.get(str(value), 0)


def _hypothesis_rank_key(h: Hypothesis) -> tuple:
    valid_rank = 1 if bool(h.valid) else 0
    conf_rank = _confidence_rank(h.confidence)
    combined_score = 0.5 * abs(float(h.correlation)) + 0.5 * float(h.mutual_info_proxy)
    p_real = float(h.p_value_chi2) if h.p_value_chi2 is not None else 1.0
    return (valid_rank, conf_rank, combined_score, -p_real)


def _summarize_results(mode: str, results: dict) -> HVRSummary:
    all_hypotheses: List[Hypothesis] = list(results.get("all_hypotheses", []))
    results_df: pd.DataFrame = results.get("results_df", pd.DataFrame()).copy()

    total = len(all_hypotheses)
    valid = sum(1 for h in all_hypotheses if bool(h.valid))
    hvr = (valid / total) if total else 0.0

    batch_band = _batch_to_drift_band(results_df) if not results_df.empty else {}

    band_counts = {
        "low": {"total": 0, "valid": 0},
        "medium": {"total": 0, "valid": 0},
        "high": {"total": 0, "valid": 0},
    }
    for h in all_hypotheses:
        band = batch_band.get(int(h.batch), "medium")
        band_counts[band]["total"] += 1
        if bool(h.valid):
            band_counts[band]["valid"] += 1

    band_stats: Dict[str, Dict[str, float]] = {}
    for band, vals in band_counts.items():
        band_total = int(vals["total"])
        band_valid = int(vals["valid"])
        band_hvr = (band_valid / band_total) if band_total else 0.0
        band_stats[band] = {
            "total": band_total,
            "valid": band_valid,
            "hvr": band_hvr,
        }

    top_h = sorted(all_hypotheses, key=_hypothesis_rank_key, reverse=True)[:3]

    return HVRSummary(
        mode=mode,
        total_hypotheses=total,
        valid_hypotheses=valid,
        hvr=hvr,
        band_stats=band_stats,
        top_hypotheses=top_h,
    )


def _run_once(config: dict) -> dict:
    uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))
    loader = UCIGasSensorLoader(**uci_cfg)
    orchestrator = ManualOrchestrator(config)
    return orchestrator.run(loader)


def _summary_rows(summary: HVRSummary) -> List[dict]:
    rows = [
        {
            "mode": summary.mode,
            "drift_level": "overall",
            "total_hypotheses": summary.total_hypotheses,
            "valid_hypotheses": summary.valid_hypotheses,
            "hvr": round(summary.hvr, 6),
        }
    ]

    for band in ["low", "medium", "high"]:
        s = summary.band_stats[band]
        rows.append(
            {
                "mode": summary.mode,
                "drift_level": band,
                "total_hypotheses": int(s["total"]),
                "valid_hypotheses": int(s["valid"]),
                "hvr": round(float(s["hvr"]), 6),
            }
        )

    return rows


def _print_summary(summary: HVRSummary) -> None:
    print("\n" + "=" * 72)
    print(f"Hypothesis Evaluation ({summary.mode})")
    print("=" * 72)
    print(f"Total hypotheses generated: {summary.total_hypotheses}")
    print(
        "Valid hypotheses: "
        f"{summary.valid_hypotheses} ({summary.hvr * 100:.2f}%) "
        f"[target > 70%: {'PASS' if summary.hvr > 0.70 else 'BELOW TARGET'}]"
    )

    print("\nHVR by drift level:")
    for band in ["low", "medium", "high"]:
        s = summary.band_stats[band]
        print(
            f"- {band:>6}: "
            f"{int(s['valid'])}/{int(s['total'])} "
            f"({float(s['hvr']) * 100:.2f}%)"
        )

    print("\nTop 3 hypotheses (best-ranked):")
    if not summary.top_hypotheses:
        print("- No hypotheses generated.")
        return

    for i, h in enumerate(summary.top_hypotheses, start=1):
        p_real = float(h.p_value_chi2) if h.p_value_chi2 is not None else float("nan")
        p_syn = float(h.p_value_synthetic) if h.p_value_synthetic is not None else float("nan")
        print(
            f"{i}. {h.id} | {h.feature_a}-{h.feature_b} | "
            f"valid={bool(h.valid)} | confidence={h.confidence} | "
            f"r={h.correlation:.4f} | MI={h.mutual_info_proxy:.4f} | "
            f"p_real={p_real:.6f} | p_synth={p_syn:.6f}"
        )


def main() -> None:
    base_cfg = _load_config("config.yaml")

    print("=" * 72)
    print("Week 7 Hypothesis Evaluation Runner")
    print("=" * 72)

    # Run with real MI (Hypothesizer v2 default)
    real_cfg = _prepare_config(base_cfg, real_mi=True)
    real_results = _run_once(real_cfg)
    real_summary = _summarize_results("real_mi", real_results)
    _print_summary(real_summary)

    # Run comparison with proxy MI
    proxy_summary: Optional[HVRSummary] = None
    try:
        proxy_cfg = _prepare_config(base_cfg, real_mi=False)
        proxy_results = _run_once(proxy_cfg)
        proxy_summary = _summarize_results("proxy_mi", proxy_results)
        _print_summary(proxy_summary)
    except Exception as exc:
        print("\nCould not run proxy-MI comparison.")
        print(f"Reason: {exc}")

    # Export CSV summary
    rows = _summary_rows(real_summary)
    if proxy_summary is not None:
        rows.extend(_summary_rows(proxy_summary))

    out_df = pd.DataFrame(rows)
    out_path = Path("experiments/results/hypothesis_evaluation_week7.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 72)
    print(f"Saved Week 7 hypothesis evaluation to: {out_path}")

    if proxy_summary is not None:
        delta = real_summary.hvr - proxy_summary.hvr
        trend = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
        print(
            "HVR comparison (real MI vs proxy MI): "
            f"{real_summary.hvr * 100:.2f}% vs {proxy_summary.hvr * 100:.2f}% "
            f"(real MI is {trend} by {abs(delta) * 100:.2f} pp)"
        )


if __name__ == "__main__":
    main()
