"""Quick UCI loader + orchestrator smoke test.

What this script does:
1. Loads UCI Gas Sensor batches using UCIGasSensorLoader
2. Prints shapes of the first 3 batches
3. Prints mean(sensor_0) across the first 3 batches to show natural drift
4. Runs the full ManualOrchestrator on 3 UCI batches and prints results

Run:
    python test_uci_loader.py
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import yaml

from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.streaming.uci_loader import UCIGasSensorLoader


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _first_three_batch_report(loader: UCIGasSensorLoader) -> List[Tuple[int, float]]:
    """Print shape of first 3 batches and return sensor_0 means."""
    print("\n=== First 3 UCI Batches: Shapes ===")
    means: List[Tuple[int, float]] = []

    for i in range(1, 4):
        X, y = loader.next_batch()
        if X is None or y is None:
            print(f"Batch {i}: unavailable")
            continue

        print(f"Batch {i}: X shape={X.shape}, y shape={y.shape}")

        if "sensor_0" in X.columns:
            sensor0_mean = float(X["sensor_0"].mean())
            means.append((i, sensor0_mean))
        else:
            # Fallback for unexpected column naming
            sensor0_mean = float(X.iloc[:, 0].mean())
            means.append((i, sensor0_mean))

    print("\n=== Mean(sensor_0) Across First 3 Batches ===")
    for batch_id, mean_val in means:
        print(f"Batch {batch_id}: mean(sensor_0) = {mean_val:.6f}")

    return means


def _run_orchestrator_three_batches(config: dict) -> None:
    """Run full manual orchestrator on UCI batches 1..3 and print results."""
    uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))
    uci_cfg["batch_numbers"] = [1, 2, 3]

    loader = UCIGasSensorLoader(**uci_cfg)
    orchestrator = ManualOrchestrator(config)

    print("\n=== Running ManualOrchestrator on 3 UCI Batches ===")
    results = orchestrator.run(loader)
    orchestrator.print_results(results)


def main() -> None:
    config = _load_config()

    uci_cfg = dict(config.get("uci_loader", config.get("uci_streaming", {})))
    uci_cfg["batch_numbers"] = [1, 2, 3]

    loader = UCIGasSensorLoader(**uci_cfg)
    _first_three_batch_report(loader)

    _run_orchestrator_three_batches(config)


if __name__ == "__main__":
    main()
