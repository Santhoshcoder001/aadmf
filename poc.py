"""
AADMF Proof of Concept - single-file entry point.
Run: python poc.py

Demonstrates all 7 components working on a laptop in < 30 seconds.
"""

import yaml

from aadmf.orchestrator.manual import ManualOrchestrator, build_streamer_from_config


def main() -> None:
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Execution mode switches (backward compatible):
    # - execution.mode: "full" (default) or "debug"
    # - execution.use_langgraph: explicit override when present
    execution_cfg = config.get("execution", {})
    execution_mode = str(execution_cfg.get("mode", "full")).lower()
    use_langgraph = execution_cfg.get("use_langgraph")
    if use_langgraph is None:
        use_langgraph = execution_mode == "full"
    use_langgraph = bool(use_langgraph)

    # Create streamer from config["streaming"]["dataset"]
    stream_cfg = config.get("streaming", {})

    # Full mode defaults to UCI pipeline for Week 5+ real-data workflow.
    if execution_mode == "full":
        stream_cfg["dataset"] = "uci"

    dataset = str(stream_cfg.get("dataset", "synthetic")).lower()
    streamer = build_streamer_from_config(config)

    if use_langgraph:
        print("LangGraph Mode Activated")

    if dataset == "uci":
        print("Using UCI Gas Sensor Dataset - Real Drift Mode")

    # Run orchestrator
    orchestrator = ManualOrchestrator(config, use_langgraph=use_langgraph)
    results = orchestrator.run(streamer)
    orchestrator.print_results(results)

    # Export provenance from whichever orchestration path actually ran.
    export_path = config.get("provenance", {}).get("export_path", "provenance.json")
    if use_langgraph and getattr(orchestrator, "_langgraph_flow", None) is not None:
        orchestrator._langgraph_flow.prov.export_json(export_path)
    else:
        orchestrator.prov.export_json(export_path)

    print("\nProvenance exported to provenance.json")


if __name__ == "__main__":
    main()
