"""Neo4j Provenance Logger - Full Integration Test with Visualization

This script validates Neo4j provenance logging and visualization:
1. Loads UCI Gas Sensor dataset
2. Runs the full pipeline (manual orchestrator) with Neo4j backend enabled
3. Queries all events and displays in a formatted table
4. Verifies full hash-chain integrity
5. Demonstrates tamper detection by modifying an event in Neo4j
6. Generates a pyvis graph visualization
7. Prints useful Cypher queries for Neo4j Browser analysis

Run:
    python test_neo4j_provenance.py

Prerequisites:
    - Neo4j server running at bolt://localhost:7687
    - config.yaml has provenance.backend = "neo4j" and credentials set
    - Sufficient UCI data in dataset1/ (batches 1-10)
    - pyvis installed (pip install pyvis)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from neo4j import GraphDatabase

from aadmf.orchestrator.manual import ManualOrchestrator
from aadmf.streaming.uci_loader import UCIGasSensorLoader


def _load_config(path: str = "config.yaml") -> dict:
    """Load and adapt config for Neo4j testing with UCI data."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure Neo4j backend is enabled
    config.setdefault("provenance", {})
    config["provenance"]["backend"] = "neo4j"
    config["provenance"]["neo4j"] = config["provenance"].get("neo4j", {})
    config["provenance"]["neo4j"]["enabled"] = True

    # Configure UCI data
    config.setdefault("streaming", {})
    config["streaming"]["dataset"] = "uci"
    config["streaming"]["uci_batch_count"] = 5  # Test with 5 batches for speed

    data_dir = "dataset1"
    config.setdefault("uci_loader", {})
    config["uci_loader"]["data_dir"] = data_dir
    config["uci_loader"]["batch_numbers"] = list(range(1, 6))
    config["uci_loader"]["use_ucimlrepo"] = False

    config.setdefault("uci_streaming", {})
    config["uci_streaming"]["data_dir"] = data_dir
    config["uci_streaming"]["batch_numbers"] = list(range(1, 6))
    config["uci_streaming"]["use_ucimlrepo"] = False

    return config


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_events_table(events: List[Dict[str, Any]]) -> None:
    """Print events in a formatted table."""
    if not events:
        print("(No events found)")
        return

    # Convert to DataFrame for nice tabular display
    rows = []
    for evt in events:
        details_str = json.dumps(evt.get("details", {}))
        if len(details_str) > 50:
            details_str = details_str[:47] + "..."

        rows.append({
            "Seq": evt["seq"],
            "Type": evt["type"],
            "Hash": evt["hash"],
            "Prev Hash": evt["prev_hash"][:8] + ("..." if len(evt["prev_hash"]) > 8 else ""),
            "Details": details_str,
            "Timestamp": f"{evt['ts']:.2f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTotal events: {len(events)}")


def run_orchestrator(config: dict, num_batches: int = 5) -> tuple:
    """Run the manual orchestrator with Neo4j backend.

    Returns:
        (orchestrator, provenance_logger)
    """
    print_section("Running Orchestrator with Neo4j Backend")

    orchestrator = ManualOrchestrator(config, use_langgraph=False)
    streamer = UCIGasSensorLoader(
        data_dir=config["uci_loader"]["data_dir"],
        batch_numbers=config["uci_loader"]["batch_numbers"],
        use_ucimlrepo=False,
    )

    for i in range(num_batches):
        X, y = streamer.next_batch()
        if X is None:
            print(f"Batch {i+1}: No data available")
            break

        print(f"Processing batch {i+1}/{num_batches} (shape: {X.shape})...")
        orchestrator.run(X, y)

    return orchestrator, orchestrator.prov


def query_all_events(logger: Any) -> List[Dict[str, Any]]:
    """Query all events from Neo4j logger."""
    return logger.query_by_type("*") if hasattr(logger.query_by_type, "__call__") else _raw_query_all_events(logger)


def _raw_query_all_events(logger: Any) -> List[Dict[str, Any]]:
    """Fallback: directly query Neo4j for all events."""
    driver = logger.driver
    events: List[Dict[str, Any]] = []

    with driver.session() as session:
        records = list(
            session.run(
                """
                MATCH (e:Event)
                RETURN e.seq AS seq, e.type AS type, e.ts AS ts,
                       e.details AS details, e.hash AS hash, e.prev_hash AS prev_hash
                ORDER BY e.seq ASC
                """
            )
        )

    for rec in records:
        events.append({
            "seq": int(rec["seq"]),
            "type": str(rec["type"]),
            "ts": float(rec["ts"]),
            "details": json.loads(rec["details"] or "{}"),
            "hash": str(rec["hash"]),
            "prev_hash": str(rec["prev_hash"]),
        })

    return events


def verify_and_report(logger: Any) -> bool:
    """Run verify_integrity() and report results."""
    print_section("Integrity Verification")

    intact, broken_at = logger.verify_integrity()

    if intact:
        print("✓ Chain is INTACT - all hashes verified successfully")
        return True
    else:
        print(f"✗ Chain is BROKEN at event {broken_at}")
        return False


def tamper_test(logger: Any, event_seq: int = 2) -> None:
    """Modify an event and verify tamper detection.

    This demonstrates the cryptographic tamper-detection capability.
    """
    print_section("Tamper Detection Test")

    driver = logger.driver

    # Read the target event
    with driver.session() as session:
        rec = session.run(
            "MATCH (e:Event {seq: $seq}) RETURN e.hash AS hash, e.details AS details",
            seq=event_seq,
        ).single()

        if not rec:
            print(f"Event {event_seq} not found")
            return

        original_details = json.loads(rec["details"])
        original_hash = rec["hash"]

    print(f"Original event {event_seq}: hash={original_hash}")
    print(f"Original details: {original_details}")

    # Tamper with the details
    tampered_details = original_details.copy()
    if "batch_id" in tampered_details:
        tampered_details["batch_id"] = 999
    else:
        tampered_details["TAMPERED"] = True

    print(f"\nTampering event {event_seq} details to: {tampered_details}")

    # Update in Neo4j
    with driver.session() as session:
        session.run(
            "MATCH (e:Event {seq: $seq}) SET e.details = $details",
            seq=event_seq,
            details=json.dumps(tampered_details),
        )

    print("\nEvent modified in Neo4j.")

    # Verify integrity again
    intact, broken_at = logger.verify_integrity()

    if not intact and broken_at == event_seq:
        print(f"✓ Tamper detected at event {broken_at} - integrity check FAILED as expected")
    elif not intact:
        print(f"✓ Tamper detected (first break at event {broken_at})")
    else:
        print("✗ ERROR: Tamper not detected (chain still reports as intact)")

    # Restore original
    print(f"\nRestoring original details...")
    with driver.session() as session:
        session.run(
            "MATCH (e:Event {seq: $seq}) SET e.details = $details",
            seq=event_seq,
            details=json.dumps(original_details),
        )

    # Verify restored
    intact, _ = logger.verify_integrity()
    if intact:
        print("✓ Chain integrity restored after reverting changes")
    else:
        print("✗ Chain still broken after reverting (unexpected)")


def generate_pyvis_visualization(logger: Any, output_path: str = "provenance_graph.html") -> None:
    """Generate an interactive pyvis graph visualization."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("\n[INFO] pyvis not installed. Skipping visualization.")
        print("       Install with: pip install pyvis")
        return

    print_section("Generating Interactive Graph Visualization")

    driver = logger.driver
    events: List[Dict[str, Any]] = []
    edges: List[tuple] = []

    # Fetch all events and relationships
    with driver.session() as session:
        # Get events
        records = list(
            session.run(
                """
                MATCH (e:Event)
                RETURN e.seq AS seq, e.type AS type, e.hash AS hash
                ORDER BY e.seq ASC
                """
            )
        )
        events = [
            {
                "seq": int(r["seq"]),
                "type": str(r["type"]),
                "hash": str(r["hash"]),
            }
            for r in records
        ]

        # Get PRECEDES relationships
        rel_records = list(
            session.run(
                """
                MATCH (a:Event)-[:PRECEDES]->(b:Event)
                RETURN a.seq AS from_seq, b.seq AS to_seq
                """
            )
        )
        edges = [(int(r["from_seq"]), int(r["to_seq"])) for r in rel_records]

    if not events:
        print("No events to visualize")
        return

    # Create pyvis network
    net = Network(directed=True, height="750px", width="100%", physics=True)

    # Add nodes
    for evt in events:
        label = f"[{evt['seq']}] {evt['type']}\n{evt['hash']}"
        color = "#FF6B6B" if evt["type"] == "SYSTEM" else "#4ECDC4"
        net.add_node(
            evt["seq"],
            label=label,
            title=f"Seq: {evt['seq']}\nType: {evt['type']}\nHash: {evt['hash']}",
            color=color,
            size=30,
        )

    # Add edges (PRECEDES relationships)
    for from_seq, to_seq in edges:
        net.add_edge(from_seq, to_seq, arrows="to")

    # Configure physics
    net.show_buttons(filter_=["physics"])
    net.write_html(output_path)

    print(f"✓ Graph visualization saved to: {output_path}")
    print(f"  Open in browser to explore the provenance chain interactively")


def print_cypher_queries() -> None:
    """Print useful Cypher queries for Neo4j Browser analysis."""
    print_section("Useful Cypher Queries for Neo4j Browser")

    queries = [
        {
            "name": "Show all events in order",
            "query": """MATCH (e:Event)
RETURN e.seq, e.type, e.ts, e.hash
ORDER BY e.seq ASC""",
        },
        {
            "name": "Show provenance chain for a specific batch",
            "query": """MATCH (e:Event)
WHERE e.details CONTAINS 'batch_id'
RETURN e.seq, e.type, e.hash, e.details
ORDER BY e.seq ASC""",
        },
        {
            "name": "Find all hypotheses with HIGH confidence",
            "query": """MATCH (e:Event)
WHERE e.type = 'HYPOTHESIS_GENERATED'
AND e.details CONTAINS 'HIGH'
RETURN e.seq, e.ts, e.details
ORDER BY e.seq DESC""",
        },
        {
            "name": "Show full hash chain (trace tamper detection)",
            "query": """MATCH (e:Event)
RETURN e.seq, e.hash, e.prev_hash, e.type
ORDER BY e.seq ASC""",
        },
        {
            "name": "Count events by type",
            "query": """MATCH (e:Event)
RETURN e.type, count(*) AS count
ORDER BY count DESC""",
        },
        {
            "name": "Show event precedence relationships",
            "query": """MATCH (a:Event)-[:PRECEDES]->(b:Event)
RETURN a.seq, a.type, b.seq, b.type
ORDER BY a.seq ASC""",
        },
        {
            "name": "Find events with specific secondary labels",
            "query": """MATCH (e:ALGO_SELECTED)
RETURN e.seq, e.type, e.details
ORDER BY e.seq ASC""",
        },
        {
            "name": "Show timeline of events (with timestamps)",
            "query": """MATCH (e:Event)
RETURN e.seq, e.type, datetime(e.ts*1000) AS timestamp, e.hash
ORDER BY e.ts ASC""",
        },
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{i}. {q['name']}:")
        print("-" * 78)
        for line in q["query"].split("\n"):
            print(f"  {line}")
        print()


def print_summary(logger: Any, events: List[Dict[str, Any]]) -> None:
    """Print a summary of the provenance chain."""
    print_section("Provenance Summary")

    summary = logger.summary()
    print(f"Total events: {summary['total_events']}")
    print(f"First hash: {summary['first_hash']}")
    print(f"Last hash: {summary['last_hash']}")

    print("\nEvents by type:")
    for event_type, count in sorted(summary["by_type"].items()):
        print(f"  {event_type:30s}: {count:3d}")

    if events:
        earliest = min(evt["ts"] for evt in events)
        latest = max(evt["ts"] for evt in events)
        duration = latest - earliest
        print(f"\nTimeline: {duration:.2f} seconds ({len(events)} events)")


def main() -> None:
    """Main test execution."""
    print("\n" + "=" * 80)
    print("  Neo4j Provenance Logger - Full Integration Test")
    print("=" * 80)

    # Load configuration
    config = _load_config()

    # Verify Neo4j is configured
    neo4j_cfg = config.get("provenance", {}).get("neo4j", {})
    if not neo4j_cfg.get("enabled"):
        print("\n[ERROR] Neo4j not enabled in config.yaml")
        print("  Set: provenance.neo4j.enabled = true")
        return

    try:
        # Run orchestrator
        orchestrator, logger = run_orchestrator(config, num_batches=5)

        # Query events
        print_section("Fetching All Provenance Events")
        events = _raw_query_all_events(logger)
        print(f"Fetched {len(events)} events from Neo4j")

        # Display events table
        print_section("Event Log (First 30 events)")
        print_events_table(events[:30])

        # Summary
        print_summary(logger, events)

        # Integrity verification
        verify_and_report(logger)

        # Tamper test
        if len(events) > 2:
            tamper_test(logger, event_seq=2)
        else:
            print("\n[INFO] Not enough events for tamper test")

        # Visualization
        generate_pyvis_visualization(logger)

        # Print Cypher queries
        print_cypher_queries()

        # Export JSON
        export_path = "provenance_export.json"
        logger.export_json(export_path)
        print_section("Export")
        print(f"✓ Provenance chain exported to: {export_path}")

        print_section("Test Complete")
        print("If Neo4j Browser is running, visit: http://localhost:7474")
        print("Run the Cypher queries above to explore the provenance graph interactively")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
