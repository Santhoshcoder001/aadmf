"""Provenance logger with SHA-256 hash-chained event ledger (Phase 1).

This module provides the DictChainLogger class for tamper-evident provenance
tracking. Each event is cryptographically hashed and linked to the previous
event via its hash, forming an immutable chain. Any modification to any event
breaks the integrity chain, detectable via verify_integrity().

Patent Novelty Point 3: Hash-chained event ledger for agentic data mining
decision provenance.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Tuple, Any, Optional


class DictChainLogger:
    """Tamper-evident hash-chained event ledger.

    Each event is SHA-256 hashed and chained to its predecessor. The chain
    starts with a "GENESIS" sentinel hash. Any modification to any event
    breaks the chain, allowing detection of tampering via verify_integrity().

    Example:
        >>> logger = DictChainLogger()
        >>> logger.log("SYSTEM_START", {"components": ["A", "B"]})
        >>> logger.log("DATA_INGESTED", {"batch": 0, "rows": 150})
        >>> intact, broken_at = logger.verify_integrity()
        >>> print(intact)  # True if chain unmodified
        >>> results = logger.summary()
    """

    def __init__(self) -> None:
        """Initialize an empty provenance chain with GENESIS sentinel."""
        self.chain: List[Dict[str, Any]] = []
        self.prev_hash: str = "GENESIS"
        self._type_index: Dict[str, List[int]] = {}

    def _compute_hash(self, event: dict) -> str:
        """Compute SHA-256 hash of an event dict.

        YOUR ORIGINAL ALGORITHM (PoC):
        - Exclude the 'hash' key from the payload (if present)
        - JSON-serialize with sort_keys=True for determinism
        - Use default=str to handle un-serializable types
        - Compute SHA-256 and return first 16 hex characters

        Args:
            event: dict with event data (excluding 'hash' key)

        Returns:
            16-character hex string (first 16 chars of SHA-256 digest)
        """
        payload = json.dumps(event, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def log(self, event_type: str, details: dict) -> str:
        """Append a new event to the chain.

        YOUR ORIGINAL ALGORITHM (PoC):
        1. Build event dict with: seq, type, ts, details, prev_hash
        2. Compute hash via _compute_hash
        3. Add hash to event dict
        4. Append to chain
        5. Update prev_hash for next event
        6. Update _type_index for query_by_type()
        7. Return the hash

        Args:
            event_type: String identifier (e.g., "DATA_INGESTED", "ALGO_SELECTED")
            details: Dict of event-specific data (e.g., batch number, algorithm name)

        Returns:
            Hash of the new event (16-char hex string)
        """
        event = {
            "seq": len(self.chain),
            "type": event_type,
            "ts": time.time(),
            "details": details,
            "prev_hash": self.prev_hash,
        }
        h = self._compute_hash(event)
        event["hash"] = h
        self.chain.append(event)
        self.prev_hash = h
        self._type_index.setdefault(event_type, []).append(event["seq"])
        return h

    def verify_integrity(self) -> Tuple[bool, int]:
        """Walk the full chain and verify every hash link.

        YOUR ORIGINAL ALGORITHM (PoC):
        - Start with prev = "GENESIS"
        - For each event in chain:
          1. Check event["prev_hash"] == prev
          2. Recompute hash excluding 'hash' key
          3. Check computed hash == stored hash
          4. Update prev = event["hash"]
        - Return (True, -1) if all OK; (False, broken_at_index) if tampering found

        Returns:
            Tuple (intact, broken_at):
            - intact: bool — True if all hashes verify
            - broken_at: int — -1 if intact; index of first broken link if tampering detected
        """
        prev = "GENESIS"
        for i, event in enumerate(self.chain):
            stored_hash = event["hash"]
            prev_in_event = event["prev_hash"]
            if prev_in_event != prev:
                return False, i
            # Recompute hash excluding the 'hash' key itself
            check_data = {k: v for k, v in event.items() if k != "hash"}
            computed = self._compute_hash(check_data)
            if computed != stored_hash:
                return False, i
            prev = stored_hash
        return True, -1

    def query_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Return all events of a given type.

        Args:
            event_type: String identifier to filter by

        Returns:
            List of event dicts matching the type, in order of creation
        """
        seqs = self._type_index.get(event_type, [])
        return [self.chain[s] for s in seqs]

    def export_json(self, path: str) -> None:
        """Export full provenance chain to JSON file.

        Args:
            path: File path to write (e.g., "provenance.json")
        """
        with open(path, "w") as f:
            json.dump(self.chain, f, indent=2, default=str)

    def summary(self) -> dict:
        """Return summary statistics of the chain.

        Returns:
            Dict with keys:
            - total_events: int — number of events logged
            - by_type: dict — count of events per type
            - first_hash: str — hash of first event (or None if empty)
            - last_hash: str — hash of last event (or None if empty)
        """
        types: Dict[str, int] = {}
        for e in self.chain:
            types[e["type"]] = types.get(e["type"], 0) + 1
        return {
            "total_events": len(self.chain),
            "by_type": types,
            "first_hash": self.chain[0]["hash"] if self.chain else None,
            "last_hash": self.chain[-1]["hash"] if self.chain else None,
        }


# ─────────────────────────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────────────────────────


def test_chain_intact_after_logging():
    """Verify that an unmodified chain passes integrity check."""
    logger = DictChainLogger()
    logger.log("SYSTEM_START", {"components": ["Simulator", "Detector", "Planner"]})
    logger.log("DATA_INGESTED", {"batch": 0, "rows": 150})
    logger.log("ALGO_SELECTED", {"algorithm": "IsolationForest", "drift_score": 0.0})

    intact, broken_at = logger.verify_integrity()
    assert intact is True, "Chain should be intact"
    assert broken_at == -1, "broken_at should be -1 when chain is intact"


def test_tamper_breaks_chain():
    """Verify that modifying an event breaks the chain at that position."""
    logger = DictChainLogger()
    logger.log("SYSTEM_START", {})
    logger.log("DATA_INGESTED", {"batch": 0})
    logger.log("MINING_RESULT", {"algorithm": "DBSCAN"})
    logger.log("HYPOTHESIS_VALIDATED", {"id": "H0_1_2", "valid": True})

    # Tamper with event at index 1 (DATA_INGESTED)
    logger.chain[1]["details"]["TAMPERED"] = True

    intact, broken_at = logger.verify_integrity()
    assert intact is False, "Chain should be broken after tampering"
    assert broken_at == 1, "Tampering should be detected at index 1"


def test_tamper_cascades_downstream():
    """Verify that tampering at index i breaks detection at i (not downstream)."""
    logger = DictChainLogger()
    logger.log("SYSTEM_START", {})
    logger.log("DATA_INGESTED", {"batch": 0})
    logger.log("DRIFT_CHECK", {"score": 0.1})

    # Tamper middle event
    logger.chain[1]["details"]["CORRUPTED"] = "yes"

    intact, broken_at = logger.verify_integrity()
    assert broken_at == 1, "Break should be detected at exact tampered event"


def test_query_by_type():
    """Verify filtering events by type."""
    logger = DictChainLogger()
    logger.log("DATA_INGESTED", {"batch": 0})
    logger.log("DATA_INGESTED", {"batch": 1})
    logger.log("DATA_INGESTED", {"batch": 2})
    logger.log("ALGO_SELECTED", {"algorithm": "IF"})
    logger.log("ALGO_SELECTED", {"algorithm": "DBSCAN"})
    logger.log("MINING_RESULT", {"quality": 0.9})

    ingested = logger.query_by_type("DATA_INGESTED")
    assert len(ingested) == 3, "Should find 3 DATA_INGESTED events"
    assert all(e["type"] == "DATA_INGESTED" for e in ingested)

    selected = logger.query_by_type("ALGO_SELECTED")
    assert len(selected) == 2, "Should find 2 ALGO_SELECTED events"

    mining = logger.query_by_type("MINING_RESULT")
    assert len(mining) == 1, "Should find 1 MINING_RESULT event"

    nonexistent = logger.query_by_type("NONEXISTENT")
    assert len(nonexistent) == 0, "Should return empty list for nonexistent type"


def test_genesis_hash():
    """Verify that first event has prev_hash = 'GENESIS'."""
    logger = DictChainLogger()
    logger.log("SYSTEM_START", {})
    assert logger.chain[0]["prev_hash"] == "GENESIS"
    assert logger.chain[0]["type"] == "SYSTEM_START"


def test_hash_chain_linkage():
    """Verify that each event's prev_hash matches previous event's hash."""
    logger = DictChainLogger()
    logger.log("EVENT_A", {})
    logger.log("EVENT_B", {})
    logger.log("EVENT_C", {})

    # Check linkage
    assert logger.chain[0]["prev_hash"] == "GENESIS"
    assert logger.chain[1]["prev_hash"] == logger.chain[0]["hash"]
    assert logger.chain[2]["prev_hash"] == logger.chain[1]["hash"]


def test_summary():
    """Verify summary statistics are computed correctly."""
    logger = DictChainLogger()
    for i in range(5):
        logger.log("DATA_INGESTED", {"batch": i})
    logger.log("DRIFT_CHECK", {"score": 0.0})
    logger.log("DRIFT_CHECK", {"score": 0.5})

    summary = logger.summary()
    assert summary["total_events"] == 7, "Should have 7 total events"
    assert summary["by_type"]["DATA_INGESTED"] == 5, "Should have 5 DATA_INGESTED"
    assert summary["by_type"]["DRIFT_CHECK"] == 2, "Should have 2 DRIFT_CHECK"
    assert summary["first_hash"] == logger.chain[0]["hash"]
    assert summary["last_hash"] == logger.chain[-1]["hash"]


def test_empty_chain_summary():
    """Verify summary of empty chain."""
    logger = DictChainLogger()
    summary = logger.summary()
    assert summary["total_events"] == 0
    assert summary["by_type"] == {}
    assert summary["first_hash"] is None
    assert summary["last_hash"] is None


def test_hash_determinism():
    """Verify that the same event dict produces the same hash."""
    logger = DictChainLogger()
    event1 = {"type": "TEST", "value": 42}
    event2 = {"type": "TEST", "value": 42}
    h1 = logger._compute_hash(event1)
    h2 = logger._compute_hash(event2)
    assert h1 == h2, "Same event dicts should produce same hash"


def test_hash_sensitivity():
    """Verify that changing event details changes the hash."""
    logger = DictChainLogger()
    event1 = {"type": "TEST", "value": 42}
    event2 = {"type": "TEST", "value": 43}
    h1 = logger._compute_hash(event1)
    h2 = logger._compute_hash(event2)
    assert h1 != h2, "Different event dicts should produce different hashes"


def test_hash_excludes_hash_key():
    """Verify that the 'hash' key itself is excluded from hashing."""
    logger = DictChainLogger()
    event_without_hash = {"type": "TEST", "value": 42}
    event_with_hash = {"type": "TEST", "value": 42, "hash": "anything"}
    h1 = logger._compute_hash(event_without_hash)
    h2 = logger._compute_hash(event_with_hash)
    assert h1 == h2, "Hash key should not affect computed hash"


def test_log_returns_hash():
    """Verify that log() returns the event's hash."""
    logger = DictChainLogger()
    h1 = logger.log("TEST", {"value": 1})
    h2 = logger.log("TEST", {"value": 2})

    assert h1 == logger.chain[0]["hash"]
    assert h2 == logger.chain[1]["hash"]
    assert h1 != h2


def test_multiple_tampers_detects_first():
    """Verify that verify_integrity detects the first tampered event."""
    logger = DictChainLogger()
    for i in range(5):
        logger.log("EVENT", {"index": i})

    # Tamper multiple events
    logger.chain[1]["details"]["TAMPERED"] = True
    logger.chain[3]["details"]["TAMPERED"] = True

    intact, broken_at = logger.verify_integrity()
    assert intact is False
    assert broken_at == 1, "Should detect the first break"


if __name__ == "__main__":
    # Run all tests
    test_chain_intact_after_logging()
    test_tamper_breaks_chain()
    test_tamper_cascades_downstream()
    test_query_by_type()
    test_genesis_hash()
    test_hash_chain_linkage()
    test_summary()
    test_empty_chain_summary()
    test_hash_determinism()
    test_hash_sensitivity()
    test_hash_excludes_hash_key()
    test_log_returns_hash()
    test_multiple_tampers_detects_first()
    print("✅ All tests passed!")
