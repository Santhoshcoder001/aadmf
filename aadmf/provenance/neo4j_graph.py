"""Neo4j-backed provenance logger.

This module provides ``Neo4jLogger`` with the same operational interface as
``DictChainLogger`` while persisting events as graph nodes and edges.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase


class Neo4jLogger:
    """Tamper-evident provenance logger persisted in Neo4j.

    Public methods intentionally mirror ``DictChainLogger``:
    - log(event_type, details) -> str
    - verify_integrity() -> tuple[bool, int]
    - query_by_type(event_type) -> list
    - summary() -> dict
    - export_json(path) -> None
    """

    DEFAULT_URI = "bolt://localhost:7687"
    DEFAULT_USER = "neo4j"

    def __init__(
        self,
        password: str,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
    ) -> None:
        if not password:
            raise ValueError("Neo4j password is required")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.prev_hash = "GENESIS"
        self._ensure_schema()
        self._sync_prev_hash()

    @classmethod
    def from_config(cls, config: dict) -> "Neo4jLogger":
        """Build logger from config dict.

        Expected shape:
        config["provenance"]["neo4j"] = {
          "uri": "bolt://localhost:7687",
          "user": "neo4j",
          "password": "..."
        }
        """
        prov_cfg = (config or {}).get("provenance", {})
        neo4j_cfg = prov_cfg.get("neo4j", {})
        return cls(
            password=str(neo4j_cfg.get("password", "")),
            uri=str(neo4j_cfg.get("uri", cls.DEFAULT_URI)),
            user=str(neo4j_cfg.get("user", cls.DEFAULT_USER)),
        )

    def close(self) -> None:
        """Close Neo4j driver resources."""
        self.driver.close()

    def _ensure_schema(self) -> None:
        """Create lightweight constraints/indexes for event lookup."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT event_seq_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.seq IS UNIQUE")
            session.run("CREATE CONSTRAINT event_hash_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.hash IS UNIQUE")
            session.run("CREATE INDEX event_type_idx IF NOT EXISTS FOR (e:Event) ON (e.type)")

    def _sync_prev_hash(self) -> None:
        """Restore chain tip from persisted data so restarts continue linkage."""
        with self.driver.session() as session:
            rec = session.run(
                """
                MATCH (e:Event)
                RETURN e.hash AS hash
                ORDER BY e.seq DESC
                LIMIT 1
                """
            ).single()
        if rec and rec["hash"]:
            self.prev_hash = str(rec["hash"])

    def _next_seq(self) -> int:
        with self.driver.session() as session:
            rec = session.run(
                """
                MATCH (e:Event)
                RETURN coalesce(max(e.seq), -1) + 1 AS next_seq
                """
            ).single()
        return int(rec["next_seq"] if rec else 0)

    def _event_label(self, event_type: str) -> str:
        """Convert event type into a safe Neo4j secondary label."""
        label = re.sub(r"[^A-Za-z0-9_]", "_", str(event_type).upper())
        if not label:
            return "UNSPECIFIED_EVENT"
        if label[0].isdigit():
            return f"E_{label}"
        return label

    def _compute_hash(self, event: dict) -> str:
        """Compute SHA-256 hash using DictChainLogger's exact algorithm.

        Patent Claim 3 core: hash-chained tamper-proof provenance in graph form.
        """
        payload = json.dumps(event, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def log(self, event_type: str, details: dict) -> str:
        """Append one event node and return event hash."""
        seq = self._next_seq()
        event = {
            "seq": seq,
            "type": event_type,
            "ts": time.time(),
            "details": details,
            "prev_hash": self.prev_hash,
        }
        h = self._compute_hash(event)
        label = self._event_label(event_type)

        with self.driver.session() as session:
            create_query = (
                "CREATE (e:Event:" + label + " {"
                "seq: $seq, type: $type, ts: $ts, details: $details, "
                "hash: $hash, prev_hash: $prev_hash"
                "})"
            )
            session.run(
                create_query,
                seq=event["seq"],
                type=event["type"],
                ts=event["ts"],
                details=json.dumps(event["details"], default=str),
                hash=h,
                prev_hash=event["prev_hash"],
            )

            if seq > 0:
                session.run(
                    """
                    MATCH (prev:Event {seq: $prev_seq}), (curr:Event {seq: $curr_seq})
                    MERGE (prev)-[:PRECEDES]->(curr)
                    """,
                    prev_seq=seq - 1,
                    curr_seq=seq,
                )

        self.prev_hash = h
        return h

    def _read_chain(self) -> List[Dict[str, Any]]:
        """Load all events in sequence order from Neo4j."""
        with self.driver.session() as session:
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

        chain: List[Dict[str, Any]] = []
        for rec in records:
            chain.append(
                {
                    "seq": int(rec["seq"]),
                    "type": str(rec["type"]),
                    "ts": float(rec["ts"]),
                    "details": json.loads(rec["details"] or "{}"),
                    "hash": str(rec["hash"]),
                    "prev_hash": str(rec["prev_hash"]),
                }
            )
        return chain

    def verify_integrity(self) -> Tuple[bool, int]:
        """Verify full hash chain and linkage; return (intact, broken_at)."""
        chain = self._read_chain()
        prev = "GENESIS"

        for i, event in enumerate(chain):
            stored_hash = event["hash"]
            if event["prev_hash"] != prev:
                return False, i

            check_data = {k: v for k, v in event.items() if k != "hash"}
            computed = self._compute_hash(check_data)
            if computed != stored_hash:
                return False, i

            prev = stored_hash

        return True, -1

    def query_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Return all events of a given type in sequence order."""
        with self.driver.session() as session:
            records = list(
                session.run(
                    """
                    MATCH (e:Event {type: $event_type})
                    RETURN e.seq AS seq, e.type AS type, e.ts AS ts,
                           e.details AS details, e.hash AS hash, e.prev_hash AS prev_hash
                    ORDER BY e.seq ASC
                    """,
                    event_type=event_type,
                )
            )

        events: List[Dict[str, Any]] = []
        for rec in records:
            events.append(
                {
                    "seq": int(rec["seq"]),
                    "type": str(rec["type"]),
                    "ts": float(rec["ts"]),
                    "details": json.loads(rec["details"] or "{}"),
                    "hash": str(rec["hash"]),
                    "prev_hash": str(rec["prev_hash"]),
                }
            )
        return events

    def summary(self) -> Dict[str, Any]:
        """Return provenance summary with counts and chain endpoints."""
        with self.driver.session() as session:
            total_rec = session.run("MATCH (e:Event) RETURN count(e) AS total").single()
            type_recs = list(
                session.run(
                    """
                    MATCH (e:Event)
                    RETURN e.type AS type, count(*) AS count
                    """
                )
            )
            first_rec = session.run(
                """
                MATCH (e:Event)
                RETURN e.hash AS hash
                ORDER BY e.seq ASC
                LIMIT 1
                """
            ).single()
            last_rec = session.run(
                """
                MATCH (e:Event)
                RETURN e.hash AS hash
                ORDER BY e.seq DESC
                LIMIT 1
                """
            ).single()

        by_type = {str(r["type"]): int(r["count"]) for r in type_recs}
        return {
            "total_events": int(total_rec["total"] if total_rec else 0),
            "by_type": by_type,
            "first_hash": first_rec["hash"] if first_rec else None,
            "last_hash": last_rec["hash"] if last_rec else None,
        }

    def export_json(self, path: str) -> None:
        """Fallback export to JSON with the same event shape as DictChainLogger."""
        chain = self._read_chain()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chain, f, indent=2, default=str)


# Backward-compatible alias for older imports.
Neo4jGraphLogger = Neo4jLogger
