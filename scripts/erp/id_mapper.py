"""Bidirectional ID mapper: sim string IDs <-> integer PKs.

Used for stable integer primary keys across all CSV tables, enabling
both Postgres COPY and Neo4j LOAD CSV with integer matching.
"""

from __future__ import annotations

import json
from pathlib import Path


class IdMapper:
    """Maps simulation string IDs to sequential integer PKs."""

    def __init__(self) -> None:
        self._fwd: dict[str, dict[str, int]] = {}  # domain -> sim_id -> int
        self._rev: dict[str, dict[int, str]] = {}  # domain -> int -> sim_id
        self._counters: dict[str, int] = {}

    def get(self, domain: str, sim_id: str) -> int:
        """Get or create an integer PK for a sim string ID."""
        if domain not in self._fwd:
            self._fwd[domain] = {}
            self._rev[domain] = {}
            self._counters[domain] = 1

        if sim_id not in self._fwd[domain]:
            pk = self._counters[domain]
            self._fwd[domain][sim_id] = pk
            self._rev[domain][pk] = sim_id
            self._counters[domain] = pk + 1

        return self._fwd[domain][sim_id]

    def lookup(self, domain: str, sim_id: str) -> int | None:
        """Lookup without auto-creating. Returns None if missing."""
        return self._fwd.get(domain, {}).get(sim_id)

    def reverse(self, domain: str, pk: int) -> str | None:
        """Reverse lookup: integer PK -> sim string ID."""
        return self._rev.get(domain, {}).get(pk)

    def all_ids(self, domain: str) -> dict[str, int]:
        """Return all sim_id -> int mappings for a domain."""
        return dict(self._fwd.get(domain, {}))

    def count(self, domain: str) -> int:
        """Number of IDs registered in a domain."""
        return len(self._fwd.get(domain, {}))

    def save(self, path: Path) -> None:
        """Serialize all forward mappings to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._fwd, f, indent=2)
