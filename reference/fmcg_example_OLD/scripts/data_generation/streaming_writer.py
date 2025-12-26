"""
StreamingWriter - Memory-efficient streaming output for FMCG data generation.

Writes PostgreSQL COPY format directly to file with buffering and
FK-aware memory management via DependencyTracker.

Flow: Vectorized Engine → RealismMonitor → StreamingWriter → Disk

Usage:
    writer = StreamingWriter(output_path, buffer_size_mb=10)

    # Write rows as they're generated
    writer.write_batch("orders", order_rows, ORDER_COLUMNS)

    # Mark table as finalized when generation complete
    writer.finalize_table("orders")

    # Flush and close
    writer.close()
"""

from __future__ import annotations

import io
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

# Default FK dependency graph for safe table memory management
# Tables can only be purged after all downstream FK dependents are finalized
DEFAULT_FK_DEPENDENCIES: dict[str, list[str]] = {
    # Level 0-2: Reference data (few rows, keep in memory)
    "divisions": ["distribution_centers", "retail_locations"],
    "channels": ["retail_accounts", "orders"],
    "products": ["skus", "formulas"],
    "ingredients": ["supplier_ingredients", "formula_ingredients"],
    "packaging_types": ["skus"],
    "ports": ["route_segments"],
    "carriers": ["carrier_contracts", "shipments"],
    "emission_factors": ["shipment_emissions"],

    # Level 1: Master data
    "suppliers": ["supplier_ingredients", "purchase_orders", "certifications"],
    "plants": ["production_lines", "distribution_centers"],
    "production_lines": ["work_orders"],
    "carrier_contracts": ["carrier_rates"],
    "route_segments": ["route_segment_assignments", "shipment_legs"],

    # Level 2: Relationships
    "supplier_ingredients": ["purchase_order_lines"],
    "formulas": ["formula_ingredients", "work_orders"],
    "formula_ingredients": ["work_order_materials", "batch_ingredients"],
    "routes": ["route_segment_assignments", "shipments"],

    # Level 3: Locations
    "retail_accounts": ["retail_locations", "promotions"],
    "retail_locations": ["orders", "pos_sales", "inventory"],
    "distribution_centers": ["inventory", "shipments", "pick_waves"],

    # Level 4: SKUs and promotions
    "skus": ["sku_costs", "sku_substitutes", "order_lines", "pos_sales",
             "inventory", "demand_forecasts", "batches"],
    "promotions": ["promotion_skus", "promotion_accounts"],

    # Level 5: Orders/POs
    "purchase_orders": ["purchase_order_lines", "goods_receipts"],
    "goods_receipts": ["goods_receipt_lines"],
    "work_orders": ["work_order_materials", "batches"],

    # Level 6-7: Manufacturing
    "purchase_order_lines": ["goods_receipt_lines"],
    "batches": ["batch_ingredients", "batch_cost_ledger", "order_allocations",
                "shipment_lines"],

    # Level 8-9: Demand and orders
    "orders": ["order_lines", "order_allocations", "shipments"],
    "order_lines": ["order_allocations"],
    "pick_waves": ["pick_wave_orders"],

    # Level 10-11: Shipments
    "shipments": ["shipment_lines", "shipment_legs", "shipment_emissions"],

    # Level 12-13: Returns
    "rma_authorizations": ["returns"],
    "returns": ["return_lines"],
    "return_lines": ["disposition_logs"],
}


class DependencyTracker:
    """
    Tracks FK dependencies to determine when tables can be safely purged.

    Tables can only have their in-memory data cleared after all child tables
    (that depend on them via FK) have been finalized.

    Attributes:
        deps: Dict mapping parent tables to their child tables
        finalized: Set of tables that have completed generation
    """

    __slots__ = ("deps", "finalized", "row_counts")

    def __init__(self, deps: dict[str, list[str]] | None = None) -> None:
        """
        Initialize dependency tracker.

        Args:
            deps: Custom dependency graph (uses DEFAULT_FK_DEPENDENCIES if None)
        """
        self.deps = deps if deps is not None else DEFAULT_FK_DEPENDENCIES
        self.finalized: set[str] = set()
        self.row_counts: dict[str, int] = {}

    def can_purge(self, table: str) -> bool:
        """
        Check if a table's in-memory data can be safely purged.

        A table can be purged only if ALL its child tables are finalized.

        Args:
            table: Table name to check

        Returns:
            True if safe to purge, False otherwise
        """
        children = self.deps.get(table, [])
        return all(child in self.finalized for child in children)

    def mark_finalized(self, table: str, row_count: int = 0) -> None:
        """
        Mark a table as finalized (generation complete).

        Args:
            table: Table name
            row_count: Number of rows generated
        """
        self.finalized.add(table)
        self.row_counts[table] = row_count

    def is_finalized(self, table: str) -> bool:
        """Check if a table has been finalized."""
        return table in self.finalized

    def get_purgeable_tables(self) -> list[str]:
        """Return list of all tables that can currently be purged."""
        return [
            table for table in self.finalized
            if self.can_purge(table)
        ]

    def get_blocked_tables(self) -> dict[str, list[str]]:
        """
        Return dict of tables blocked from purging and their blocking children.

        Returns:
            Dict mapping blocked table names to list of unfinalized children
        """
        blocked = {}
        for table in self.finalized:
            if not self.can_purge(table):
                children = self.deps.get(table, [])
                unfinalized = [c for c in children if c not in self.finalized]
                if unfinalized:
                    blocked[table] = unfinalized
        return blocked

    def get_summary(self) -> dict[str, Any]:
        """Return summary of tracker state."""
        return {
            "finalized_count": len(self.finalized),
            "total_rows": sum(self.row_counts.values()),
            "purgeable": self.get_purgeable_tables(),
            "blocked": self.get_blocked_tables(),
        }


# =============================================================================
# COPY Format Helpers
# =============================================================================

def copy_str(val: str | None) -> str:
    """Format string for COPY format (tab-separated, \\N for NULL)."""
    if val is None:
        return "\\N"
    # Escape tabs, newlines, backslashes
    return val.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def copy_num(val: float | int | Decimal | None) -> str:
    """Format number for COPY format."""
    if val is None:
        return "\\N"
    return str(val)


def copy_bool(val: bool | None) -> str:
    """Format boolean for COPY format."""
    if val is None:
        return "\\N"
    return "t" if val else "f"


def copy_date(val: date | None) -> str:
    """Format date for COPY format."""
    if val is None:
        return "\\N"
    return val.isoformat()


def copy_timestamp(val: datetime | None) -> str:
    """Format timestamp for COPY format."""
    if val is None:
        return "\\N"
    return val.isoformat()


def format_copy_value(val: Any) -> str:
    """Auto-detect type and format value for COPY."""
    if val is None:
        return "\\N"
    if isinstance(val, bool):
        return copy_bool(val)
    if isinstance(val, datetime):
        return copy_timestamp(val)
    if isinstance(val, date):
        return copy_date(val)
    if isinstance(val, (int, float, Decimal)):
        return copy_num(val)
    return copy_str(str(val))


class StreamingWriter:
    """
    Memory-efficient streaming writer for PostgreSQL COPY format.

    Buffers output in memory and flushes to disk when buffer exceeds
    threshold. Integrates with DependencyTracker for FK-aware memory
    management.

    Attributes:
        output_path: Path to output SQL file
        buffer_size_bytes: Buffer flush threshold in bytes
        tracker: DependencyTracker for FK management
    """

    def __init__(
        self,
        output_path: Path | str,
        buffer_size_mb: float = 10.0,
        dependency_graph: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Initialize streaming writer.

        Args:
            output_path: Path to output SQL file
            buffer_size_mb: Buffer size in megabytes before flush
            dependency_graph: Custom FK dependency graph
        """
        self.output_path = Path(output_path)
        self.buffer_size_bytes = int(buffer_size_mb * 1024 * 1024)
        self.tracker = DependencyTracker(dependency_graph)

        # Buffer for accumulating output
        self._buffer = io.StringIO()
        self._buffer_bytes = 0

        # Track tables that have been started/completed
        self._started_tables: set[str] = set()
        self._completed_tables: set[str] = set()

        # File handle (opened on first write)
        self._file: TextIO | None = None
        self._total_bytes_written = 0

        # Row counts per table
        self._row_counts: dict[str, int] = {}

        # Max IDs per table for sequence reset
        self._max_ids: dict[str, int] = {}

    def _ensure_file_open(self) -> None:
        """Ensure output file is open."""
        if self._file is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.output_path, "w", encoding="utf-8")
            # Write header
            self._file.write("-- FMCG Data Generation - Streaming Output\n")
            self._file.write("-- Generated by StreamingWriter\n")
            self._file.write("--\n")
            self._file.write("-- Load with: psql -f seed.sql\n\n")
            self._file.write("SET client_encoding = 'UTF8';\n")
            self._file.write("SET standard_conforming_strings = on;\n\n")

    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        if self._buffer_bytes > 0:
            self._ensure_file_open()
            content = self._buffer.getvalue()
            self._file.write(content)
            self._total_bytes_written += len(content.encode("utf-8"))
            self._buffer = io.StringIO()
            self._buffer_bytes = 0

    def _write(self, text: str) -> None:
        """Write text to buffer, flushing if needed."""
        self._buffer.write(text)
        self._buffer_bytes += len(text.encode("utf-8"))

        if self._buffer_bytes >= self.buffer_size_bytes:
            self._flush_buffer()

    def start_table(self, table: str, columns: list[str]) -> None:
        """
        Start COPY block for a table.

        Args:
            table: Table name
            columns: List of column names
        """
        if table in self._started_tables:
            raise ValueError(f"Table {table} already started")

        self._started_tables.add(table)
        self._row_counts[table] = 0

        # Write COPY header
        cols_str = ", ".join(columns)
        self._write(f"\n-- Table: {table}\n")
        self._write(f"COPY {table} ({cols_str}) FROM stdin;\n")

    def write_row(self, table: str, row: dict[str, Any], columns: list[str]) -> None:
        """
        Write a single row to the COPY block.

        Args:
            table: Table name
            row: Row data as dict
            columns: Column order
        """
        if table not in self._started_tables:
            raise ValueError(f"Table {table} not started - call start_table first")
        if table in self._completed_tables:
            raise ValueError(f"Table {table} already completed")

        # Format row values
        values = [format_copy_value(row.get(col)) for col in columns]
        line = "\t".join(values) + "\n"
        self._write(line)

        self._row_counts[table] = self._row_counts.get(table, 0) + 1

        # Track max ID for sequence reset
        if "id" in row and row["id"] is not None:
            current_max = self._max_ids.get(table, 0)
            self._max_ids[table] = max(current_max, row["id"])

    def write_batch(
        self,
        table: str,
        rows: list[dict[str, Any]],
        columns: list[str],
        auto_start: bool = True,
    ) -> int:
        """
        Write a batch of rows to the COPY block.

        Args:
            table: Table name
            rows: List of row dicts
            columns: Column order
            auto_start: Automatically start table if not started

        Returns:
            Number of rows written
        """
        if not rows:
            return 0

        if auto_start and table not in self._started_tables:
            self.start_table(table, columns)

        # Build batch output in one go for efficiency
        lines = []
        for row in rows:
            values = [format_copy_value(row.get(col)) for col in columns]
            lines.append("\t".join(values))

            # Track max ID
            if "id" in row and row["id"] is not None:
                current_max = self._max_ids.get(table, 0)
                self._max_ids[table] = max(current_max, row["id"])

        self._write("\n".join(lines) + "\n")
        self._row_counts[table] = self._row_counts.get(table, 0) + len(rows)

        return len(rows)

    def end_table(self, table: str, write_sequence_reset: bool = True) -> None:
        """
        End COPY block for a table.

        Args:
            table: Table name
            write_sequence_reset: Whether to write sequence reset statement
        """
        if table not in self._started_tables:
            raise ValueError(f"Table {table} not started")
        if table in self._completed_tables:
            raise ValueError(f"Table {table} already completed")

        self._completed_tables.add(table)

        # Write COPY terminator
        self._write("\\.\n")

        # Write sequence reset if we have a max ID
        if write_sequence_reset and table in self._max_ids:
            max_id = self._max_ids[table]
            self._write(f"SELECT setval('{table}_id_seq', {max_id});\n")

        # Mark as finalized in tracker
        self.tracker.mark_finalized(table, self._row_counts.get(table, 0))

    def finalize_table(
        self,
        table: str,
        rows: list[dict[str, Any]] | None = None,
        columns: list[str] | None = None,
    ) -> int:
        """
        Write all rows and finalize a table in one call.

        Convenience method that handles start, write, and end.

        Args:
            table: Table name
            rows: Optional rows to write (if not already written)
            columns: Column order (required if rows provided)

        Returns:
            Number of rows written
        """
        count = 0

        if rows is not None:
            if columns is None:
                raise ValueError("columns required when rows provided")
            if table not in self._started_tables:
                self.start_table(table, columns)
            count = self.write_batch(table, rows, columns, auto_start=False)

        if table not in self._completed_tables:
            self.end_table(table)

        return count

    def can_purge_data(self, table: str) -> bool:
        """
        Check if a table's in-memory data can be safely purged.

        Delegates to DependencyTracker.
        """
        return self.tracker.can_purge(table)

    def get_stats(self) -> dict[str, Any]:
        """Return writer statistics."""
        return {
            "output_path": str(self.output_path),
            "total_bytes_written": self._total_bytes_written,
            "buffer_bytes": self._buffer_bytes,
            "tables_started": len(self._started_tables),
            "tables_completed": len(self._completed_tables),
            "row_counts": dict(self._row_counts),
            "total_rows": sum(self._row_counts.values()),
            "tracker_summary": self.tracker.get_summary(),
        }

    def flush(self) -> None:
        """Force flush buffer to disk."""
        self._flush_buffer()

    def close(self) -> None:
        """Flush buffer and close file."""
        self._flush_buffer()

        if self._file is not None:
            # Write summary comment
            self._file.write("\n-- Generation Summary\n")
            total_rows = sum(self._row_counts.values())
            self._file.write(f"-- Total rows: {total_rows:,}\n")
            self._file.write(f"-- Tables: {len(self._completed_tables)}\n")

            self._file.close()
            self._file = None

    def __enter__(self) -> "StreamingWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
