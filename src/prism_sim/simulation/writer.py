"""
Simulation data export with streaming support for large-scale runs.

[Task 7.3] Implements streaming CSV/Parquet writers to handle 365-day runs
without memory bottlenecks.
"""

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prism_sim.network.core import Batch, Order, Shipment

# Parquet support is optional - only import if available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class StreamingCSVWriter:
    """
    Streaming CSV writer that writes rows incrementally to disk.

    Opens file handle on first write and keeps it open until close().
    This prevents memory accumulation for high-volume tables.
    """

    def __init__(self, filepath: Path, fieldnames: list[str]) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._file: Any = None
        self._writer: csv.DictWriter[str] | None = None
        self._row_count = 0

    def _ensure_open(self) -> None:
        """Lazily open file and write header on first write."""
        if self._file is None:
            self._file = open(self.filepath, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        """Write a single row to the CSV file."""
        self._ensure_open()
        if self._writer:
            self._writer.writerow(row)
            self._row_count += 1

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Write multiple rows to the CSV file."""
        self._ensure_open()
        if self._writer:
            self._writer.writerows(rows)
            self._row_count += len(rows)

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    @property
    def row_count(self) -> int:
        """Return the number of rows written."""
        return self._row_count


class StreamingParquetWriter:
    """
    Streaming Parquet writer that batches rows and flushes periodically.

    Accumulates rows in memory until batch_size is reached, then writes
    a row group to the Parquet file. This balances memory usage with
    compression efficiency.
    """

    def __init__(
        self,
        filepath: Path,
        schema: "pa.Schema",
        batch_size: int = 10000,
    ) -> None:
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "pyarrow is required for Parquet support. "
                "Install with: poetry add pyarrow"
            )
        self.filepath = filepath
        self.schema = schema
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._writer: Any = None
        self._row_count = 0

    def _ensure_open(self) -> None:
        """Lazily open Parquet writer on first write."""
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.filepath, self.schema)

    def _flush_buffer(self) -> None:
        """Write buffered rows as a row group."""
        if not self._buffer:
            return
        self._ensure_open()
        # Convert buffer to columnar format
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self._row_count += len(self._buffer)
        self._buffer = []

    def write_row(self, row: dict[str, Any]) -> None:
        """Add a row to the buffer, flushing if batch_size is reached."""
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self._flush_buffer()

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Add multiple rows to the buffer."""
        self._buffer.extend(rows)
        while len(self._buffer) >= self.batch_size:
            # Flush in chunks
            chunk = self._buffer[: self.batch_size]
            self._buffer = self._buffer[self.batch_size :]
            self._ensure_open()
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self._writer.write_table(table)
            self._row_count += len(chunk)

    def flush(self) -> None:
        """Flush any remaining buffered data."""
        self._flush_buffer()

    def close(self) -> None:
        """Flush remaining data and close the file."""
        self._flush_buffer()
        if self._writer:
            self._writer.close()
            self._writer = None

    @property
    def row_count(self) -> int:
        """Return the number of rows written."""
        return self._row_count + len(self._buffer)


# Schema definitions for Parquet writers
ORDER_FIELDS = [
    "order_id",
    "day",
    "source_id",
    "target_id",
    "product_id",
    "quantity",
    "status",
]

SHIPMENT_FIELDS = [
    "shipment_id",
    "creation_day",
    "arrival_day",
    "source_id",
    "target_id",
    "product_id",
    "quantity",
    "total_weight_kg",
    "total_volume_m3",
    "status",
]

BATCH_FIELDS = [
    "batch_id",
    "plant_id",
    "product_id",
    "day_produced",
    "quantity",
    "yield_pct",
    "status",
    "notes",
]

INVENTORY_FIELDS = [
    "day",
    "node_id",
    "product_id",
    "perceived_inventory",
    "actual_inventory",
]


def _get_parquet_schema(table_name: str) -> "pa.Schema":
    """Return PyArrow schema for a given table."""
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow not available")

    schemas = {
        "orders": pa.schema(
            [
                ("order_id", pa.string()),
                ("day", pa.int32()),
                ("source_id", pa.string()),
                ("target_id", pa.string()),
                ("product_id", pa.string()),
                ("quantity", pa.float64()),
                ("status", pa.string()),
            ]
        ),
        "shipments": pa.schema(
            [
                ("shipment_id", pa.string()),
                ("creation_day", pa.int32()),
                ("arrival_day", pa.int32()),
                ("source_id", pa.string()),
                ("target_id", pa.string()),
                ("product_id", pa.string()),
                ("quantity", pa.float64()),
                ("total_weight_kg", pa.float64()),
                ("total_volume_m3", pa.float64()),
                ("status", pa.string()),
            ]
        ),
        "batches": pa.schema(
            [
                ("batch_id", pa.string()),
                ("plant_id", pa.string()),
                ("product_id", pa.string()),
                ("day_produced", pa.int32()),
                ("quantity", pa.float64()),
                ("yield_pct", pa.float64()),
                ("status", pa.string()),
                ("notes", pa.string()),
            ]
        ),
        "inventory": pa.schema(
            [
                ("day", pa.int32()),
                ("node_id", pa.string()),
                ("product_id", pa.string()),
                ("perceived_inventory", pa.float64()),
                ("actual_inventory", pa.float64()),
            ]
        ),
    }
    return schemas[table_name]


class SimulationWriter:
    """
    Handles data export for the Prism Digital Twin.

    Supports two modes:
    - **Buffered (default):** Accumulates data in memory, writes at end.
      Suitable for short runs (<90 days) or when logging is disabled.
    - **Streaming:** Writes data incrementally to disk as it arrives.
      Required for long runs (365+ days) to prevent memory exhaustion.

    Output formats:
    - **CSV:** Universal compatibility, streaming support.
    - **Parquet:** Columnar compression, fast analytics (requires pyarrow).
    """

    def __init__(  # noqa: PLR0913
        self,
        output_dir: str = "data/output",
        enable_logging: bool = False,
        streaming: bool = False,
        output_format: str = "csv",
        parquet_batch_size: int = 10000,
        inventory_sample_rate: int = 1,
    ) -> None:
        """
        Initialize the simulation writer.

        Args:
            output_dir: Directory for output files.
            enable_logging: If False, all logging is skipped (fast mode).
            streaming: If True, write data incrementally instead of buffering.
            output_format: "csv" or "parquet" (parquet requires pyarrow).
            parquet_batch_size: Rows to buffer before writing a Parquet row group.
            inventory_sample_rate: Log inventory every N days (1=daily, 7=weekly).
        """
        self.output_dir = Path(output_dir)
        self.enable_logging = enable_logging
        self.streaming = streaming
        self.output_format = output_format
        self.parquet_batch_size = parquet_batch_size
        self.inventory_sample_rate = inventory_sample_rate

        # Validate format
        if output_format == "parquet" and not PARQUET_AVAILABLE:
            raise ImportError(
                "Parquet format requested but pyarrow is not installed. "
                "Install with: poetry add pyarrow"
            )

        # Create output directory if logging enabled
        if self.enable_logging:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize writers or buffers
        self._orders_writer: StreamingCSVWriter | StreamingParquetWriter | None = None
        self._shipments_writer: StreamingCSVWriter | StreamingParquetWriter | None = (
            None
        )
        self._batches_writer: StreamingCSVWriter | StreamingParquetWriter | None = None
        self._inventory_writer: StreamingCSVWriter | StreamingParquetWriter | None = (
            None
        )

        # Buffered mode storage (legacy compatibility)
        self.orders: list[dict[str, Any]] = []
        self.shipments: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []

        # Initialize streaming writers if enabled
        if self.enable_logging and self.streaming:
            self._init_streaming_writers()

    def _init_streaming_writers(self) -> None:
        """Initialize streaming writers for each table."""
        ext = ".parquet" if self.output_format == "parquet" else ".csv"

        if self.output_format == "parquet":
            self._orders_writer = StreamingParquetWriter(
                self.output_dir / f"orders{ext}",
                _get_parquet_schema("orders"),
                self.parquet_batch_size,
            )
            self._shipments_writer = StreamingParquetWriter(
                self.output_dir / f"shipments{ext}",
                _get_parquet_schema("shipments"),
                self.parquet_batch_size,
            )
            self._batches_writer = StreamingParquetWriter(
                self.output_dir / f"batches{ext}",
                _get_parquet_schema("batches"),
                self.parquet_batch_size,
            )
            self._inventory_writer = StreamingParquetWriter(
                self.output_dir / f"inventory{ext}",
                _get_parquet_schema("inventory"),
                self.parquet_batch_size,
            )
        else:
            self._orders_writer = StreamingCSVWriter(
                self.output_dir / f"orders{ext}",
                ORDER_FIELDS,
            )
            self._shipments_writer = StreamingCSVWriter(
                self.output_dir / f"shipments{ext}",
                SHIPMENT_FIELDS,
            )
            self._batches_writer = StreamingCSVWriter(
                self.output_dir / f"batches{ext}",
                BATCH_FIELDS,
            )
            self._inventory_writer = StreamingCSVWriter(
                self.output_dir / f"inventory{ext}",
                INVENTORY_FIELDS,
            )

    def log_orders(self, orders: list[Order], day: int) -> None:
        """Log order data for a given day."""
        if not self.enable_logging:
            return

        rows = []
        for order in orders:
            for line in order.lines:
                row = {
                    "order_id": order.id,
                    "day": day,
                    "source_id": order.source_id,
                    "target_id": order.target_id,
                    "product_id": line.product_id,
                    "quantity": line.quantity,
                    "status": order.status,
                }
                rows.append(row)

        if self.streaming and self._orders_writer:
            self._orders_writer.write_rows(rows)
        else:
            self.orders.extend(rows)

    def log_shipments(self, shipments: list[Shipment], day: int) -> None:
        """Log shipment data for a given day."""
        if not self.enable_logging:
            return

        rows = []
        for s in shipments:
            for line in s.lines:
                row = {
                    "shipment_id": s.id,
                    "creation_day": s.creation_day,
                    "arrival_day": s.arrival_day,
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "product_id": line.product_id,
                    "quantity": line.quantity,
                    "total_weight_kg": s.total_weight_kg,
                    "total_volume_m3": s.total_volume_m3,
                    "status": s.status.value,
                }
                rows.append(row)

        if self.streaming and self._shipments_writer:
            self._shipments_writer.write_rows(rows)
        else:
            self.shipments.extend(rows)

    def log_batches(self, batches: list[Batch], day: int) -> None:
        """Log production batch data for a given day."""
        if not self.enable_logging:
            return

        rows = []
        for b in batches:
            row = {
                "batch_id": b.id,
                "plant_id": b.plant_id,
                "product_id": b.product_id,
                "day_produced": day,
                "quantity": b.quantity_cases,
                "yield_pct": b.yield_percent,
                "status": b.status.value,
                "notes": b.notes or "",
            }
            rows.append(row)

        if self.streaming and self._batches_writer:
            self._batches_writer.write_rows(rows)
        else:
            self.batches.extend(rows)

    def log_inventory(self, state: Any, world: Any, day: int) -> None:
        """Log inventory snapshot for a given day."""
        if not self.enable_logging:
            return

        # Apply sampling rate to reduce data volume
        if day % self.inventory_sample_rate != 0:
            return

        rows = []
        for node_id, node_idx in state.node_id_to_idx.items():
            for prod_id, prod_idx in state.product_id_to_idx.items():
                perceived = float(state.perceived_inventory[node_idx, prod_idx])
                actual = float(state.actual_inventory[node_idx, prod_idx])
                # Only log non-zero inventory to reduce volume
                if perceived != 0 or actual != 0:
                    row = {
                        "day": day,
                        "node_id": node_id,
                        "product_id": prod_id,
                        "perceived_inventory": perceived,
                        "actual_inventory": actual,
                    }
                    rows.append(row)

        if self.streaming and self._inventory_writer:
            self._inventory_writer.write_rows(rows)
        else:
            self.inventory_history.extend(rows)

    def flush(self) -> None:
        """Flush all streaming writers to disk."""
        if self.streaming:
            if self._orders_writer:
                self._orders_writer.flush()
            if self._shipments_writer:
                self._shipments_writer.flush()
            if self._batches_writer:
                self._batches_writer.flush()
            if self._inventory_writer:
                self._inventory_writer.flush()

    def save(self, final_metrics: dict[str, Any]) -> None:
        """Write all data to disk and close file handles."""
        if not self.enable_logging:
            print("SimulationWriter: Logging disabled, skipping export.")
            return

        if self.streaming:
            # Close streaming writers
            self._close_streaming_writers()
            row_counts = self._get_row_counts()
            print(
                f"Streaming export complete: "
                f"Orders={row_counts['orders']:,}, "
                f"Shipments={row_counts['shipments']:,}, "
                f"Batches={row_counts['batches']:,}, "
                f"Inventory={row_counts['inventory']:,}"
            )
        else:
            # Buffered mode: write accumulated data
            self._write_csv("orders.csv", self.orders)
            self._write_csv("shipments.csv", self.shipments)
            self._write_csv("batches.csv", self.batches)
            self._write_csv("inventory.csv", self.inventory_history)
            print(f"Buffered export complete to {self.output_dir}")

        # Always write metrics as JSON
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)

        print(f"Simulation data exported to {self.output_dir}")

    def _close_streaming_writers(self) -> None:
        """Close all streaming writer file handles."""
        if self._orders_writer:
            self._orders_writer.close()
        if self._shipments_writer:
            self._shipments_writer.close()
        if self._batches_writer:
            self._batches_writer.close()
        if self._inventory_writer:
            self._inventory_writer.close()

    def _get_row_counts(self) -> dict[str, int]:
        """Get row counts from streaming writers."""
        return {
            "orders": self._orders_writer.row_count if self._orders_writer else 0,
            "shipments": (
                self._shipments_writer.row_count if self._shipments_writer else 0
            ),
            "batches": self._batches_writer.row_count if self._batches_writer else 0,
            "inventory": (
                self._inventory_writer.row_count if self._inventory_writer else 0
            ),
        }

    def _write_csv(self, filename: str, data: list[dict[str, Any]]) -> None:
        """Write buffered data to CSV file."""
        if not data:
            return

        filepath = self.output_dir / filename
        keys = list(data[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    @contextmanager
    def streaming_context(self) -> Iterator["SimulationWriter"]:
        """Context manager for streaming mode - ensures cleanup on exit."""
        try:
            yield self
        finally:
            if self.streaming:
                self._close_streaming_writers()
