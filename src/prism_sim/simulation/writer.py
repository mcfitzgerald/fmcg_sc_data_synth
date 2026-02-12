"""
Simulation data export with streaming support for large-scale runs.

[Task 7.3] Implements streaming CSV/Parquet writers to handle 365-day runs
without memory bottlenecks.
"""

import csv
import json
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from prism_sim.network.core import Batch, Order, ProductionOrder, Return, Shipment

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
        self._writer: Any = None  # csv.writer instance
        self._row_count = 0

    def _ensure_open(self) -> None:
        """Lazily open file and write header on first write."""
        if self._file is None:
            self._file = open(self.filepath, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.fieldnames)

    def write_row(self, row: dict[str, Any]) -> None:
        """Write a single row to the CSV file."""
        self._ensure_open()
        if self._writer:
            fn = self.fieldnames
            self._writer.writerow(tuple(row[k] for k in fn))
            self._row_count += 1

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Write multiple rows to the CSV file."""
        self._ensure_open()
        if self._writer:
            fn = self.fieldnames
            self._writer.writerows(
                tuple(row[k] for k in fn) for row in rows
            )
            self._row_count += len(rows)

    def write_raw_rows(self, rows: Any, count: int) -> None:
        """Write pre-ordered rows (tuples/iterables) without dict key lookup."""
        self._ensure_open()
        if self._writer:
            self._writer.writerows(rows)
            self._row_count += count

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

    def write_arrow_table(self, table: "pa.Table") -> None:
        """Write a pre-built PyArrow table directly, bypassing the dict buffer."""
        self._ensure_open()
        self._writer.write_table(table)
        self._row_count += len(table)

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


class _InventoryBatch(NamedTuple):
    """Lightweight message passed from main thread to background writer."""

    day: int
    node_indices: np.ndarray  # int64 from np.nonzero
    prod_indices: np.ndarray  # int64 from np.nonzero
    perc_vals: np.ndarray  # float32
    act_vals: np.ndarray  # float32
    row_count: int


class ThreadedParquetWriter:
    """
    Background-threaded Parquet writer for high-volume inventory data.

    Main thread (log_inventory): does numpy ops (mask, nonzero, fancy-index),
    puts raw numpy arrays on a bounded queue.

    Background thread: dequeues arrays, builds DictionaryArray columns,
    constructs pa.Table, calls write_table() (which releases GIL for C++
    encoding/compression/IO).
    """

    def __init__(
        self,
        filepath: Path,
        schema: "pa.Schema",
        node_ids: list[str],
        product_ids: list[str],
        maxsize: int = 4,
    ) -> None:
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for ThreadedParquetWriter")
        self._inner = pq.ParquetWriter(filepath, schema)
        self._schema = schema
        self._queue: queue.Queue[_InventoryBatch | None] = queue.Queue(
            maxsize=maxsize
        )
        self._row_count = 0
        self._error: BaseException | None = None

        # Pre-build PyArrow dictionaries (immutable for the entire sim)
        self._pa_node_dict = pa.array(node_ids, type=pa.string())
        self._pa_prod_dict = pa.array(product_ids, type=pa.string())

        self._thread = threading.Thread(
            target=self._writer_loop, name="inventory-parquet-writer", daemon=True
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        """Background thread: consume batches and write Parquet row groups."""
        try:
            while True:
                batch = self._queue.get()
                if batch is None:
                    self._queue.task_done()
                    break  # Sentinel — shutdown requested
                try:
                    self._write_batch(batch)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc

    def _write_batch(self, batch: _InventoryBatch) -> None:
        """Build a pa.Table from raw numpy arrays and write it."""
        n = batch.row_count
        day_col = pa.array(np.full(n, batch.day, dtype=np.int32))
        node_col = pa.DictionaryArray.from_arrays(
            batch.node_indices.astype(np.int32), self._pa_node_dict
        )
        prod_col = pa.DictionaryArray.from_arrays(
            batch.prod_indices.astype(np.int32), self._pa_prod_dict
        )
        perc_col = pa.array(batch.perc_vals, type=pa.float32())
        act_col = pa.array(batch.act_vals, type=pa.float32())

        table = pa.table(
            {
                "day": day_col,
                "node_id": node_col,
                "product_id": prod_col,
                "perceived_inventory": perc_col,
                "actual_inventory": act_col,
            },
            schema=self._schema,
        )
        self._inner.write_table(table)
        self._row_count += n

    def submit(self, batch: _InventoryBatch) -> None:
        """Enqueue a batch for the background writer. Blocks if queue is full."""
        if self._error is not None:
            raise RuntimeError(
                "Background inventory writer failed"
            ) from self._error
        self._queue.put(batch)

    def flush(self) -> None:
        """Wait for the background queue to drain."""
        self._queue.join()

    def close(self) -> None:
        """Shutdown the background thread and close the Parquet writer."""
        self._queue.put(None)  # Sentinel
        self._thread.join(timeout=30)
        self._inner.close()
        if self._error is not None:
            raise RuntimeError(
                "Background inventory writer failed"
            ) from self._error

    @property
    def row_count(self) -> int:
        """Return the number of rows written (approximate — thread-safe read)."""
        return self._row_count


# Schema definitions for Parquet writers
ORDER_FIELDS = [
    "order_id",
    "day",
    "source_id",
    "target_id",
    "product_id",
    "quantity",
    "status",
    "requested_date",
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
    "emissions_kg",
]

BATCH_FIELDS = [
    "batch_id",
    "production_order_id",
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

PRODUCTION_ORDER_FIELDS = [
    "po_id",
    "plant_id",
    "product_id",
    "quantity",
    "creation_day",
    "due_day",
    "status",
]

FORECAST_FIELDS = [
    "day",
    "product_id",
    "forecast_quantity",
]

RETURNS_FIELDS = [
    "rma_id",
    "day",
    "source_id",
    "target_id",
    "product_id",
    "quantity",
    "disposition",
    "status",
]

BATCH_INGREDIENT_FIELDS = [
    "batch_id",
    "ingredient_id",
    "quantity_kg",
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
                ("requested_date", pa.int32()),
            ]
        ),
        "forecasts": pa.schema(
            [
                ("day", pa.int32()),
                ("product_id", pa.string()),
                ("forecast_quantity", pa.float64()),
            ]
        ),
        "production_orders": pa.schema(
            [
                ("po_id", pa.string()),
                ("plant_id", pa.string()),
                ("product_id", pa.string()),
                ("quantity", pa.float64()),
                ("creation_day", pa.int32()),
                ("due_day", pa.int32()),
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
                ("emissions_kg", pa.float64()),
            ]
        ),
        "batches": pa.schema(
            [
                ("batch_id", pa.string()),
                ("production_order_id", pa.string()),
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
                ("node_id", pa.dictionary(pa.int32(), pa.string())),
                ("product_id", pa.dictionary(pa.int32(), pa.string())),
                ("perceived_inventory", pa.float32()),
                ("actual_inventory", pa.float32()),
            ]
        ),
        "returns": pa.schema(
            [
                ("rma_id", pa.string()),
                ("day", pa.int32()),
                ("source_id", pa.string()),
                ("target_id", pa.string()),
                ("product_id", pa.string()),
                ("quantity", pa.float64()),
                ("disposition", pa.string()),
                ("status", pa.string()),
            ]
        ),
        "batch_ingredients": pa.schema(
            [
                ("batch_id", pa.string()),
                ("ingredient_id", pa.string()),
                ("quantity_kg", pa.float64()),
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

    def __init__(
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
        self._production_orders_writer: (
            StreamingCSVWriter | StreamingParquetWriter | None
        ) = None
        self._shipments_writer: StreamingCSVWriter | StreamingParquetWriter | None = (
            None
        )
        self._batches_writer: StreamingCSVWriter | StreamingParquetWriter | None = None
        self._inventory_writer: (
            StreamingCSVWriter | StreamingParquetWriter | ThreadedParquetWriter | None
        ) = None
        self._forecasts_writer: StreamingCSVWriter | StreamingParquetWriter | None = (
            None
        )
        self._returns_writer: StreamingCSVWriter | StreamingParquetWriter | None = (
            None
        )
        self._batch_ingredients_writer: (
            StreamingCSVWriter | StreamingParquetWriter | None
        ) = None

        # Buffered mode storage (legacy compatibility)
        self.orders: list[dict[str, Any]] = []
        self.production_orders: list[dict[str, Any]] = []
        self.shipments: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.forecasts: list[dict[str, Any]] = []
        self.returns: list[dict[str, Any]] = []
        self.batch_ingredients: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []

        # Cached reverse-index arrays for vectorized inventory export
        self._node_ids_arr: np.ndarray | None = None
        self._prod_ids_arr: np.ndarray | None = None

        # Sorted ID lists for ThreadedParquetWriter (built lazily)
        self._sorted_node_ids: list[str] | None = None
        self._sorted_prod_ids: list[str] | None = None

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
            self._production_orders_writer = StreamingParquetWriter(
                self.output_dir / f"production_orders{ext}",
                _get_parquet_schema("production_orders"),
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
            self._forecasts_writer = StreamingParquetWriter(
                self.output_dir / f"forecasts{ext}",
                _get_parquet_schema("forecasts"),
                self.parquet_batch_size,
            )
            # Inventory writer is created lazily in log_inventory() because
            # ThreadedParquetWriter needs node/product ID lists from state.
            self._returns_writer = StreamingParquetWriter(
                self.output_dir / f"returns{ext}",
                _get_parquet_schema("returns"),
                self.parquet_batch_size,
            )
            self._batch_ingredients_writer = StreamingParquetWriter(
                self.output_dir / f"batch_ingredients{ext}",
                _get_parquet_schema("batch_ingredients"),
                self.parquet_batch_size,
            )
        else:
            self._orders_writer = StreamingCSVWriter(
                self.output_dir / f"orders{ext}",
                ORDER_FIELDS,
            )
            self._production_orders_writer = StreamingCSVWriter(
                self.output_dir / f"production_orders{ext}",
                PRODUCTION_ORDER_FIELDS,
            )
            self._shipments_writer = StreamingCSVWriter(
                self.output_dir / f"shipments{ext}",
                SHIPMENT_FIELDS,
            )
            self._batches_writer = StreamingCSVWriter(
                self.output_dir / f"batches{ext}",
                BATCH_FIELDS,
            )
            self._forecasts_writer = StreamingCSVWriter(
                self.output_dir / f"forecasts{ext}",
                FORECAST_FIELDS,
            )
            self._inventory_writer = StreamingCSVWriter(
                self.output_dir / f"inventory{ext}",
                INVENTORY_FIELDS,
            )
            self._returns_writer = StreamingCSVWriter(
                self.output_dir / f"returns{ext}",
                RETURNS_FIELDS,
            )
            self._batch_ingredients_writer = StreamingCSVWriter(
                self.output_dir / f"batch_ingredients{ext}",
                BATCH_INGREDIENT_FIELDS,
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
                    "requested_date": order.requested_date,
                }
                rows.append(row)

        if self.streaming and self._orders_writer:
            self._orders_writer.write_rows(rows)
        else:
            self.orders.extend(rows)

    def log_production_orders(self, orders: list[ProductionOrder], day: int) -> None:
        """Log production order data for a given day."""
        if not self.enable_logging:
            return

        rows = []
        for order in orders:
            row = {
                "po_id": order.id,
                "plant_id": order.plant_id,
                "product_id": order.product_id,
                "quantity": order.quantity_cases,
                "creation_day": order.creation_day,
                "due_day": order.due_day,
                "status": order.status.value,
            }
            rows.append(row)

        if self.streaming and self._production_orders_writer:
            self._production_orders_writer.write_rows(rows)
        else:
            self.production_orders.extend(rows)

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
                    "emissions_kg": s.emissions_kg,
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
                "production_order_id": b.production_order_id,
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

    def log_batch_ingredients(self, batches: list[Batch], day: int) -> None:
        """Log ingredient consumption for production batches."""
        if not self.enable_logging:
            return

        rows = []
        for b in batches:
            for ing_id, qty_kg in b.ingredients_consumed.items():
                row = {
                    "batch_id": b.id,
                    "ingredient_id": ing_id,
                    "quantity_kg": qty_kg,
                }
                rows.append(row)

        if self.streaming and self._batch_ingredients_writer:
            self._batch_ingredients_writer.write_rows(rows)
        else:
            self.batch_ingredients.extend(rows)

    def log_forecasts(self, forecast_vec: np.ndarray, state: Any, day: int) -> None:
        """Log deterministic demand forecast for a given day."""
        if not self.enable_logging:
            return

        rows = []
        for p_idx, qty in enumerate(forecast_vec):
            if qty > 0:
                prod_id = state.product_idx_to_id[p_idx]
                row = {
                    "day": day,
                    "product_id": prod_id,
                    "forecast_quantity": float(qty),
                }
                rows.append(row)

        if self.streaming and self._forecasts_writer:
            self._forecasts_writer.write_rows(rows)
        else:
            self.forecasts.extend(rows)

    def log_returns(self, returns: list[Return], day: int) -> None:
        """Log returns (RMAs)."""
        if not self.enable_logging:
            return

        rows = []
        for r in returns:
            for line in r.lines:
                row = {
                    "rma_id": r.id,
                    "day": day,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "product_id": line.product_id,
                    "quantity": line.quantity_cases,
                    "disposition": line.disposition,
                    "status": r.status.value,
                }
                rows.append(row)

        if self.streaming and self._returns_writer:
            self._returns_writer.write_rows(rows)
        else:
            self.returns.extend(rows)

    def _init_threaded_inventory_writer(self, state: Any) -> None:
        """Lazily create the ThreadedParquetWriter on first log_inventory call.

        Requires node/product ID lists from state, which aren't available at
        SimulationWriter construction time.
        """
        # Build sorted ID lists (index-order) for DictionaryArray dictionaries
        n_nodes = len(state.node_id_to_idx)
        n_prods = len(state.product_id_to_idx)
        node_ids: list[str] = [""] * n_nodes
        for nid, idx in state.node_id_to_idx.items():
            node_ids[idx] = nid
        prod_ids: list[str] = [""] * n_prods
        for pid, idx in state.product_id_to_idx.items():
            prod_ids[idx] = pid

        self._sorted_node_ids = node_ids
        self._sorted_prod_ids = prod_ids

        self._inventory_writer = ThreadedParquetWriter(
            filepath=self.output_dir / "inventory.parquet",
            schema=_get_parquet_schema("inventory"),
            node_ids=node_ids,
            product_ids=prod_ids,
        )

    def log_inventory(self, state: Any, world: Any, day: int) -> None:
        """Log inventory snapshot for a given day (vectorized)."""
        if not self.enable_logging:
            return

        # Apply sampling rate to reduce data volume
        if day % self.inventory_sample_rate != 0:
            return

        # Lazily create ThreadedParquetWriter for parquet streaming
        if (
            self.streaming
            and self.output_format == "parquet"
            and self._inventory_writer is None
        ):
            self._init_threaded_inventory_writer(state)

        # Build reverse-index arrays on first call (cached, for CSV/buffered paths)
        if self._node_ids_arr is None:
            self._node_ids_arr = np.empty(len(state.node_id_to_idx), dtype=object)
            for nid, idx in state.node_id_to_idx.items():
                self._node_ids_arr[idx] = nid
            self._prod_ids_arr = np.empty(len(state.product_id_to_idx), dtype=object)
            for pid, idx in state.product_id_to_idx.items():
                self._prod_ids_arr[idx] = pid

        # Vectorized: find non-zero entries using numpy
        perceived = state.perceived_inventory
        actual = state.actual_inventory
        mask = (perceived != 0) | (actual != 0)
        node_indices, prod_indices = np.nonzero(mask)

        if len(node_indices) == 0:
            return

        n = len(node_indices)
        perc_vals = perceived[node_indices, prod_indices]
        act_vals = actual[node_indices, prod_indices]

        if self.streaming and self._inventory_writer:
            if self.output_format == "parquet":
                # Threaded path: submit raw numpy arrays to background writer
                assert isinstance(self._inventory_writer, ThreadedParquetWriter)
                self._inventory_writer.submit(
                    _InventoryBatch(
                        day=day,
                        node_indices=node_indices,
                        prod_indices=prod_indices,
                        perc_vals=perc_vals.astype(np.float32, copy=False),
                        act_vals=act_vals.astype(np.float32, copy=False),
                        row_count=n,
                    )
                )
            else:
                # CSV: write pre-ordered tuples via zip (no dict overhead)
                assert isinstance(self._inventory_writer, StreamingCSVWriter)
                assert self._prod_ids_arr is not None
                node_col = self._node_ids_arr[node_indices]
                prod_col = self._prod_ids_arr[prod_indices]
                day_col = np.full(n, day, dtype=np.int32)
                self._inventory_writer.write_raw_rows(
                    zip(day_col, node_col, prod_col, perc_vals, act_vals, strict=True),
                    count=n,
                )
        else:
            # Buffered mode: build dicts for legacy _write_csv compatibility
            assert self._prod_ids_arr is not None
            node_col = self._node_ids_arr[node_indices]
            prod_col = self._prod_ids_arr[prod_indices]
            for i in range(n):
                self.inventory_history.append(
                    {
                        "day": day,
                        "node_id": node_col[i],
                        "product_id": prod_col[i],
                        "perceived_inventory": float(perc_vals[i]),
                        "actual_inventory": float(act_vals[i]),
                    }
                )

    def flush(self) -> None:
        """Flush all streaming writers to disk."""
        if self.streaming:
            if self._orders_writer:
                self._orders_writer.flush()
            if self._production_orders_writer:
                self._production_orders_writer.flush()
            if self._shipments_writer:
                self._shipments_writer.flush()
            if self._batches_writer:
                self._batches_writer.flush()
            if self._forecasts_writer:
                self._forecasts_writer.flush()
            if self._returns_writer:
                self._returns_writer.flush()
            if self._batch_ingredients_writer:
                self._batch_ingredients_writer.flush()
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
                f"ProdOrders={row_counts['production_orders']:,}, "
                f"Shipments={row_counts['shipments']:,}, "
                f"Batches={row_counts['batches']:,}, "
                f"BatchIngredients={row_counts['batch_ingredients']:,}, "
                f"Forecasts={row_counts['forecasts']:,}, "
                f"Returns={row_counts['returns']:,}, "
                f"Inventory={row_counts['inventory']:,}"
            )
        else:
            # Buffered mode: write accumulated data
            self._write_csv("orders.csv", self.orders)
            self._write_csv("production_orders.csv", self.production_orders)
            self._write_csv("shipments.csv", self.shipments)
            self._write_csv("batches.csv", self.batches)
            self._write_csv("batch_ingredients.csv", self.batch_ingredients)
            self._write_csv("forecasts.csv", self.forecasts)
            self._write_csv("returns.csv", self.returns)
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
        if self._production_orders_writer:
            self._production_orders_writer.close()
        if self._shipments_writer:
            self._shipments_writer.close()
        if self._batches_writer:
            self._batches_writer.close()
        if self._batch_ingredients_writer:
            self._batch_ingredients_writer.close()
        if self._forecasts_writer:
            self._forecasts_writer.close()
        if self._returns_writer:
            self._returns_writer.close()
        if self._inventory_writer:
            self._inventory_writer.close()

    def _get_row_counts(self) -> dict[str, int]:
        """Get row counts from streaming writers."""
        return {
            "orders": (
                self._orders_writer.row_count if self._orders_writer else 0
            ),
            "production_orders": (
                self._production_orders_writer.row_count
                if self._production_orders_writer else 0
            ),
            "shipments": (
                self._shipments_writer.row_count if self._shipments_writer else 0
            ),
            "batches": (
                self._batches_writer.row_count if self._batches_writer else 0
            ),
            "batch_ingredients": (
                self._batch_ingredients_writer.row_count
                if self._batch_ingredients_writer else 0
            ),
            "forecasts": (
                self._forecasts_writer.row_count if self._forecasts_writer else 0
            ),
            "returns": (
                self._returns_writer.row_count if self._returns_writer else 0
            ),
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
