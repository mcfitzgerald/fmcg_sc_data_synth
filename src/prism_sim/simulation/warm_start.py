"""
Warm-start state loader for prism-sim.

Reads converged inventory, in-transit shipments, and active production orders
from a prior simulation run's parquet output. This eliminates the synthetic
priming transient (first-30-day production overshoot) by starting from
a known steady state.

Usage:
    ws = load_warm_start_state("data/output", state, world)
    # Then apply ws.perceived_inventory, ws.active_shipments, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from prism_sim.network.core import (
    OrderLine,
    ProductionOrder,
    ProductionOrderStatus,
    Shipment,
    ShipmentStatus,
)
from prism_sim.simulation.state import StateManager

if TYPE_CHECKING:
    from prism_sim.simulation.world import World


@dataclass
class WarmStartState:
    """Converged state loaded from a prior simulation run's parquet output."""

    checkpoint_day: int
    perceived_inventory: np.ndarray  # [n_nodes, n_products]
    actual_inventory: np.ndarray  # [n_nodes, n_products]
    active_shipments: list[Shipment] = field(default_factory=list)
    active_production_orders: list[ProductionOrder] = field(default_factory=list)
    source_dir: str = ""
    n_nodes_loaded: int = 0
    n_products_loaded: int = 0


def load_warm_start_state(
    source_dir: str,
    state: StateManager,
    world: World,
) -> WarmStartState:
    """Load converged state from a prior run's parquet output.

    Args:
        source_dir: Path to directory containing parquet files.
        state: StateManager (for index mappings).
        world: World (for node/product validation).

    Returns:
        WarmStartState with all restored data.

    Raises:
        FileNotFoundError: If required parquet files are missing.
        ValueError: If inventory parquet has no data rows.
    """
    src = Path(source_dir)

    required = ["inventory.parquet", "shipments.parquet", "production_orders.parquet"]
    missing = [f for f in required if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Warm-start requires {', '.join(missing)} in {source_dir}. "
            f"Run source sim with: --streaming --format parquet"
        )

    print(f"Loading warm-start state from {source_dir}...")

    perceived, actual, checkpoint_day = _load_inventory(src, state)
    shipments = _load_active_shipments(src, checkpoint_day, state, world)
    production_orders = _load_active_production_orders(
        src, checkpoint_day, state, world
    )

    n_nodes = int(np.count_nonzero(np.any(actual > 0, axis=1)))
    n_products = int(np.count_nonzero(np.any(actual > 0, axis=0)))

    return WarmStartState(
        checkpoint_day=checkpoint_day,
        perceived_inventory=perceived,
        actual_inventory=actual,
        active_shipments=shipments,
        active_production_orders=production_orders,
        source_dir=source_dir,
        n_nodes_loaded=n_nodes,
        n_products_loaded=n_products,
    )


def _load_inventory(
    source_dir: Path,
    state: StateManager,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load inventory tensors from the latest snapshot in inventory.parquet.

    Streams row groups to find max(day), then loads that day's snapshot
    into dense [n_nodes, n_products] tensors.

    Returns:
        (perceived_inventory, actual_inventory, checkpoint_day)
    """
    inv_path = source_dir / "inventory.parquet"
    pf = pq.ParquetFile(inv_path)

    if pf.metadata.num_row_groups == 0:
        raise ValueError(
            f"inventory.parquet in {source_dir} is empty. "
            f"Source simulation may have used --no-logging."
        )

    # First pass: find max day (scan day column only)
    max_day = 0
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=["day"])
        rg_max = pc.max(table.column("day")).as_py()
        if rg_max is not None and rg_max > max_day:
            max_day = rg_max

    if max_day == 0:
        raise ValueError(
            f"inventory.parquet in {source_dir} has no valid day entries."
        )

    # Second pass: read rows matching max_day into dense tensors
    perceived = np.zeros((state.n_nodes, state.n_products), dtype=np.float32)
    actual = np.zeros((state.n_nodes, state.n_products), dtype=np.float32)
    loaded = 0
    skipped = 0

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        mask = pc.equal(table.column("day"), max_day)
        filtered = table.filter(mask)

        if filtered.num_rows == 0:
            continue

        # Dictionary-encoded columns need .to_pylist() for correct decoding
        node_ids = filtered.column("node_id").to_pylist()
        product_ids = filtered.column("product_id").to_pylist()
        perc_vals = filtered.column("perceived_inventory").to_numpy()
        act_vals = filtered.column("actual_inventory").to_numpy()

        for j in range(filtered.num_rows):
            n_idx = state.node_id_to_idx.get(node_ids[j])
            p_idx = state.product_id_to_idx.get(product_ids[j])
            if n_idx is not None and p_idx is not None:
                perceived[n_idx, p_idx] = perc_vals[j]
                actual[n_idx, p_idx] = act_vals[j]
                loaded += 1
            else:
                skipped += 1

    print(f"  Inventory: {loaded:,} entries from day {max_day}"
          + (f" ({skipped:,} skipped — world mismatch)" if skipped else ""))
    return perceived, actual, max_day


def _load_active_shipments(
    source_dir: Path,
    checkpoint_day: int,
    state: StateManager,
    world: World,
) -> list[Shipment]:
    """Load shipments still in transit at the checkpoint day.

    Shipments are logged once at creation (status=in_transit, never re-logged
    on delivery). We identify still-active shipments by arrival_day > checkpoint_day.

    Lines sharing the same shipment_id are grouped into a single Shipment object.
    Days are remapped: arrival_day -= checkpoint_day, creation_day = 0.
    """
    ship_path = source_dir / "shipments.parquet"
    pf = pq.ParquetFile(ship_path)

    # Collect shipment lines grouped by shipment_id
    shipment_data: dict[str, dict] = {}  # sid -> {meta, lines}

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        mask = pc.greater(table.column("arrival_day"), checkpoint_day)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue

        sids = filtered.column("shipment_id").to_pylist()
        creation_days = filtered.column("creation_day").to_pylist()
        arrival_days = filtered.column("arrival_day").to_pylist()
        source_ids = filtered.column("source_id").to_pylist()
        target_ids = filtered.column("target_id").to_pylist()
        product_ids = filtered.column("product_id").to_pylist()
        quantities = filtered.column("quantity").to_pylist()

        for j in range(filtered.num_rows):
            sid = sids[j]
            if sid not in shipment_data:
                shipment_data[sid] = {
                    "source_id": source_ids[j],
                    "target_id": target_ids[j],
                    "creation_day": creation_days[j],
                    "arrival_day": arrival_days[j],
                    "lines": [],
                }
            shipment_data[sid]["lines"].append(
                OrderLine(product_ids[j], quantities[j],
                          product_idx=state.product_id_to_idx.get(product_ids[j], -1))
            )

    # Build Shipment objects with remapped days
    shipments: list[Shipment] = []
    skipped = 0
    for sid, data in shipment_data.items():
        if data["source_id"] not in world.nodes:
            skipped += 1
            continue
        if data["target_id"] not in world.nodes:
            skipped += 1
            continue

        shipments.append(
            Shipment(
                id=f"WS-{sid}",
                source_id=data["source_id"],
                target_id=data["target_id"],
                creation_day=0,
                arrival_day=data["arrival_day"] - checkpoint_day,
                lines=data["lines"],
                status=ShipmentStatus.IN_TRANSIT,
                source_idx=state.node_id_to_idx.get(data["source_id"], -1),
                target_idx=state.node_id_to_idx.get(data["target_id"], -1),
            )
        )

    print(f"  Shipments: {len(shipments)} in-transit"
          + (f" ({skipped} skipped — world mismatch)" if skipped else ""))
    return shipments


def _load_active_production_orders(
    source_dir: Path,
    checkpoint_day: int,
    state: StateManager,
    world: World,
) -> list[ProductionOrder]:
    """Load production orders still active at the checkpoint day.

    Filters for status in (in_progress, planned, released).
    Estimates produced_quantity from elapsed fraction (not in parquet).
    Days are remapped relative to new sim start.
    """
    po_path = source_dir / "production_orders.parquet"
    pf = pq.ParquetFile(po_path)

    active_statuses = {"in_progress", "planned", "released"}
    status_map = {
        "in_progress": ProductionOrderStatus.IN_PROGRESS,
        "planned": ProductionOrderStatus.PLANNED,
        "released": ProductionOrderStatus.RELEASED,
    }

    orders: list[ProductionOrder] = []
    skipped = 0

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)

        po_ids = table.column("po_id").to_pylist()
        plant_ids = table.column("plant_id").to_pylist()
        product_ids = table.column("product_id").to_pylist()
        quantities = table.column("quantity").to_pylist()
        creation_days = table.column("creation_day").to_pylist()
        due_days = table.column("due_day").to_pylist()
        statuses = table.column("status").to_pylist()

        for j in range(table.num_rows):
            if statuses[j] not in active_statuses:
                continue
            if plant_ids[j] not in world.nodes:
                skipped += 1
                continue
            if product_ids[j] not in state.product_id_to_idx:
                skipped += 1
                continue

            qty = quantities[j]

            # Estimate produced_quantity from elapsed fraction
            elapsed = checkpoint_day - creation_days[j]
            total_duration = max(1, due_days[j] - creation_days[j])
            fraction = min(max(elapsed / total_duration, 0.0), 0.9)
            produced = qty * fraction if statuses[j] == "in_progress" else 0.0

            # Remap days relative to new sim start
            new_creation = creation_days[j] - checkpoint_day
            new_due = max(1, due_days[j] - checkpoint_day)

            orders.append(
                ProductionOrder(
                    id=f"WS-{po_ids[j]}",
                    plant_id=plant_ids[j],
                    product_id=product_ids[j],
                    quantity_cases=qty,
                    creation_day=new_creation,
                    due_day=new_due,
                    status=status_map[statuses[j]],
                    produced_quantity=produced,
                )
            )

    print(f"  Production orders: {len(orders)} active"
          + (f" ({skipped} skipped — world mismatch)" if skipped else ""))
    return orders
