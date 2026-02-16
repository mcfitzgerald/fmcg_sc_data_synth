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

import json
from collections import deque
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
    """Converged state loaded from a prior simulation run's parquet output.

    v0.76.0: Extended with agent history buffers to eliminate warm-start transient.
    All agent state fields default to None for backward compatibility with old
    snapshots (without agent_state/ directory).
    """

    checkpoint_day: int
    perceived_inventory: np.ndarray  # [n_nodes, n_products]
    actual_inventory: np.ndarray  # [n_nodes, n_products]
    active_shipments: list[Shipment] = field(default_factory=list)
    active_production_orders: list[ProductionOrder] = field(default_factory=list)
    source_dir: str = ""
    n_nodes_loaded: int = 0
    n_products_loaded: int = 0

    # v0.76.0: Agent history buffers (None = not available, use synthetic priming)
    # MRP history
    mrp_demand_history: np.ndarray | None = None
    mrp_consumption_history: np.ndarray | None = None
    mrp_production_history: np.ndarray | None = None
    mrp_history_ptr: int = 0
    mrp_consumption_ptr: int = 0
    mrp_prod_hist_ptr: int = 0

    # Replenisher history
    rep_demand_history_buffer: np.ndarray | None = None
    rep_smoothed_demand: np.ndarray | None = None
    rep_history_idx: int = 0
    rep_outflow_history: np.ndarray | None = None
    rep_inflow_history: np.ndarray | None = None
    rep_outflow_ptr: int = 0
    rep_inflow_ptr: int = 0

    # Lead time history
    lt_history: dict[tuple[int, int], deque[float]] | None = None
    lt_mu_cache: dict[tuple[int, int], float] | None = None
    lt_sigma_cache: dict[tuple[int, int], float] | None = None

    # StateManager
    inventory_age: np.ndarray | None = None

    # ABC classification state
    rep_product_volume_history: np.ndarray | None = None
    rep_z_scores_vec: np.ndarray | None = None

    # MRP velocity tracking
    mrp_week1_demand_sum: float = 0.0
    mrp_week2_demand_sum: float = 0.0

    # Seasonal phase (for alignment validation)
    seasonal_phase: int = 0


def _load_agent_state(
    source_dir: Path,
    checkpoint_day: int,
    state: StateManager,
) -> dict:
    """Load agent history buffers from agent_state/ subdirectory.

    Returns dict with agent state fields, or empty dict if agent_state/ doesn't exist
    (backward compatibility with old snapshots).

    Validates array shapes against current state dimensions and warns on mismatch.
    """
    agent_dir = source_dir / "agent_state"
    if not agent_dir.exists():
        print("  Agent state: not found (legacy snapshot) — will use synthetic priming")
        return {}

    result: dict = {}

    # Load metadata
    meta_path = agent_dir / "metadata.json"
    if not meta_path.exists():
        print("  Agent state: metadata.json missing — skipping agent state")
        return {}

    with open(meta_path) as f:
        metadata = json.load(f)

    result["seasonal_phase"] = metadata.get("seasonal_phase", 0)
    cycle_days = metadata.get("cycle_days", 365)

    # Warn about seasonal phase misalignment
    if result["seasonal_phase"] != 0:
        print(
            f"  WARNING: Snapshot has seasonal phase offset "
            f"{result['seasonal_phase']}/{cycle_days}."
        )
        print(
            f"  Inventory positions are tuned for day {checkpoint_day} seasonality,"
        )
        print("  but new sim starts at day 0. This may cause a brief transient.")

    # Load MRP history
    mrp_path = agent_dir / "mrp_history.npz"
    if mrp_path.exists():
        try:
            mrp_data = np.load(mrp_path)
            # Validate shapes
            if mrp_data["demand_history"].shape[1] == state.n_products:
                result["mrp_demand_history"] = mrp_data["demand_history"]
                result["mrp_consumption_history"] = mrp_data["consumption_history"]
                result["mrp_production_history"] = mrp_data["production_history"]
                result["mrp_history_ptr"] = int(mrp_data["history_ptr"])
                result["mrp_consumption_ptr"] = int(mrp_data["consumption_ptr"])
                result["mrp_prod_hist_ptr"] = int(mrp_data["prod_hist_ptr"])
                # v0.76.0b: MRP velocity tracking
                if "week1_demand_sum" in mrp_data:
                    result["mrp_week1_demand_sum"] = float(
                        mrp_data["week1_demand_sum"]
                    )
                    result["mrp_week2_demand_sum"] = float(
                        mrp_data["week2_demand_sum"]
                    )
            else:
                n_prod_snapshot = mrp_data["demand_history"].shape[1]
                print(
                    f"  WARNING: MRP history shape mismatch "
                    f"({n_prod_snapshot} vs {state.n_products} products) "
                    f"— skipping MRP history"
                )
        except Exception as e:
            print(f"  WARNING: Failed to load MRP history: {e}")

    # Load replenisher history
    rep_path = agent_dir / "replenisher_history.npz"
    if rep_path.exists():
        try:
            rep_data = np.load(rep_path)
            # Validate shapes
            if (
                rep_data["demand_history_buffer"].shape[1] == state.n_nodes
                and rep_data["demand_history_buffer"].shape[2] == state.n_products
            ):
                result["rep_demand_history_buffer"] = rep_data["demand_history_buffer"]
                result["rep_smoothed_demand"] = rep_data["smoothed_demand"]
                result["rep_history_idx"] = int(rep_data["history_idx"])
                result["rep_outflow_history"] = rep_data["outflow_history"]
                result["rep_inflow_history"] = rep_data["inflow_history"]
                result["rep_outflow_ptr"] = int(rep_data["outflow_ptr"])
                result["rep_inflow_ptr"] = int(rep_data["inflow_ptr"])
                # v0.76.0b: ABC classification state
                if "product_volume_history" in rep_data:
                    pvh = rep_data["product_volume_history"]
                    if pvh.shape == (state.n_products,):
                        result["rep_product_volume_history"] = pvh
                        result["rep_z_scores_vec"] = rep_data["z_scores_vec"]
            else:
                print(
                    "  WARNING: Replenisher history shape mismatch — skipping"
                )
        except Exception as e:
            print(f"  WARNING: Failed to load replenisher history: {e}")

    # Load LT history
    lt_path = agent_dir / "lt_history.npz"
    if lt_path.exists():
        try:
            lt_data = np.load(lt_path)
            keys_arr = lt_data["keys"]
            lengths_arr = lt_data["lengths"]
            values_arr = lt_data["values"]
            mu_arr = lt_data["mu"]
            sigma_arr = lt_data["sigma"]

            # Reconstruct dicts
            lt_history: dict[tuple[int, int], deque[float]] = {}
            lt_mu_cache: dict[tuple[int, int], float] = {}
            lt_sigma_cache: dict[tuple[int, int], float] = {}

            for i in range(len(keys_arr)):
                key = (int(keys_arr[i, 0]), int(keys_arr[i, 1]))
                length = int(lengths_arr[i])
                values = values_arr[i, :length]
                # Filter out NaNs
                values = values[~np.isnan(values)]
                lt_history[key] = deque(values.tolist(), maxlen=len(values_arr[i]))
                lt_mu_cache[key] = float(mu_arr[i])
                lt_sigma_cache[key] = float(sigma_arr[i])

            result["lt_history"] = lt_history
            result["lt_mu_cache"] = lt_mu_cache
            result["lt_sigma_cache"] = lt_sigma_cache
        except Exception as e:
            print(f"  WARNING: Failed to load LT history: {e}")

    # Load inventory age
    age_path = agent_dir / "inventory_age.npy"
    if age_path.exists():
        try:
            age_arr = np.load(age_path)
            if age_arr.shape == (state.n_nodes, state.n_products):
                result["inventory_age"] = age_arr
            else:
                print(
                    f"  WARNING: Inventory age shape mismatch "
                    f"({age_arr.shape} vs ({state.n_nodes}, {state.n_products})) "
                    f"— skipping"
                )
        except Exception as e:
            print(f"  WARNING: Failed to load inventory age: {e}")

    if result:
        loaded_components = []
        if "mrp_demand_history" in result:
            loaded_components.append("MRP")
        if "rep_demand_history_buffer" in result:
            abc = "+ABC" if "rep_product_volume_history" in result else ""
            loaded_components.append(f"Replenisher{abc}")
        if "lt_history" in result:
            loaded_components.append(f"LT history ({len(result['lt_history'])} links)")
        if "inventory_age" in result:
            loaded_components.append("inventory age")
        print(f"  Agent state: loaded {', '.join(loaded_components)}")

    return result


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

    # v0.76.0: Load agent history buffers (if available)
    agent_state = _load_agent_state(src, checkpoint_day, state)

    return WarmStartState(
        checkpoint_day=checkpoint_day,
        perceived_inventory=perceived,
        actual_inventory=actual,
        active_shipments=shipments,
        active_production_orders=production_orders,
        source_dir=source_dir,
        n_nodes_loaded=n_nodes,
        n_products_loaded=n_products,
        # Agent history (v0.76.0)
        **agent_state,
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
