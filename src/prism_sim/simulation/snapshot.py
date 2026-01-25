"""
Snapshot utilities for warm-start checkpointing system.

This module provides shared functions for creating and managing simulation
snapshots, used by both:
- scripts/generate_warm_start.py (manual snapshot generation)
- Orchestrator (automatic checkpointing)

v0.33.0: Initial implementation for automatic steady-state checkpointing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from prism_sim.simulation.orchestrator import Orchestrator

# Serialization threshold for filtering near-zero values (file size optimization)
SERIALIZATION_THRESHOLD = 0.01


def compute_config_hash(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    """
    Compute a hash of the configuration to detect stale snapshots.

    Includes both simulation_config.json and world_definition.json content
    to ensure snapshot validity when either changes.

    Args:
        config: Simulation configuration dict
        manifest: World definition/manifest dict

    Returns:
        16-character hex hash string
    """
    # Serialize deterministically (sorted keys)
    config_str = json.dumps(config, sort_keys=True)
    manifest_str = json.dumps(manifest, sort_keys=True)
    combined = config_str + manifest_str
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def serialize_shipment(shipment: Any) -> dict[str, Any]:
    """Serialize a Shipment object to JSON-compatible dict."""
    status_val = (
        shipment.status.value
        if hasattr(shipment.status, "value")
        else str(shipment.status)
    )
    weight = float(shipment.total_weight_kg) if shipment.total_weight_kg else 0.0
    volume = float(shipment.total_volume_m3) if shipment.total_volume_m3 else 0.0
    return {
        "id": shipment.id,
        "source_id": shipment.source_id,
        "target_id": shipment.target_id,
        "creation_day": shipment.creation_day,
        "arrival_day": shipment.arrival_day,
        "original_order_day": shipment.original_order_day,
        "status": status_val,
        "lines": [
            {"product_id": line.product_id, "quantity": float(line.quantity)}
            for line in shipment.lines
        ],
        "total_weight_kg": weight,
        "total_volume_m3": volume,
    }


def serialize_production_order(order: Any) -> dict[str, Any]:
    """Serialize a ProductionOrder object to JSON-compatible dict."""
    status_val = (
        order.status.value if hasattr(order.status, "value") else str(order.status)
    )
    return {
        "id": order.id,
        "plant_id": order.plant_id,
        "product_id": order.product_id,
        "quantity_cases": float(order.quantity_cases),
        "creation_day": order.creation_day,
        "due_day": order.due_day,
        "status": status_val,
        "produced_quantity": float(order.produced_quantity),
    }


def capture_minimal_state(sim: Orchestrator) -> dict[str, Any]:
    """
    Capture the minimal state needed for warm-start.

    This includes only state that cannot be derived from config:
    - Inventory tensors (actual + perceived)
    - Active shipments in transit
    - Active production orders

    Args:
        sim: Orchestrator instance at steady state

    Returns:
        Dict with inventory, shipments, and production orders
    """
    state = sim.state

    # Inventory tensors - convert to nested dict for JSON serialization
    # Format: {node_id: {product_id: quantity}}
    actual_inv: dict[str, dict[str, float]] = {}
    perceived_inv: dict[str, dict[str, float]] = {}

    for node_idx in range(state.n_nodes):
        node_id = state.node_idx_to_id[node_idx]
        actual_inv[node_id] = {}
        perceived_inv[node_id] = {}

        for prod_idx in range(state.n_products):
            prod_id = state.product_idx_to_id[prod_idx]
            actual_qty = float(state.actual_inventory[node_idx, prod_idx])
            perceived_qty = float(state.perceived_inventory[node_idx, prod_idx])

            # Only store non-zero values to reduce file size
            if actual_qty > SERIALIZATION_THRESHOLD:
                actual_inv[node_id][prod_id] = round(actual_qty, 2)
            if perceived_qty > SERIALIZATION_THRESHOLD:
                perceived_inv[node_id][prod_id] = round(perceived_qty, 2)

    # Active shipments
    shipments = [serialize_shipment(s) for s in state.active_shipments]

    # Active production orders
    production_orders = [
        serialize_production_order(po) for po in sim.active_production_orders
    ]

    return {
        "actual_inventory": actual_inv,
        "perceived_inventory": perceived_inv,
        "active_shipments": shipments,
        "active_production_orders": production_orders,
    }


def derive_initialization_params(
    sim: Orchestrator, burn_in_days: int
) -> dict[str, Any]:
    """
    Derive steady-state parameters for history buffer initialization.

    These values are used to pre-populate history buffers on warm-start,
    avoiding the need to capture full history tensors.

    Args:
        sim: Orchestrator instance at steady state
        burn_in_days: Number of burn-in days completed

    Returns:
        Dict with derived parameters for buffer initialization
    """
    state = sim.state
    replenisher = sim.replenisher
    mrp = sim.mrp_engine

    # 1. Average demand per product (from replenisher's demand history)
    # Use the demand_history_buffer which has 28 days of history
    if replenisher.demand_history_buffer is not None:
        # Sum across nodes, average across days: [days, products]
        daily_totals = np.sum(replenisher.demand_history_buffer, axis=1)
        avg_demand = np.mean(daily_totals, axis=0)
        demand_sigma = np.std(daily_totals, axis=0)
    else:
        avg_demand = np.zeros(state.n_products)
        demand_sigma = np.zeros(state.n_products)

    avg_demand_dict = {
        state.product_idx_to_id[i]: round(float(avg_demand[i]), 4)
        for i in range(state.n_products)
        if avg_demand[i] > SERIALIZATION_THRESHOLD
    }
    demand_sigma_dict = {
        state.product_idx_to_id[i]: round(float(demand_sigma[i]), 4)
        for i in range(state.n_products)
        if demand_sigma[i] > SERIALIZATION_THRESHOLD
    }

    # 2. Average lead time per link (from replenisher's sparse lead time storage)
    # PERF: Iterate only over links with actual observations (sparse)
    avg_lead_times: dict[str, dict[str, float]] = {}
    lead_time_sigma: dict[str, dict[str, float]] = {}

    for link_key, mu in replenisher._lt_mu_cache_sparse.items():
        target_idx, source_idx = link_key
        target_id = state.node_idx_to_id[target_idx]
        source_id = state.node_idx_to_id[source_idx]
        sigma = replenisher._lt_sigma_cache_sparse.get(link_key, 0.0)

        if target_id not in avg_lead_times:
            avg_lead_times[target_id] = {}
            lead_time_sigma[target_id] = {}

        avg_lead_times[target_id][source_id] = round(mu, 2)
        lead_time_sigma[target_id][source_id] = round(sigma, 2)

    # 3. ABC classifications (from MRP engine)
    abc_classes = {
        state.product_idx_to_id[i]: int(mrp.abc_class[i])
        for i in range(state.n_products)
    }

    # 4. Product volume history (for ABC recalculation)
    volume_history = {
        state.product_idx_to_id[i]: round(
            float(replenisher.product_volume_history[i]), 2
        )
        for i in range(state.n_products)
        if replenisher.product_volume_history[i] > 0
    }

    # 5. Z-scores per product
    z_scores = {
        state.product_idx_to_id[i]: round(float(replenisher.z_scores_vec[i]), 2)
        for i in range(state.n_products)
    }

    return {
        "avg_demand_by_product": avg_demand_dict,
        "demand_sigma_by_product": demand_sigma_dict,
        "avg_lead_time_by_link": avg_lead_times,
        "lead_time_sigma_by_link": lead_time_sigma,
        "abc_classifications": abc_classes,
        "product_volume_history": volume_history,
        "z_scores_by_product": z_scores,
        "burn_in_days": burn_in_days,
    }


def save_snapshot(
    sim: Orchestrator,
    burn_in_days: int,
    config_hash: str,
    output_path: Path,
    auto_generated: bool = False,
) -> None:
    """
    Save a simulation snapshot to disk.

    Args:
        sim: Orchestrator instance at steady state
        burn_in_days: Number of burn-in days completed
        config_hash: Config hash for validation
        output_path: Path to save the snapshot
        auto_generated: Whether this is an auto-checkpoint
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build snapshot
    snapshot = {
        "metadata": {
            "version": "0.33.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "burn_in_days": burn_in_days,
            "config_hash": config_hash,
            "auto_generated": auto_generated,
            "n_nodes": sim.state.n_nodes,
            "n_products": sim.state.n_products,
        },
        "state": capture_minimal_state(sim),
        "derived": derive_initialization_params(sim, burn_in_days),
    }

    # Update metadata with counts
    snapshot["metadata"]["n_shipments"] = len(snapshot["state"]["active_shipments"])
    snapshot["metadata"]["n_production_orders"] = len(
        snapshot["state"]["active_production_orders"]
    )

    # Write compressed snapshot
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(snapshot, f, separators=(",", ":"))

    # Report size
    file_size = output_path.stat().st_size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{file_size / 1024:.1f} KB"

    print(f"Checkpoint saved: {output_path.name} ({size_str})")
