import csv
import json
import os
from typing import Any

from prism_sim.network.core import Batch, Order, Shipment


class SimulationWriter:
    """
    Handles data export for the Prism Digital Twin.
    Generates SCOR-DS compatible datasets.
    """

    def __init__(self, output_dir: str = "data/output", enable_logging: bool = False):
        self.output_dir = output_dir
        self.enable_logging = enable_logging

        if self.enable_logging:
            os.makedirs(output_dir, exist_ok=True)

        # Buffers for data
        self.orders: list[dict[str, Any]] = []
        self.shipments: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.inventory_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []

    def log_orders(self, orders: list[Order], day: int) -> None:
        if not self.enable_logging:
            return

        for order in orders:
            for line in order.lines:
                self.orders.append(
                    {
                        "order_id": order.id,
                        "day": day,
                        "source_id": order.source_id,
                        "target_id": order.target_id,
                        "product_id": line.product_id,
                        "quantity": line.quantity,
                        "status": order.status,
                    }
                )

    def log_shipments(self, shipments: list[Shipment], day: int) -> None:
        if not self.enable_logging:
            return

        for s in shipments:
            for line in s.lines:
                self.shipments.append(
                    {
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
                )

    def log_batches(self, batches: list[Batch], day: int) -> None:
        if not self.enable_logging:
            return

        for b in batches:
            self.batches.append(
                {
                    "batch_id": b.id,
                    "plant_id": b.plant_id,
                    "product_id": b.product_id,
                    "day_produced": day,
                    "quantity": b.quantity_cases,
                    "yield_pct": b.yield_percent,
                    "status": b.status.value,
                    "notes": b.notes,
                }
            )

    def log_inventory(self, state: Any, world: Any, day: int) -> None:
        if not self.enable_logging:
            return

        # Vectorized state to flat format
        # This can be large if we do it every day for every node/product
        # We might want to sample or only do RDCs
        for node_id, node_idx in state.node_id_to_idx.items():
            for prod_id, prod_idx in state.product_id_to_idx.items():
                perceived = state.perceived_inventory[node_idx, prod_idx]
                actual = state.actual_inventory[node_idx, prod_idx]
                if perceived != 0 or actual != 0:
                    self.inventory_history.append(
                        {
                            "day": day,
                            "node_id": node_id,
                            "product_id": prod_id,
                            "perceived_inventory": perceived,
                            "actual_inventory": actual,
                        }
                    )

    def save(self, final_metrics: dict[str, Any]) -> None:
        """Write all buffers to disk."""
        if not self.enable_logging:
            print("SimulationWriter: Logging disabled, skipping CSV export.")
            return

        self._write_csv("orders.csv", self.orders)
        self._write_csv("shipments.csv", self.shipments)
        self._write_csv("batches.csv", self.batches)
        self._write_csv("inventory.csv", self.inventory_history)

        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)

        print(f"Simulation data exported to {self.output_dir}")

    def _write_csv(self, filename: str, data: list[dict[str, Any]]) -> None:
        if not data:
            return

        filepath = os.path.join(self.output_dir, filename)
        keys = data[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
