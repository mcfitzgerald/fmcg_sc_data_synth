"""
MRP Engine: Translates Distribution Requirements (DRP) into Production Orders.

[Task 5.1] [Intent: 4. Architecture - Phase 2: The Time Loop]

This module monitors RDC inventory levels and generates Production Orders
for Plants when stock falls below reorder points.
"""

from typing import Any

import numpy as np

from prism_sim.network.core import NodeType, ProductionOrder, ProductionOrderStatus
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


class MRPEngine:
    """
    Material Requirements Planning engine that generates Production Orders.

    Monitors RDC inventory levels (DRP) and creates Production Orders
    for Plants to maintain target stock levels.
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        # Extract manufacturing config
        mrp_config = config.get("simulation_parameters", {}).get("manufacturing", {})
        self.target_days_supply = mrp_config.get("target_days_supply", 14.0)
        self.reorder_point_days = mrp_config.get("reorder_point_days", 7.0)
        self.min_production_qty = mrp_config.get("min_production_qty", 100.0)
        self.production_lead_time = mrp_config.get("production_lead_time_days", 3)

        # Cache RDC and Plant node indices
        self._rdc_ids: list[str] = []
        self._plant_ids: list[str] = []
        self._finished_product_ids: list[str] = []

        self._cache_node_info()

        # Production Order counter for unique IDs
        self._po_counter = 0

    def _cache_node_info(self) -> None:
        """Cache RDC and Plant node IDs for efficient lookups."""
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.DC:
                self._rdc_ids.append(node_id)
            elif node.type == NodeType.PLANT:
                self._plant_ids.append(node_id)

        # Cache finished product IDs (non-ingredients)
        for product_id, product in self.world.products.items():
            if product.category != ProductCategory.INGREDIENT:
                self._finished_product_ids.append(product_id)

    def generate_production_orders(
        self, current_day: int, daily_demand: np.ndarray
    ) -> list[ProductionOrder]:
        """
        Generate Production Orders based on RDC inventory and demand.

        Uses DRP logic:
        1. Calculate total RDC inventory per product
        2. Calculate average demand per product
        3. Generate Production Orders when inventory < reorder point

        Args:
            current_day: Current simulation day
            daily_demand: Demand tensor [nodes, products] for demand estimation

        Returns:
            List of Production Orders to be processed by TransformEngine
        """
        production_orders: list[ProductionOrder] = []

        # Calculate total inventory across all RDCs per product
        for product_id in self._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue

            # Sum inventory across RDCs
            total_rdc_inventory = 0.0
            for rdc_id in self._rdc_ids:
                n_idx = self.state.node_id_to_idx.get(rdc_id)
                if n_idx is not None:
                    total_rdc_inventory += float(self.state.inventory[n_idx, p_idx])

            # Estimate average daily demand for this product
            avg_daily_demand = self._estimate_demand(p_idx, daily_demand)

            # Calculate days of supply
            if avg_daily_demand > 0:
                days_of_supply = total_rdc_inventory / avg_daily_demand
            else:
                days_of_supply = float("inf")

            # Check if we need to order
            if days_of_supply < self.reorder_point_days:
                # Calculate quantity needed to reach target days supply
                target_inventory = avg_daily_demand * self.target_days_supply
                net_requirement = target_inventory - total_rdc_inventory

                # Round up to minimum batch size
                order_qty = max(net_requirement, self.min_production_qty)

                # Assign to a plant (simple round-robin for now)
                plant_id = self._select_plant(product_id)

                # Create Production Order
                po = ProductionOrder(
                    id=self._generate_po_id(current_day),
                    plant_id=plant_id,
                    product_id=product_id,
                    quantity_cases=order_qty,
                    creation_day=current_day,
                    due_day=current_day + self.production_lead_time,
                    status=ProductionOrderStatus.PLANNED,
                    planned_start_day=current_day + 1,
                )

                production_orders.append(po)

        return production_orders

    def _estimate_demand(self, product_idx: int, daily_demand: np.ndarray) -> float:
        """
        Estimate average daily demand for a product.

        Uses the current day's demand as a proxy (could be enhanced with
        moving average or forecast in future milestones).
        """
        # Sum demand across all nodes for this product
        total_demand = float(np.sum(daily_demand[:, product_idx]))
        return max(total_demand, 1.0)  # Avoid division by zero

    def _select_plant(self, product_id: str) -> str:
        """
        Select a plant for production.

        Simple round-robin assignment. Could be enhanced with:
        - Capacity-based selection
        - Product-specific plant assignments
        - Transportation cost optimization
        """
        if not self._plant_ids:
            raise ValueError("No plants available for production")

        # Round-robin based on order counter
        plant_idx = self._po_counter % len(self._plant_ids)
        return self._plant_ids[plant_idx]

    def _generate_po_id(self, current_day: int) -> str:
        """Generate unique Production Order ID."""
        self._po_counter += 1
        return f"PO-{current_day:03d}-{self._po_counter:06d}"
