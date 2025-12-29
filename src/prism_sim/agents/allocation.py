from typing import Any

import numpy as np

from prism_sim.network.core import NodeType, Order
from prism_sim.simulation.state import StateManager


class AllocationAgent:
    """
    Handles the allocation of inventory to orders (Level 9).
    Implements 'Fair Share' logic when demand > supply.
    """

    def __init__(
        self, state: StateManager, config: dict[str, Any] | None = None
    ) -> None:
        self.state = state

        # Get epsilon from config or default
        if config:
            sim_params = config.get("simulation_parameters", {})
            self.epsilon = sim_params.get("global_constants", {}).get("epsilon", 0.001)
        else:
            self.epsilon = 0.001

    def _group_orders_by_source(self, orders: list[Order]) -> dict[str, list[Order]]:
        """Group orders by their source node."""
        orders_by_source: dict[str, list[Order]] = {}
        for order in orders:
            if order.source_id not in orders_by_source:
                orders_by_source[order.source_id] = []
            orders_by_source[order.source_id].append(order)
        return orders_by_source

    def _calculate_demand_vector(self, source_orders: list[Order]) -> np.ndarray:
        """Calculate total demand per product for a list of orders."""
        demand_vector = np.zeros(self.state.n_products)
        for order in source_orders:
            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    demand_vector[p_idx] += line.quantity
        return demand_vector

    def _calculate_fill_ratios(
        self, demand_vector: np.ndarray, current_inv: np.ndarray
    ) -> np.ndarray:
        """Calculate fill ratios based on demand vs inventory."""
        with np.errstate(divide="ignore", invalid="ignore"):
            fill_ratios = np.where(
                demand_vector > current_inv, current_inv / demand_vector, 1.0
            )
            return np.nan_to_num(fill_ratios, nan=1.0)

    def _apply_ratios_to_orders(
        self, source_orders: list[Order], fill_ratios: np.ndarray
    ) -> list[Order]:
        """Apply fill ratios to orders and return non-empty orders."""
        allocated = []
        for order in source_orders:
            new_lines = []
            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    ratio = fill_ratios[p_idx]
                    new_qty = line.quantity * ratio
                    if new_qty > self.epsilon:
                        line.quantity = new_qty
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            order.lines = new_lines
            if new_lines:
                allocated.append(order)
        return allocated

    def allocate_orders(self, orders: list[Order]) -> list[Order]:
        """
        Processes orders against available inventory at the source.
        
        Implements 'Fill or Kill' logic for FMCG realism:
        - Orders are processed immediately against available stock.
        - Any quantity not filled is 'Killed' (not backlogged).
        - Orders are marked 'CLOSED' after processing.
        - The Replenisher will re-order naturally if inventory remains low.

        Returns:
            List of processed orders with updated quantities (allocated amount).
        """
        orders_by_source = self._group_orders_by_source(orders)
        allocated_orders: list[Order] = []

        for source_id, source_orders in orders_by_source.items():
            source_node = self.state.world.nodes.get(source_id)
            source_idx = self.state.node_id_to_idx.get(source_id)

            if (
                source_node is None
                or source_idx is None
            ):
                # Unknown node - skip
                continue

            if source_node.type == NodeType.SUPPLIER:
                # Supplier Logic: Capacity Constraints
                total_demand = sum(line.quantity for o in source_orders for line in o.lines)
                capacity = getattr(source_node, "throughput_capacity", float("inf"))
                
                ratio = 1.0
                if total_demand > capacity and capacity > 0:
                    ratio = capacity / total_demand
                
                # Apply ratio
                for order in source_orders:
                    if ratio < 1.0:
                        for line in order.lines:
                            line.quantity *= ratio
                    order.status = "CLOSED"
                    allocated_orders.append(order)
                continue

            # RDC/Plant Logic: Inventory Constraints
            # Calculate demand and fill ratios
            # Use ACTUAL inventory to prevent phantom inventory causing negatives
            # When shrinkage occurs, actual < perceived. We must allocate based on
            # what's physically available to prevent negative inventory.
            demand_vector = self._calculate_demand_vector(source_orders)
            actual_inv = self.state.actual_inventory[source_idx, :]
            current_inv = np.maximum(0, actual_inv)  # Guard against any existing negatives
            fill_ratios = self._calculate_fill_ratios(demand_vector, current_inv)

            # Decrement inventory (Sync actual and perceived)
            for p_idx, ratio in enumerate(fill_ratios):
                if ratio > 0:
                    allocated_qty = demand_vector[p_idx] * ratio
                    p_id = self.state.product_idx_to_id[p_idx]
                    self.state.update_inventory(source_id, p_id, -allocated_qty)

            # Apply ratios to orders and CLOSE them
            for order in source_orders:
                new_lines = []
                for line in order.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        ratio = fill_ratios[p_idx]
                        new_qty = line.quantity * ratio
                        if new_qty > self.epsilon:
                            line.quantity = new_qty
                            new_lines.append(line)
                    else:
                        # Product not found, keep as is (or kill?) - keeping for safety
                        new_lines.append(line)
                
                order.lines = new_lines
                # Mark as CLOSED (Fill or Kill)
                order.status = "CLOSED"
                
                # Only return if there's something to ship
                if new_lines:
                    allocated_orders.append(order)

        return allocated_orders
