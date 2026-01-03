from dataclasses import dataclass
from typing import Any

import numpy as np

from prism_sim.network.core import NodeType, Order, OrderType
from prism_sim.simulation.state import StateManager


@dataclass
class AllocationResult:
    """Result of allocation with tracking for mass balance audit."""

    allocated_orders: list[Order]
    allocation_matrix: np.ndarray  # Shape [n_nodes, n_products] - qty decremented


class AllocationAgent:
    """
    Handles the allocation of inventory to orders (Level 9).
    Implements 'Fair Share' logic when demand > supply.
    """

    def __init__(self, state: StateManager, config: dict[str, Any]) -> None:
        self.state = state
        self.config = config

        sim_params = config.get("simulation_parameters", {})
        constants = sim_params.get("global_constants", {})
        self.epsilon = float(constants.get("epsilon", 0.001))

        # Product velocity vector for ABC prioritization (A-items first)
        self.product_velocity: np.ndarray | None = None

    def set_product_velocity(self, velocity: np.ndarray) -> None:
        """Set product velocity vector for ABC prioritization."""
        self.product_velocity = velocity

    def _group_orders_by_source(self, orders: list[Order]) -> dict[str, list[Order]]:
        """Group orders by their source node."""
        orders_by_source: dict[str, list[Order]] = {}
        for order in orders:
            if order.source_id not in orders_by_source:
                orders_by_source[order.source_id] = []
            orders_by_source[order.source_id].append(order)
        return orders_by_source

    def _prioritize_orders(self, orders: list[Order]) -> list[Order]:
        """
        Sort orders by priority.
        1. Order Type (RUSH > PROMO > STANDARD > BACKORDER)
        2. Product Velocity (ABC Prioritization: High velocity > Low velocity)
        3. Order Priority Int (Tie-breaker)
        """
        priority_map = {
            OrderType.RUSH: 1,
            OrderType.PROMOTIONAL: 2,
            OrderType.STANDARD: 3,
            OrderType.BACKORDER: 4,
        }

        def order_sort_key(order: Order) -> tuple[int, float, int]:
            # Primary sort: Order Type
            type_priority = priority_map.get(order.order_type, 5)

            # Secondary sort: Product Velocity (A-items first)
            # Calculate total velocity of lines in order
            order_velocity = 0.0
            if self.product_velocity is not None:
                for line in order.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        order_velocity += self.product_velocity[p_idx]

            # Return tuple: (low=high_priority, high=high_velocity, low=high_priority)
            # We want high velocity to be first, so negate it
            return (type_priority, -order_velocity, order.priority)

        return sorted(orders, key=order_sort_key)

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

    def allocate_orders(self, orders: list[Order]) -> AllocationResult:
        """
        Processes orders against available inventory at the source.

        Implements 'Fill or Kill' logic for FMCG realism.
        Respects Order Type priority.

        Returns:
            AllocationResult with allocated_orders and allocation_matrix tracking
            inventory decrements for mass balance auditing.
        """
        orders_by_source = self._group_orders_by_source(orders)
        allocated_orders: list[Order] = []

        # Track all allocations for mass balance audit
        allocation_matrix = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        for source_id, source_orders in orders_by_source.items():
            source_node = self.state.world.nodes.get(source_id)
            source_idx = self.state.node_id_to_idx.get(source_id)

            if source_node is None or source_idx is None:
                continue

            # Sort by priority before allocation
            sorted_orders = self._prioritize_orders(source_orders)

            if source_node.type == NodeType.SUPPLIER:
                # Supplier Logic - infinite source, no inventory decrement
                total_demand = sum(
                    line.quantity for o in source_orders for line in o.lines
                )
                capacity = getattr(source_node, "throughput_capacity", float("inf"))

                ratio = 1.0
                if total_demand > capacity > 0:
                    ratio = capacity / total_demand

                for order in source_orders:
                    if ratio < 1.0:
                        for line in order.lines:
                            line.quantity *= ratio
                    order.status = "CLOSED"
                    allocated_orders.append(order)
                continue

            # RDC/Plant Logic
            demand_vector = self._calculate_demand_vector(source_orders)
            actual_inv = self.state.actual_inventory[source_idx, :]
            current_inv = np.maximum(0, actual_inv)
            fill_ratios = self._calculate_fill_ratios(demand_vector, current_inv)

            # Decrement inventory (constrained by fill_ratios)
            for p_idx, ratio in enumerate(fill_ratios):
                if ratio > 0:
                    allocated_qty = demand_vector[p_idx] * ratio
                    p_id = self.state.product_idx_to_id[p_idx]
                    self.state.update_inventory(source_id, p_id, -allocated_qty)
                    # Track for mass balance audit
                    allocation_matrix[source_idx, p_idx] += allocated_qty

            # Apply to orders
            for order in sorted_orders:
                new_lines = []
                for line in order.lines:
                    prod_idx = self.state.product_id_to_idx.get(line.product_id)
                    if prod_idx is not None:
                        ratio = fill_ratios[prod_idx]
                        new_qty = line.quantity * ratio
                        if new_qty > self.epsilon:
                            line.quantity = new_qty
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                order.lines = new_lines
                order.status = "CLOSED"
                if new_lines:
                    allocated_orders.append(order)

        return AllocationResult(
            allocated_orders=allocated_orders, allocation_matrix=allocation_matrix
        )
