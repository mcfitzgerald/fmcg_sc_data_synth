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
    unmet_demand_matrix: np.ndarray  # [n_nodes, n_products] - unfilled qty


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

        PERF v0.69.3: Schwartzian transform — compute sort key once per order
        instead of O(n log n) times via closure-based key function.
        """
        priority_map = {
            OrderType.RUSH: 1,
            OrderType.PROMOTIONAL: 2,
            OrderType.STANDARD: 3,
            OrderType.BACKORDER: 4,
        }

        velocity = self.product_velocity
        keys: list[tuple[int, float, int]] = []
        for order in orders:
            v = 0.0
            if velocity is not None:
                for line in order.lines:
                    p_idx = (
                        line.product_idx if line.product_idx >= 0
                        else self.state.product_id_to_idx.get(line.product_id)
                    )
                    if p_idx is not None and p_idx >= 0:
                        v += velocity[p_idx]
            keys.append((priority_map.get(order.order_type, 5), -v, order.priority))

        paired = sorted(zip(keys, orders, strict=False), key=lambda x: x[0])
        return [o for _, o in paired]

    def _calculate_demand_vector(self, source_orders: list[Order]) -> np.ndarray:
        """Calculate total demand per product for a list of orders."""
        demand_vector = np.zeros(self.state.n_products)
        for order in source_orders:
            for line in order.lines:
                p_idx = (
                    line.product_idx if line.product_idx >= 0
                    else self.state.product_id_to_idx.get(line.product_id)
                )
                if p_idx is not None and p_idx >= 0:
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

        v0.38.0: Now also tracks unmet demand (unfilled qty) to improve
        replenishment signal. When orders can't be fully filled, the shortfall
        is recorded so replenishment can account for true demand.

        Returns:
            AllocationResult with allocated_orders, allocation_matrix tracking
            inventory decrements, and unmet_demand_matrix for signal improvement.
        """
        orders_by_source = self._group_orders_by_source(orders)
        allocated_orders: list[Order] = []

        # Track all allocations for mass balance audit
        allocation_matrix = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        # v0.38.0: Track unmet demand (what couldn't be filled)
        unmet_demand_matrix = np.zeros(
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

            # PERF: Vectorized inventory decrement (replaces 500K+ per-product calls)
            # Calculate allocated quantities for all products at once
            allocated_qty_vec = demand_vector * fill_ratios

            # FIFO age reduction: oldest units ship first, reducing remaining age
            old_qty = np.maximum(0.0, self.state.actual_inventory[source_idx, :])
            with np.errstate(divide='ignore', invalid='ignore'):
                fraction_remaining = np.where(
                    old_qty > 0,
                    np.clip(
                        (old_qty - allocated_qty_vec) / old_qty, 0.0, 1.0
                    ),
                    0.0,
                )
            self.state.inventory_age[source_idx, :] *= fraction_remaining

            # Update inventory tensors directly (batch operation)
            self.state.actual_inventory[source_idx, :] -= allocated_qty_vec
            self.state.perceived_inventory[source_idx, :] -= allocated_qty_vec

            # Floor to zero - prevent floating point noise from creating negatives
            np.maximum(
                self.state.actual_inventory[source_idx, :],
                0,
                out=self.state.actual_inventory[source_idx, :],
            )
            np.maximum(
                self.state.perceived_inventory[source_idx, :],
                0,
                out=self.state.perceived_inventory[source_idx, :],
            )

            # Track for mass balance audit
            allocation_matrix[source_idx, :] += allocated_qty_vec

            # v0.38.0: Track unmet demand (what couldn't be filled) - VECTORIZED
            unfilled_qty_vec = demand_vector * (1.0 - fill_ratios)
            # Only track where ratio < 1 and unfilled_qty > epsilon
            unmet_mask = (fill_ratios < 1.0) & (unfilled_qty_vec > self.epsilon)
            unmet_demand_matrix[source_idx, unmet_mask] += unfilled_qty_vec[unmet_mask]

            # Apply to orders
            for order in sorted_orders:
                new_lines = []
                for line in order.lines:
                    prod_idx = (
                        line.product_idx if line.product_idx >= 0
                        else self.state.product_id_to_idx.get(line.product_id)
                    )
                    if prod_idx is not None and prod_idx >= 0:
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

        # v0.38.0: Record unmet demand in state manager for replenishment signal
        if np.any(unmet_demand_matrix > 0):
            self.state.record_unmet_demand_batch(unmet_demand_matrix)

        return AllocationResult(
            allocated_orders=allocated_orders,
            allocation_matrix=allocation_matrix,
            unmet_demand_matrix=unmet_demand_matrix,
        )
