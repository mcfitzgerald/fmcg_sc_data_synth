from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prism_sim.network.core import (
    NodeType,
    Order,
    OrderBatch,
    OrderLine,
    OrderPriority,
    OrderType,
)
from prism_sim.simulation.state import StateManager

# OrderType → int mapping shared between batch and _materialize
_ORDER_TYPE_INT_MAP: dict[int, OrderType] = {
    1: OrderType.RUSH,
    2: OrderType.PROMOTIONAL,
    3: OrderType.STANDARD,
    4: OrderType.BACKORDER,
}

_PRIORITY_INT_MAP: dict[int, OrderPriority] = {
    int(OrderPriority.RUSH): OrderPriority.RUSH,
    int(OrderPriority.HIGH): OrderPriority.HIGH,
    int(OrderPriority.STANDARD): OrderPriority.STANDARD,
    int(OrderPriority.LOW): OrderPriority.LOW,
}


@dataclass
class AllocationResult:
    """Result of allocation with tracking for mass balance audit."""

    allocated_orders: list[Order]
    allocation_matrix: np.ndarray  # Shape [n_nodes, n_products] - qty decremented
    unmet_demand_matrix: np.ndarray  # [n_nodes, n_products] - unfilled qty
    # v0.86.0: Pre-allocation batch for signal consumers (MRP, inflow, etc.)
    order_batch: OrderBatch | None = field(default=None, repr=False)


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

        # Pre-compute supplier boolean mask (indexed by node_idx)
        self._is_supplier = np.zeros(state.n_nodes, dtype=bool)
        for n_id, node in state.world.nodes.items():
            if node.type == NodeType.SUPPLIER:
                idx = state.node_id_to_idx.get(n_id)
                if idx is not None:
                    self._is_supplier[idx] = True

    def set_product_velocity(self, velocity: np.ndarray) -> None:
        """Set product velocity vector for ABC prioritization."""
        self.product_velocity = velocity

    # ------------------------------------------------------------------
    # OrderBatch fast path (v0.86.0)
    # ------------------------------------------------------------------

    def allocate_batch(
        self,
        batch: OrderBatch | None,
        extra_orders: list[Order] | None = None,
    ) -> AllocationResult:
        """Vectorized allocation for OrderBatch + optional list[Order] extras.

        The batch path avoids creating Order/OrderLine objects until after
        allocation has pruned zero-qty lines (~30-40% of originals).

        extra_orders (ingredient POs) are small and still go through
        allocate_orders(). Results are merged.

        PERF v0.86.0: Replaces allocate_orders() as the primary entry point.
        """
        allocation_matrix = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )
        unmet_demand_matrix = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )
        all_orders: list[Order] = []

        if batch is not None and batch.n_lines > 0:
            self._allocate_batch_core(
                batch, allocation_matrix, unmet_demand_matrix
            )
            # Materialize surviving lines into Order objects for logistics
            all_orders.extend(self._materialize_orders(batch))

        # Handle ingredient POs via legacy path (small set, ~100-200 orders)
        if extra_orders:
            extra_result = self.allocate_orders(extra_orders)
            all_orders.extend(extra_result.allocated_orders)
            allocation_matrix += extra_result.allocation_matrix
            unmet_demand_matrix += extra_result.unmet_demand_matrix

        # Record unmet demand
        if np.any(unmet_demand_matrix > 0):
            self.state.record_unmet_demand_batch(unmet_demand_matrix)

        return AllocationResult(
            allocated_orders=all_orders,
            allocation_matrix=allocation_matrix,
            unmet_demand_matrix=unmet_demand_matrix,
            order_batch=batch,
        )

    def _allocate_batch_core(
        self,
        batch: OrderBatch,
        allocation_matrix: np.ndarray,
        unmet_demand_matrix: np.ndarray,
    ) -> None:
        """Apply Fair Share allocation to OrderBatch in-place.

        Groups by source_idx, computes per-source demand vector, fill ratios,
        and modifies batch.quantity in-place.  Dead lines (qty → 0) remain
        in arrays but are filtered during materialization.
        """
        # Group lines by source
        unique_sources, src_inverse = np.unique(
            batch.source_idx, return_inverse=True
        )

        for gi in range(len(unique_sources)):
            source_idx = int(unique_sources[gi])
            source_node = self.state.world.nodes.get(
                self.state.node_idx_to_id.get(source_idx, "")
            )
            if source_node is None:
                continue

            mask = src_inverse == gi

            if source_node.type == NodeType.SUPPLIER:
                # Supplier: infinite source, capacity-limited
                total_demand = float(np.sum(batch.quantity[mask]))
                capacity = getattr(
                    source_node, "throughput_capacity", float("inf")
                )
                if total_demand > capacity > 0:
                    batch.quantity[mask] *= capacity / total_demand
                continue

            # RDC/Plant: fair-share allocation against actual inventory
            demand_vector = np.zeros(self.state.n_products)
            np.add.at(
                demand_vector,
                batch.product_idx[mask],
                batch.quantity[mask],
            )

            actual_inv = self.state.actual_inventory[source_idx, :]
            current_inv = np.maximum(0, actual_inv)
            fill_ratios = self._calculate_fill_ratios(demand_vector, current_inv)

            allocated_qty_vec = demand_vector * fill_ratios

            # FIFO age reduction
            old_qty = np.maximum(
                0.0, self.state.actual_inventory[source_idx, :]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction_remaining = np.where(
                    old_qty > 0,
                    np.clip(
                        (old_qty - allocated_qty_vec) / old_qty, 0.0, 1.0
                    ),
                    0.0,
                )
            self.state.inventory_age[source_idx, :] *= fraction_remaining

            # Decrement inventory
            self.state.actual_inventory[source_idx, :] -= allocated_qty_vec
            self.state.perceived_inventory[source_idx, :] -= allocated_qty_vec
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

            allocation_matrix[source_idx, :] += allocated_qty_vec

            # Unmet demand tracking
            unfilled_qty_vec = demand_vector * (1.0 - fill_ratios)
            unmet_mask = (fill_ratios < 1.0) & (
                unfilled_qty_vec > self.epsilon
            )
            unmet_demand_matrix[source_idx, unmet_mask] += unfilled_qty_vec[
                unmet_mask
            ]

            # Apply fill ratios to batch lines in-place
            line_fill = fill_ratios[batch.product_idx[mask]]
            batch.quantity[mask] *= line_fill

    def _materialize_orders(self, batch: OrderBatch) -> list[Order]:
        """Convert surviving batch lines (qty > epsilon) into Order objects.

        Groups by (target_idx, source_idx) since each group = one Order.
        Only creates objects for lines that survived allocation.
        """
        alive = batch.quantity > self.epsilon
        if not np.any(alive):
            return []

        # Extract alive slices (no string arrays — IDs deferred to lookup)
        a_target = batch.target_idx[alive]
        a_source = batch.source_idx[alive]
        a_product = batch.product_idx[alive]
        a_qty = batch.quantity[alive]
        a_price = batch.unit_price[alive]
        a_otype = batch.order_type[alive]
        a_priority = batch.priority[alive]
        a_req_date = batch.requested_date[alive]
        a_inverse = batch._tgt_inverse[alive]

        # Lookups for string IDs (deferred to materialization)
        pid_lookup = self.state.product_idx_to_id
        nid_lookup = self.state.node_idx_to_id
        tgt_src_ids = batch._tgt_source_ids
        tgt_ord_ids = batch._tgt_order_ids

        # Sort by (target, source) for grouping
        sort_key = a_target.astype(np.int64) * 1_000_000 + a_source.astype(
            np.int64
        )
        order = np.argsort(sort_key, kind="mergesort")
        sorted_key = sort_key[order]

        # Find group boundaries
        breaks = np.where(np.diff(sorted_key) != 0)[0] + 1
        group_starts = np.concatenate(([0], breaks))
        group_ends = np.concatenate((breaks, [len(order)]))

        orders: list[Order] = []
        for g_start, g_end in zip(group_starts, group_ends, strict=True):
            idxs = order[g_start:g_end]
            first = idxs[0]

            lines = [
                OrderLine(
                    product_id=pid_lookup[int(a_product[i])],
                    quantity=float(a_qty[i]),
                    product_idx=int(a_product[i]),
                    unit_price=float(a_price[i]),
                )
                for i in idxs
            ]

            ot_int = int(a_otype[first])
            pri_int = int(a_priority[first])
            gi = int(a_inverse[first])

            orders.append(
                Order(
                    id=tgt_ord_ids[gi],
                    source_id=tgt_src_ids[gi],
                    target_id=nid_lookup[int(a_target[first])],
                    creation_day=batch.creation_day,
                    lines=lines,
                    status="CLOSED",
                    order_type=_ORDER_TYPE_INT_MAP.get(
                        ot_int, OrderType.STANDARD
                    ),
                    promo_id=batch.promo_ids.get(int(a_target[first])),
                    priority=_PRIORITY_INT_MAP.get(
                        pri_int, OrderPriority.STANDARD
                    ),
                    requested_date=int(a_req_date[first]),
                    source_idx=int(a_source[first]),
                    target_idx=int(a_target[first]),
                )
            )

        return orders

    # ------------------------------------------------------------------
    # Legacy Order-object path (used for ingredient POs)
    # ------------------------------------------------------------------

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

        return AllocationResult(
            allocated_orders=allocated_orders,
            allocation_matrix=allocation_matrix,
            unmet_demand_matrix=unmet_demand_matrix,
        )
