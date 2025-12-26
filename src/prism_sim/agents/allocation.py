import numpy as np

from prism_sim.constants import EPSILON
from prism_sim.network.core import Order
from prism_sim.simulation.state import StateManager


class AllocationAgent:
    """
    Handles the allocation of inventory to orders (Level 9).
    Implements 'Fair Share' logic when demand > supply.
    """

    def __init__(self, state: StateManager) -> None:
        self.state = state

    def allocate_orders(self, orders: list[Order]) -> list[Order]:
        """
        Processes orders against available inventory at the source.
        Modifies order quantities in-place based on availability.
        Updates (decrements) source inventory immediately.

        Returns:
            List of allocated orders (may be fewer if some are fully cancelled,
            though typically we just reduce qty to 0).
        """
        # 1. Group by Source Node
        orders_by_source: dict[str, list[Order]] = {}
        for order in orders:
            if order.source_id not in orders_by_source:
                orders_by_source[order.source_id] = []
            orders_by_source[order.source_id].append(order)

        allocated_orders = []

        # 2. Process each source independently
        for source_id, source_orders in orders_by_source.items():
            source_idx = self.state.node_id_to_idx.get(source_id)
            if source_idx is None:
                # Source not found in state (e.g. external supplier infinite capacity?)
                # For now, assume infinite if not in state tracking (or log warning)
                # But our RDCs are in state. Suppliers might not be fully tracked yet.
                # If source is external supplier, we might assume 100% fill for now
                # unless we track supplier inventory.
                # Let's check if it's in the state mapping.
                allocated_orders.extend(source_orders)
                continue

            # 3. Calculate Total Demand per Product for this Source
            # We need a vector of demand: [Products]
            demand_vector = np.zeros(self.state.n_products)

            # Map to keep track of which order requested what for easy update
            # order_requests[order_idx][product_idx] = qty
            # This is slightly expensive, but Python loops over orders are inevitable here

            for order in source_orders:
                for line in order.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        demand_vector[p_idx] += line.quantity

            # 4. Check Inventory
            current_inv = self.state.inventory[source_idx, :]

            # 5. Calculate Fill Rates (Vectorized)
            # if demand > inv: ratio = inv / demand, else 1.0
            # Handle divide by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                fill_ratios = np.where(
                    demand_vector > current_inv, current_inv / demand_vector, 1.0
                )
                # Clean up NaNs from 0/0
                fill_ratios = np.nan_to_num(fill_ratios, nan=1.0)

            # 6. Decrement Inventory
            # We remove what we are about to ship
            # allocated_qty = min(demand, inv) -> effectively demand * fill_ratio
            allocated_total = demand_vector * fill_ratios
            self.state.inventory[source_idx, :] -= allocated_total

            # 7. Update Orders
            for order in source_orders:
                new_lines = []
                for line in order.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        ratio = fill_ratios[p_idx]
                        new_qty = line.quantity * ratio
                        if new_qty > EPSILON:  # Filter out near-zero
                            line.quantity = new_qty
                            new_lines.append(line)
                    else:
                        # Product not tracked? Keep it?
                        new_lines.append(line)

                order.lines = new_lines
                if new_lines:
                    allocated_orders.append(order)

        return allocated_orders
