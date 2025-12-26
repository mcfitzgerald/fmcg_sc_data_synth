from typing import List, Dict
import numpy as np
from prism_sim.simulation.world import World
from prism_sim.simulation.state import StateManager
from prism_sim.network.core import Order, OrderLine, NodeType


class MinMaxReplenisher:
    """
    A simple replenishment agent that implements a (s, S) policy.
    Triggers the Bullwhip effect via order batching.
    """

    def __init__(self, world: World, state: StateManager):
        self.world = world
        self.state = state

        # Policy Parameters (Could be config driven)
        self.target_days = 7.0  # Order Up To (S)
        self.reorder_point_days = 3.0  # Reorder Point (s)

        # Bullwhip parameters
        self.min_order_qty = 10.0  # Minimum order size
        self.batch_size = 50.0  # Pallet size approx

    def generate_orders(self, day: int, demand_history: np.ndarray) -> List[Order]:
        """
        Generates replenishment orders for Retail Stores.

        Args:
            day: Current simulation day
            demand_history: (Nodes, Products) tensor of RECENT avg daily demand
                            (For now, we can pass current day's demand or a moving avg)
        """
        orders = []

        # Iterate over stores (vectorized logic is harder for order generation due to object creation,
        # so we iterate. The math can be vectorized though).

        # 1. Identify Stores
        store_indices = []
        store_ids = []
        for n_id, node in self.world.nodes.items():
            if node.type == NodeType.STORE:
                store_indices.append(self.state.node_id_to_idx[n_id])
                store_ids.append(n_id)

        if not store_indices:
            return []

        store_idx_arr = np.array(store_indices)

        # 2. Get Inventory for all stores
        # Shape: [N_Stores, Products]
        current_inv = self.state.inventory[store_idx_arr, :]

        # 3. Get Demand Estimate (for calculating days of supply)
        # Using the passed demand_history (which represents avg daily demand)
        # Shape: [N_Stores, Products]
        avg_demand = demand_history[store_idx_arr, :]

        # Avoid division by zero
        avg_demand = np.maximum(avg_demand, 0.1)

        # 4. Calculate Targets
        target_stock = avg_demand * self.target_days
        reorder_point = avg_demand * self.reorder_point_days

        # 5. Determine Order Quantities
        # Mask where Inv < ReorderPoint
        needs_order = current_inv < reorder_point

        # Raw Quantity = Target - Current
        raw_qty = target_stock - current_inv

        # Apply Needs Order Mask
        order_qty = np.where(needs_order, raw_qty, 0.0)

        # 6. Apply Batching (The Bullwhip Trigger)
        # Round up to nearest batch_size
        # ceil(qty / batch) * batch
        batched_qty = np.ceil(order_qty / self.batch_size) * self.batch_size

        # 7. Convert to Order Objects
        # This part is the loop.
        # We only iterate where order_qty > 0

        rows, cols = np.nonzero(batched_qty)

        # Group by Store (Row) to create one order per store
        orders_by_store: Dict[int, List[OrderLine]] = {}

        for r, c in zip(rows, cols):
            qty = batched_qty[r, c]
            if qty <= 0:
                continue

            global_store_idx = int(store_indices[r])

            if global_store_idx not in orders_by_store:
                orders_by_store[global_store_idx] = []

            p_id = self.state.product_idx_to_id[int(c)]
            orders_by_store[global_store_idx].append(
                OrderLine(product_id=p_id, quantity=qty)
            )

        # Create Order Objects
        order_count = 0
        for s_idx, lines in orders_by_store.items():
            store_id = self.state.node_idx_to_id[int(s_idx)]

            # Find Source (Simplification: Fixed mapping or single RDC)
            # In builder, we linked stores to RDCs.
            # We can find the link where target == store_id
            source_id = self._find_supplier(store_id)

            if source_id:
                order_count += 1
                orders.append(
                    Order(
                        id=f"ORD-{day}-{store_id}-{order_count}",
                        source_id=source_id,
                        target_id=store_id,
                        creation_day=day,
                        lines=lines,
                    )
                )

        return orders

    def _find_supplier(self, store_id: str) -> str:
        # Simple lookup in world links
        # Returns the first source connected to this store
        for link in self.world.links.values():
            if link.target_id == store_id:
                return link.source_id
        return "UNKNOWN"
