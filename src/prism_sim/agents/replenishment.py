from typing import Any

import numpy as np

from prism_sim.network.core import Node, NodeType, Order, OrderLine, OrderType, CustomerChannel
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


CHANNEL_POLICIES = {
    # Tightened target-ROP gap to reduce bullwhip (smaller, more frequent orders)
    "B2M_LARGE": {
        "target_days": 7.0,
        "reorder_point_days": 5.0,  # Was 3.0 - tightened
        "batch_size": 500.0,
        "smoothing_factor": 0.3,
    },
    "B2M_CLUB": {
        "target_days": 10.0,
        "reorder_point_days": 7.0,  # Was 4.0 - tightened
        "batch_size": 200.0,
        "smoothing_factor": 0.2,
    },
    "B2M_DISTRIBUTOR": {
        "target_days": 14.0,
        "reorder_point_days": 10.0,  # Was 5.0 - tightened
        "batch_size": 100.0,
        "smoothing_factor": 0.1,
    },
    "ECOMMERCE": {
        "target_days": 5.0,
        "reorder_point_days": 3.0,  # Was 2.0 - tightened
        "batch_size": 50.0,
        "smoothing_factor": 0.4,
    },
    "default": {
        "target_days": 10.0,
        "reorder_point_days": 7.0,  # Was 4.0 - tightened
        "batch_size": 100.0,
        "smoothing_factor": 0.2,
    },
}


class MinMaxReplenisher:
    """
    A simple replenishment agent that implements a (s, S) policy.
    Triggers the Bullwhip effect via order batching.
    Now supports Order Types (Standard, Rush, Promotional).
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        params = (
            config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )

        # Base Policy Parameters (Config driven default fallback)
        self.base_target_days = float(params.get("target_days_supply", 10.0))
        self.base_reorder_point_days = float(params.get("reorder_point_days", 4.0))
        self.base_min_order_qty = float(params.get("min_order_qty", 50.0))
        self.base_batch_size = float(params.get("batch_size_cases", 100.0))

        # Optimization: Cache Store->Supplier map
        self.store_supplier_map = self._build_supplier_map()
        
        # State for Demand Smoothing
        self.smoothed_demand: np.ndarray | None = None
        
        # Vectorized Policy Parameters (N_Nodes, 1)
        self._init_policy_vectors()

    def _init_policy_vectors(self) -> None:
        """Pre-calculate policy vectors based on node channels."""
        n_nodes = self.state.n_nodes
        
        # Initialize with config defaults
        self.target_days_vec = np.full((n_nodes, 1), self.base_target_days)
        self.rop_vec = np.full((n_nodes, 1), self.base_reorder_point_days)
        self.batch_vec = np.full((n_nodes, 1), self.base_batch_size)
        self.min_qty_vec = np.full((n_nodes, 1), self.base_min_order_qty)
        self.alpha_vec = np.full((n_nodes, 1), 0.2)  # Default smoothing factor

        for n_id, node in self.world.nodes.items():
            idx = self.state.node_id_to_idx.get(n_id)
            if idx is None:
                continue

            # Determine policy key
            policy_key = "default"
            if node.channel:
                key = node.channel.name if hasattr(node.channel, "name") else str(node.channel)
                if key.upper() in CHANNEL_POLICIES:
                    policy_key = key.upper()
            
            p = CHANNEL_POLICIES.get(policy_key, CHANNEL_POLICIES["default"])
            
            self.target_days_vec[idx] = p["target_days"]
            self.rop_vec[idx] = p["reorder_point_days"]
            self.batch_vec[idx] = p["batch_size"]
            self.alpha_vec[idx] = p["smoothing_factor"]
            
            # OVERRIDE: Stores order cases, not pallets
            if node.type == NodeType.STORE:
                self.batch_vec[idx] = 20.0  # Reduced from 500/200/100 to 20
                self.min_qty_vec[idx] = 10.0

    def _build_supplier_map(self) -> dict[str, str]:
        """Builds a lookup map for Store -> Source ID."""
        mapping = {}
        for link in self.world.links.values():
            mapping[link.target_id] = link.source_id
        return mapping

    def generate_orders(self, day: int, demand_signal: np.ndarray) -> list[Order]:
        """
        Generates replenishment orders for Retail Stores and downstream DCs.
        Uses exponential smoothing on the demand signal to dampen bullwhip.
        Includes order staggering to prevent synchronized ordering waves.
        """
        orders = []
        week = (day // 7) + 1

        # Order staggering: Stores only order on certain days to reduce bullwhip
        # Use hash of node ID to determine ordering day (spreads across 3-day cycle)
        order_cycle_days = 3  # Stores order every 3 days on average

        # Identify active promos for this week
        active_promos = []
        promotions = self.config.get("promotions", [])
        for p in promotions:
            if p["start_week"] <= week <= p["end_week"]:
                active_promos.append(p)

        # 1. Identify Target Nodes (with staggering for stores)
        target_indices = []
        target_ids = []

        valid_targets = set(self.store_supplier_map.keys())

        for n_id in valid_targets:
            node = self.world.nodes.get(n_id)
            if node and (
                node.type == NodeType.STORE
                or (node.type == NodeType.DC and "RDC" not in node.id)
            ):
                idx = self.state.node_id_to_idx.get(n_id)
                if idx is None:
                    continue

                # Apply order staggering for stores to reduce bullwhip
                # Each store orders on its assigned day in the cycle
                if node.type == NodeType.STORE:
                    order_day = hash(n_id) % order_cycle_days
                    if day % order_cycle_days != order_day:
                        continue  # Skip this store today, it orders on a different day

                target_indices.append(idx)
                target_ids.append(n_id)

        if not target_indices:
            return []

        target_idx_arr = np.array(target_indices)

        # 2. Update Demand Smoothing
        if self.smoothed_demand is None:
            self.smoothed_demand = demand_signal.copy()
        else:
            # S_t = alpha * x_t + (1 - alpha) * S_{t-1}
            self.smoothed_demand = (
                self.alpha_vec * demand_signal + 
                (1.0 - self.alpha_vec) * self.smoothed_demand
            )

        # 3. Get Inventory & Smoothed Demand for Targets
        current_inv = self.state.inventory[target_idx_arr, :]
        avg_demand = self.smoothed_demand[target_idx_arr, :]
        
        # Avoid division by zero or negative targets
        avg_demand = np.maximum(avg_demand, 0.1)

        # 4. Calculate Targets (Vectorized)
        t_days = self.target_days_vec[target_idx_arr]
        rop_days = self.rop_vec[target_idx_arr]
        batch_sz = self.batch_vec[target_idx_arr]
        min_qty = self.min_qty_vec[target_idx_arr]

        target_stock = avg_demand * t_days
        reorder_point = avg_demand * rop_days

        # 5. Determine Order Quantities
        needs_order = current_inv < reorder_point
        raw_qty = target_stock - current_inv
        
        # Use vectorized min_qty
        order_qty = np.where(needs_order, np.maximum(raw_qty, min_qty), 0.0)

        # 6. Apply Batching
        batched_qty = np.ceil(order_qty / batch_sz) * batch_sz

        # 7. Create Orders
        rows, cols = np.nonzero(batched_qty)
        
        # Prepare Order Data
        orders_by_target: dict[int, dict] = {} # target_idx -> {lines: [], type: ...}

        for r, c in zip(rows, cols, strict=True):
            qty = batched_qty[r, c]
            if qty <= 0: continue

            t_idx = int(target_indices[r])
            p_idx = int(c)
            
            if t_idx not in orders_by_target:
                orders_by_target[t_idx] = {
                    "lines": [], 
                    "days_supply_min": 999.0,
                    "promo_id": None
                }
            
            p_id = self.state.product_idx_to_id[p_idx]
            orders_by_target[t_idx]["lines"].append(OrderLine(p_id, qty))
            
            # Check days supply for Rush classification
            # Note: avg_demand here is the subset for this target row r
            d_supply = current_inv[r, c] / avg_demand[r, c]
            if d_supply < orders_by_target[t_idx]["days_supply_min"]:
                orders_by_target[t_idx]["days_supply_min"] = d_supply
            
            # Check Promo
            target_node = self.world.nodes.get(target_ids[r])
            if target_node and not orders_by_target[t_idx]["promo_id"]:
                 for promo in active_promos:
                      cat_match = "all" in promo.get("affected_categories", ["all"]) or \
                                  self.world.products[p_id].category.name in promo.get("affected_categories", [])
                      chan_match = not promo.get("affected_channels") or \
                                   (target_node.channel and target_node.channel.name in promo.get("affected_channels"))
                      
                      if cat_match and chan_match:
                          orders_by_target[t_idx]["promo_id"] = promo["code"]
                          break

        # Generate Objects
        order_count = 0
        for t_idx, data in orders_by_target.items():
            target_id = self.state.node_idx_to_id[t_idx]
            source_id = self.store_supplier_map.get(target_id)
            if not source_id: continue
            
            # Determine Order Type
            o_type = OrderType.STANDARD
            priority = 5
            
            if data["promo_id"]:
                o_type = OrderType.PROMOTIONAL
                priority = 2
            elif data["days_supply_min"] < 2.0: # Critical low stock
                o_type = OrderType.RUSH
                priority = 1
            
            order_count += 1
            orders.append(Order(
                id=f"ORD-{day}-{target_id}-{order_count}",
                source_id=source_id,
                target_id=target_id,
                creation_day=day,
                lines=data["lines"],
                order_type=o_type,
                promo_id=data["promo_id"],
                priority=priority
            ))

        return orders