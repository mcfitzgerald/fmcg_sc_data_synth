from typing import Any

import numpy as np

from prism_sim.network.core import (
    NodeType,
    Order,
    OrderLine,
    OrderType,
    StoreFormat,
)
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

# Default channel policies - overridden by config if present
DEFAULT_CHANNEL_POLICIES: dict[str, dict[str, float]] = {
    "B2M_LARGE": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 500.0,
        "smoothing_factor": 0.3,
    },
    "B2M_CLUB": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 200.0,
        "smoothing_factor": 0.2,
    },
    "B2M_DISTRIBUTOR": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 100.0,
        "smoothing_factor": 0.1,
    },
    "ECOMMERCE": {
        "target_days": 10.0,
        "reorder_point_days": 7.0,
        "batch_size": 50.0,
        "smoothing_factor": 0.4,
    },
    "default": {
        "target_days": 14.0,
        "reorder_point_days": 10.0,
        "batch_size": 100.0,
        "smoothing_factor": 0.2,
    },
}


class MinMaxReplenisher:
    """
    A simple replenishment agent that implements a (s, S) policy.
    Triggers the Bullwhip effect via order batching.
    Now supports Order Types (Standard, Rush, Promotional).

    v0.15.4: Uses allocation outflow (derived demand) as signal for customer DCs
    instead of POS demand to prevent bullwhip cascade. Implements warm start
    from POSEngine equilibrium demand estimate.
    """

    def __init__(
        self,
        world: World,
        state: StateManager,
        config: dict[str, Any],
        warm_start_demand: float = 0.0,
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

        # Load channel policies from config (with defaults fallback)
        config_policies = params.get("channel_profiles", {})
        self.channel_policies: dict[str, dict[str, float]] = {}
        for key, default_policy in DEFAULT_CHANNEL_POLICIES.items():
            self.channel_policies[key] = {
                "target_days": config_policies.get(key, {}).get(
                    "target_days", default_policy["target_days"]
                ),
                "reorder_point_days": config_policies.get(key, {}).get(
                    "reorder_point_days", default_policy["reorder_point_days"]
                ),
                "batch_size": config_policies.get(key, {}).get(
                    "batch_size", default_policy["batch_size"]
                ),
                "smoothing_factor": config_policies.get(key, {}).get(
                    "smoothing_factor", default_policy["smoothing_factor"]
                ),
            }

        # Config-driven thresholds (previously hardcoded)
        self.min_demand_floor = float(params.get("min_demand_floor", 0.1))
        self.default_min_qty = float(params.get("default_min_qty", 10.0))

        # Optimization: Cache Store->Supplier map
        self.store_supplier_map = self._build_supplier_map()

        # State for Demand Smoothing (POS-based for stores)
        self.smoothed_demand: np.ndarray | None = None

        # v0.15.4: Allocation outflow tracking for customer DCs (derived demand)
        # Rolling 7-day history of allocation outflow per node
        # Shape: [7, n_nodes, n_products]
        self.outflow_history: np.ndarray | None = None
        self._outflow_ptr = 0

        # v0.15.9: Inflow tracking - orders RECEIVED by each node
        # This is the true demand signal (what downstream nodes requested)
        # vs outflow which is constrained by available inventory
        # Shape: [7, n_nodes, n_products]
        self.inflow_history: np.ndarray | None = None
        self._inflow_ptr = 0

        # Cache which nodes are customer DCs (use derived demand, not POS)
        self._customer_dc_indices: set[int] = set()
        self._downstream_store_count: dict[int, int] = {}  # dc_idx -> store count
        self._cache_customer_dcs()

        # Vectorized Policy Parameters (N_Nodes, 1)
        self._init_policy_vectors()

        # v0.15.4: Warm start outflow history with equilibrium demand
        # This prevents Day 1-2 bullwhip cascade from cold start
        self._initialize_warm_start(warm_start_demand)

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
                if key.upper() in self.channel_policies:
                    policy_key = key.upper()

            p = self.channel_policies.get(policy_key, self.channel_policies["default"])

            self.target_days_vec[idx] = p["target_days"]
            self.rop_vec[idx] = p["reorder_point_days"]
            self.batch_vec[idx] = p["batch_size"]
            self.alpha_vec[idx] = p["smoothing_factor"]

            # OVERRIDE: Stores order cases, not pallets
            if node.type == NodeType.STORE:
                self.batch_vec[idx] = 20.0  # Reduced from 500/200/100 to 20
                self.min_qty_vec[idx] = self.default_min_qty

    def _cache_customer_dcs(self) -> None:
        """
        Cache indices of customer DCs (RET-DC, DIST-DC, ECOM-FC).

        These nodes don't generate POS demand but fulfill orders to downstream
        stores/customers. Their demand signal should be based on allocation
        outflow (derived demand) rather than POS.

        Also counts downstream stores per DC for warm start calculation.
        """
        # Build reverse map: source_id -> list of target_ids
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            if link.source_id not in downstream_map:
                downstream_map[link.source_id] = []
            downstream_map[link.source_id].append(link.target_id)

        for n_id, node in self.world.nodes.items():
            idx = self.state.node_id_to_idx.get(n_id)
            if idx is None:
                continue

            # Customer DCs: DC type but NOT manufacturer RDCs
            # EXCLUDE ECOM_FC - they're B2C nodes that should use POS demand, not outflow
            # ECOM FCs sell directly to consumers, they don't have downstream stores
            is_ecom_fc = node.store_format == StoreFormat.ECOM_FC
            if node.type == NodeType.DC and not n_id.startswith("RDC-") and not is_ecom_fc:
                self._customer_dc_indices.add(idx)

                # Count downstream stores for this DC
                downstream_ids = downstream_map.get(n_id, [])
                store_count = sum(
                    1
                    for tid in downstream_ids
                    if self.world.nodes.get(tid)
                    and self.world.nodes[tid].type == NodeType.STORE
                )
                self._downstream_store_count[idx] = max(store_count, 1)

    def _initialize_warm_start(self, warm_start_demand: float) -> None:
        """
        Initialize outflow and inflow history with equilibrium demand estimate.

        This prevents the Day 1-2 bullwhip cascade caused by cold start where
        customer DCs have no demand history and use a 0.1 floor.

        For customer DCs, warm start = base_demand × downstream_store_count
        This represents the expected daily flow in steady state.

        v0.15.9: Now also initializes inflow history for true demand tracking.
        """
        if warm_start_demand <= 0:
            return

        # Initialize outflow history with warm start values
        self.outflow_history = np.zeros(
            (7, self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        # v0.15.9: Also initialize inflow history (orders received)
        # In steady state, inflow ≈ outflow (demand matches fulfillment)
        self.inflow_history = np.zeros(
            (7, self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        for dc_idx in self._customer_dc_indices:
            # Scale by downstream store count (more stores = more flow)
            store_count = self._downstream_store_count.get(dc_idx, 1)
            expected_flow = warm_start_demand * store_count

            # Set warm start for all products at this DC
            for i in range(7):
                self.outflow_history[i, dc_idx, :] = expected_flow
                self.inflow_history[i, dc_idx, :] = expected_flow

    def record_outflow(self, allocation_matrix: np.ndarray) -> None:
        """
        Record allocation outflow for demand signal calculation.

        Called after allocation to update the rolling average of outflow
        per node. Customer DCs use this instead of POS demand for their
        replenishment calculations (derived demand from MRP theory).

        Args:
            allocation_matrix: Shape [n_nodes, n_products] - qty allocated out
        """
        if self.outflow_history is None:
            # Cold start fallback (shouldn't happen if warm_start_demand > 0)
            self.outflow_history = np.zeros(
                (7, self.state.n_nodes, self.state.n_products), dtype=np.float64
            )
            # Seed all slots with current allocation to bootstrap
            for i in range(7):
                self.outflow_history[i] = allocation_matrix

        # Update rolling history with actual allocation data
        self.outflow_history[self._outflow_ptr] = allocation_matrix
        self._outflow_ptr = (self._outflow_ptr + 1) % 7

    def get_outflow_demand(self) -> np.ndarray:
        """
        Get smoothed outflow-based demand signal for all nodes.

        Returns:
            Shape [n_nodes, n_products] - rolling 7-day average of outflow
        """
        if self.outflow_history is None:
            return np.zeros((self.state.n_nodes, self.state.n_products))
        return np.mean(self.outflow_history, axis=0)

    def record_inflow(self, orders: list[Order]) -> None:
        """
        Record orders received BY each node (demand signal from downstream).

        This captures the TRUE demand signal - what downstream nodes requested,
        regardless of whether inventory was available to fulfill. Use this
        instead of outflow to prevent demand signal attenuation.

        v0.15.9: Added to fix demand signal attenuation problem.
        Customer DCs should use inflow (what was requested) not outflow
        (what was shipped) to avoid under-ordering when inventory is low.

        Args:
            orders: List of orders generated by generate_orders()
        """
        if self.inflow_history is None:
            # Initialize inflow history
            self.inflow_history = np.zeros(
                (7, self.state.n_nodes, self.state.n_products), dtype=np.float64
            )

        # Reset current day's slot before accumulating
        self.inflow_history[self._inflow_ptr] = 0

        # Aggregate orders by source (the node receiving the order)
        for order in orders:
            source_idx = self.state.node_id_to_idx.get(order.source_id)
            if source_idx is None:
                continue

            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    self.inflow_history[self._inflow_ptr, source_idx, p_idx] += (
                        line.quantity
                    )

        # Advance pointer for circular buffer
        self._inflow_ptr = (self._inflow_ptr + 1) % 7

    def get_inflow_demand(self) -> np.ndarray:
        """
        Get smoothed inflow-based demand signal for all nodes.

        This represents the TRUE demand (orders received) rather than
        constrained demand (orders fulfilled/shipped).

        Returns:
            Shape [n_nodes, n_products] - rolling 7-day average of inflow
        """
        if self.inflow_history is None:
            return np.zeros((self.state.n_nodes, self.state.n_products))
        return np.mean(self.inflow_history, axis=0)

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

        # Order staggering: Stores order daily to improve service level
        # v0.15.8: Reduced from 3-day to 1-day cycle for better service level
        # With higher target/ROP (14/10 days), daily ordering is sustainable
        order_cycle_days = 1  # Stores order daily

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

                # Apply order staggering to reduce bullwhip synchronization
                # Each node orders on its assigned day in the cycle
                # v0.15.4: Apply to both stores AND customer DCs
                if node.type == NodeType.STORE:
                    order_day = hash(n_id) % order_cycle_days
                    if day % order_cycle_days != order_day:
                        continue  # Skip this store today
                elif idx in self._customer_dc_indices:
                    # v0.15.9: Customer DCs now order daily (was 5-day cycle)
                    # Daily ordering creates smoother demand signals upstream
                    # and faster response to inventory shortages
                    dc_cycle_days = 1
                    order_day = hash(n_id) % dc_cycle_days
                    if day % dc_cycle_days != order_day:
                        continue  # Skip this DC today

                target_indices.append(idx)
                target_ids.append(n_id)

        if not target_indices:
            return []

        target_idx_arr = np.array(target_indices)

        # 2. Update Demand Smoothing (POS-based)
        if self.smoothed_demand is None:
            self.smoothed_demand = demand_signal.copy()
        else:
            # S_t = alpha * x_t + (1 - alpha) * S_{t-1}
            self.smoothed_demand = (
                self.alpha_vec * demand_signal
                + (1.0 - self.alpha_vec) * self.smoothed_demand
            )

        # 3. Get Inventory & Demand for Targets
        current_inv = self.state.inventory[target_idx_arr, :]

        # v0.15.9: Use inflow-based demand for customer DCs (orders received)
        # This prevents demand signal attenuation when DCs are short on inventory.
        # Previously used outflow (what was shipped), but that's constrained by
        # inventory - causing under-ordering when DCs are low.
        inflow_demand = self.get_inflow_demand()
        avg_demand = np.zeros((len(target_indices), self.state.n_products))

        for i, t_idx in enumerate(target_indices):
            if t_idx in self._customer_dc_indices:
                # Customer DC: use inflow-based demand (orders received)
                # This is the TRUE demand signal - what downstream nodes requested
                avg_demand[i, :] = inflow_demand[t_idx, :]
            else:
                # Store: use POS-based demand (consumer sales)
                avg_demand[i, :] = self.smoothed_demand[t_idx, :]

        # Avoid division by zero or negative targets
        avg_demand = np.maximum(avg_demand, self.min_demand_floor)

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
            orders_by_target[t_idx]["days_supply_min"] = min(orders_by_target[t_idx]["days_supply_min"], d_supply)

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
