from typing import Any

import numpy as np

from prism_sim.network.core import (
    NodeType,
    Order,
    OrderLine,
    OrderPriority,
    OrderType,
    StoreFormat,
)
from prism_sim.product.core import ProductCategory
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
        base_demand_matrix: np.ndarray | None = None,
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
        self.store_batch_size = float(params.get("store_batch_size_cases", 20.0))

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

        # v0.18.0: Calculate expected throughput for customer DCs (physics-based floor)
        # This prevents cold-start under-ordering when stores haven't placed orders yet
        self._expected_throughput: dict[int, np.ndarray] = {}
        if base_demand_matrix is not None:
            self._calculate_expected_throughput(base_demand_matrix)

        # v0.19.0: Echelon Inventory Logic (MEIO)
        # Matrix [n_customer_dcs, n_nodes] to aggregate downstream stores
        self.echelon_matrix: np.ndarray | None = None
        self.dc_idx_to_echelon_row: dict[int, int] = {}
        self._build_echelon_matrix()

        # Vectorized Policy Parameters (N_Nodes, 1)
        self._init_policy_vectors()

        # v0.15.4: Warm start outflow history with equilibrium demand
        # This prevents Day 1-2 bullwhip cascade from cold start
        self._initialize_warm_start(warm_start_demand)

        # v0.16.0: Demand Variance Tracking for Safety Stock
        # Rolling history of demand for standard deviation calculation
        self.variance_lookback = int(params.get("variance_lookback_days", 28))
        self.demand_history_buffer = np.zeros(
            (self.variance_lookback, self.state.n_nodes, self.state.n_products)
        )
        self.history_idx = 0  # Circular buffer index

        # Physics Overhaul: Lead Time Tracking (v0.17.0+)
        # Rolling history of lead times per link (target, source)
        # Shape: [n_nodes (target), n_nodes (source), history_len]
        # Using a fixed history length of 20 samples
        self.lt_history_len = int(params.get("lead_time_history_len", 20))
        self.lead_time_history = np.zeros(
            (self.state.n_nodes, self.state.n_nodes, self.lt_history_len),
            dtype=np.float32
        )
        self.lt_ptr = np.zeros((self.state.n_nodes, self.state.n_nodes), dtype=int)
        self.lt_count = np.zeros((self.state.n_nodes, self.state.n_nodes), dtype=int)

        # PERF: Cached lead time stats [n_nodes, n_nodes] - updated incrementally
        # Avoids calling np.std() 60k+ times per day
        self._lt_mu_cache = np.full(
            (self.state.n_nodes, self.state.n_nodes), 3.0, dtype=np.float32
        )
        self._lt_sigma_cache = np.zeros(
            (self.state.n_nodes, self.state.n_nodes), dtype=np.float32
        )
        # Track which specific links need cache update (not all of them)
        self._lt_dirty_links: set[tuple[int, int]] = set()

        # Physics Overhaul Phase 3: ABC Segmentation
        # Track cumulative volume per product for dynamic classification
        self.product_volume_history = np.zeros(self.state.n_products, dtype=np.float64)

        # Z-score vector per product [n_products]
        # Defaults to B-item target (1.65) until history accumulates
        segmentation = params.get("segmentation", {})
        self.z_score_A = float(segmentation.get("A", 2.33))
        self.z_score_B = float(segmentation.get("B", 1.65))
        self.z_score_C = float(segmentation.get("C", 1.28))

        self.z_scores_vec = np.full(self.state.n_products, self.z_score_B)

        # Vectorized Ingredient Mask (True = Ingredient, skip in Replenishment)
        self.ingredient_mask = np.zeros(self.state.n_products, dtype=bool)
        for p_id, product in self.world.products.items():
            if product.category == ProductCategory.INGREDIENT:
                p_idx = self.state.product_id_to_idx.get(p_id)
                if p_idx is not None:
                    self.ingredient_mask[p_idx] = True

        # v0.20.0: Pending Order Deduplication
        # Tracks orders awaiting fulfillment to prevent duplicate ordering
        # Key: (source_id, target_id, product_id) -> (quantity, creation_day)
        # Orders expire after 14 days to allow retry with fresh calculations
        self.pending_orders: dict[tuple[str, str, str], tuple[float, int]] = {}

    def record_lead_time(self, target_id: str, source_id: str, lead_time_days: float) -> None:
        """
        Record a realized lead time for a specific link.
        Phase 1 of Physics Overhaul.
        """
        t_idx = self.state.node_id_to_idx.get(target_id)
        s_idx = self.state.node_id_to_idx.get(source_id)

        if t_idx is None or s_idx is None:
            return

        ptr = self.lt_ptr[t_idx, s_idx]
        self.lead_time_history[t_idx, s_idx, ptr] = lead_time_days

        self.lt_ptr[t_idx, s_idx] = (ptr + 1) % self.lt_history_len
        if self.lt_count[t_idx, s_idx] < self.lt_history_len:
            self.lt_count[t_idx, s_idx] += 1

        # PERF: Only mark THIS specific link as dirty (not all links)
        self._lt_dirty_links.add((t_idx, s_idx))

    def _update_lt_cache(self) -> None:
        """
        PERF: Incremental update of lead time stats cache.
        Only updates links that changed since last call.
        """
        if not self._lt_dirty_links:
            return

        # Only update stats for dirty links (not all links)
        for t_idx, s_idx in self._lt_dirty_links:
            count = self.lt_count[t_idx, s_idx]
            if count >= 2:
                history = self.lead_time_history[t_idx, s_idx, :count]
                self._lt_mu_cache[t_idx, s_idx] = np.mean(history)
                self._lt_sigma_cache[t_idx, s_idx] = np.std(history, ddof=1)

        self._lt_dirty_links.clear()

    def get_lead_time_stats(self, target_id: str, source_id: str) -> tuple[float, float]:
        """
        Get Mean and StdDev of lead time for a link.
        Returns (mu_L, sigma_L).
        PERF: Now uses cached values instead of computing each time.
        """
        t_idx = self.state.node_id_to_idx.get(target_id)
        s_idx = self.state.node_id_to_idx.get(source_id)

        if t_idx is None or s_idx is None:
            return 3.0, 0.0  # Default fallback

        return float(self._lt_mu_cache[t_idx, s_idx]), float(self._lt_sigma_cache[t_idx, s_idx])

    def get_lead_time_stats_vectorized(
        self, target_indices: np.ndarray, source_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        PERF: Vectorized lead time stats lookup for multiple links.
        Returns (mu_L_vec, sigma_L_vec) arrays.
        """
        return (
            self._lt_mu_cache[target_indices, source_indices],
            self._lt_sigma_cache[target_indices, source_indices],
        )

    def _update_abc_classification(self) -> None:
        """
        Dynamically classify products into A/B/C buckets based on volume.
        Updates self.z_scores_vec.
        
        Logic:
        - Sort products by total volume (descending)
        - Calculate cumulative percentage of volume
        - A: Top X% volume (default 80%) -> Target z_score_A
        - B: Next Y% volume (default 15%) -> Target z_score_B
        - C: Bottom Z% volume (default 5%) -> Target z_score_C
        """
        total_volume = np.sum(self.product_volume_history)
        if total_volume <= 0:
            return

        # Get thresholds from config
        abc_config = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("abc_prioritization", {})
        )
        thresh_a = abc_config.get("a_threshold_pct", 0.80)
        thresh_b = abc_config.get("b_threshold_pct", 0.95)

        # Sort indices by volume descending
        sorted_indices = np.argsort(self.product_volume_history)[::-1]

        cumulative_vol = 0.0

        for p_idx in sorted_indices:
            vol = self.product_volume_history[p_idx]
            cumulative_vol += vol
            pct = cumulative_vol / total_volume

            if pct <= thresh_a:
                self.z_scores_vec[p_idx] = self.z_score_A
            elif pct <= thresh_b:
                self.z_scores_vec[p_idx] = self.z_score_B
            else:
                self.z_scores_vec[p_idx] = self.z_score_C

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
                key = (
                    node.channel.name if hasattr(node.channel, "name")
                    else str(node.channel)
                )
                if key.upper() in self.channel_policies:
                    policy_key = key.upper()

            p = self.channel_policies.get(policy_key, self.channel_policies["default"])

            self.target_days_vec[idx] = p["target_days"]
            self.rop_vec[idx] = p["reorder_point_days"]
            self.batch_vec[idx] = p["batch_size"]
            self.alpha_vec[idx] = p["smoothing_factor"]

            # OVERRIDE: Stores order cases, not pallets
            if node.type == NodeType.STORE:
                self.batch_vec[idx] = self.store_batch_size  # Config-driven override (default 20.0)
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
            # EXCLUDE ECOM_FC - they're B2C nodes
            # B2C nodes should use POS demand, not outflow
            # ECOM FCs sell directly to consumers
            is_ecom_fc = node.store_format == StoreFormat.ECOM_FC
            if (
                node.type == NodeType.DC
                and not n_id.startswith("RDC-")
                and not is_ecom_fc
            ):
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

    def _calculate_expected_throughput(self, base_demand_matrix: np.ndarray) -> None:
        """
        Calculate expected daily throughput for customer DCs from downstream stores.

        This provides a physics-based floor for the demand signal, preventing
        cold-start under-ordering when stores haven't placed orders yet (Days 1-7).

        For each customer DC, we aggregate the base_demand from all downstream
        stores. This represents the expected steady-state demand that will flow
        through the DC, regardless of whether stores have ordered yet.

        Args:
            base_demand_matrix: Shape [n_nodes, n_products] from POSEngine
        """
        # Build downstream map: source_id -> [target_ids]
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            downstream_map.setdefault(link.source_id, []).append(link.target_id)

        for dc_idx in self._customer_dc_indices:
            dc_id = self.state.node_idx_to_id[dc_idx]
            aggregated = np.zeros(self.state.n_products, dtype=np.float64)

            for target_id in downstream_map.get(dc_id, []):
                target_node = self.world.nodes.get(target_id)
                if target_node and target_node.type == NodeType.STORE:
                    t_idx = self.state.node_id_to_idx.get(target_id)
                    if t_idx is not None:
                        aggregated += base_demand_matrix[t_idx, :]

            self._expected_throughput[dc_idx] = aggregated

    def _build_echelon_matrix(self) -> None:
        """
        Builds the Echelon Matrix for MEIO logic.
        
        M_E [n_dcs, n_nodes] where M_E[i, j] = 1 if node j is in DC i's echelon.
        (i.e., node j is the DC itself or a downstream store).
        """
        if not self._customer_dc_indices:
            return

        # Map each Customer DC to a row index
        sorted_dc_indices = sorted(list(self._customer_dc_indices))
        self.dc_idx_to_echelon_row = {idx: i for i, idx in enumerate(sorted_dc_indices)}

        n_rows = len(sorted_dc_indices)
        n_cols = self.state.n_nodes
        self.echelon_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)

        # Build downstream map: source_id -> list of target_ids
        downstream_map: dict[str, list[str]] = {}
        for link in self.world.links.values():
            downstream_map.setdefault(link.source_id, []).append(link.target_id)

        # Populate matrix
        for i, dc_idx in enumerate(sorted_dc_indices):
            dc_id = self.state.node_idx_to_id[dc_idx]

            # 1. Include the DC itself
            self.echelon_matrix[i, dc_idx] = 1.0

            # 2. Include all downstream stores
            children = downstream_map.get(dc_id, [])
            for child_id in children:
                child_node = self.world.nodes.get(child_id)
                # Only include Stores in the echelon (ignore other node types if any)
                if child_node and child_node.type == NodeType.STORE:
                    child_idx = self.state.node_id_to_idx.get(child_id)
                    if child_idx is not None:
                        self.echelon_matrix[i, child_idx] = 1.0

    def _initialize_warm_start(self, warm_start_demand: float) -> None:
        """
        Initialize outflow and inflow history with equilibrium demand estimate.

        This prevents the Day 1-2 bullwhip cascade caused by cold start where
        customer DCs have no demand history and use a 0.1 floor.

        For customer DCs, warm start = base_demand x downstream_store_count
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

    def record_demand(self, daily_demand: np.ndarray) -> None:
        """
        Record daily demand for variance calculation.

        Args:
            daily_demand: Shape [n_nodes, n_products] - POS demand for the day
        """
        idx = self.history_idx % self.variance_lookback
        self.demand_history_buffer[idx] = daily_demand
        self.history_idx += 1

        # Physics Overhaul Phase 3: Update volume history
        # Sum demand across all nodes for network-wide popularity
        total_daily_vol = np.sum(daily_demand, axis=0)
        self.product_volume_history += total_daily_vol

        # Re-classify every week
        if self.history_idx % 7 == 0:
            self._update_abc_classification()

        # PERF: Update lead time stats cache once per day (not per-node)
        self._update_lt_cache()

    def get_demand_std(self) -> np.ndarray:
        """
        Calculate demand standard deviation per node-product.

        Returns:
            Shape [n_nodes, n_products] - Standard deviation of demand
        """
        # Need minimum history to calculate meaningful variance
        repl_config = self.config.get("simulation_parameters", {}).get("agents", {}).get("replenishment", {})
        min_history = int(repl_config.get("min_history_days", 7))
        if self.history_idx < min_history:
            # Fallback for cold start: assume zero std until we have history
            return np.zeros((self.state.n_nodes, self.state.n_products))

        n_samples = min(self.history_idx, self.variance_lookback)
        # Calculate std along time axis (axis 0)
        # Use ddof=1 for sample standard deviation
        return np.array(np.std(self.demand_history_buffer[:n_samples], axis=0, ddof=1))

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
        return np.array(np.mean(self.outflow_history, axis=0))

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
        return np.array(np.mean(self.inflow_history, axis=0))

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
        # v0.15.8: Increased to 3-day cycle to consolidate store signals for DCs
        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        order_cycle_days = int(params.get("order_cycle_days", 3))

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
                    # v0.19.2: Customer DCs using echelon logic ALWAYS order daily
                    # Removing cycle restriction to break negative feedback spiral.
                    # With 3-day cycles, demand signals accumulate but orders don't
                    # flow, causing stores to starve while RDCs accumulate inventory.
                    pass  # Always process Customer DCs, no cycle restriction

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
        # v0.16.0: Use Inventory Position (IP) for (s,S) decisions
        # This is fundamental to (s,S) theory per Zipkin
        # Using only on-hand causes double-ordering
        on_hand_inv = self.state.inventory[target_idx_arr, :]
        in_transit_matrix = self.state.get_in_transit_by_target()
        in_transit_inv = in_transit_matrix[target_idx_arr, :]
        inventory_position = on_hand_inv + in_transit_inv  # IP = On-Hand + In-Transit

        # v0.18.2: Use inflow-based demand for ALL nodes (7-day avg)
        # This replaces exponential smoothing which collapses on sparse signals.
        inflow_demand = self.get_inflow_demand()

        # For Day 1-7, inflow_demand might be low, so we blend with POS demand for stores
        pos_demand = self.smoothed_demand

        avg_demand = np.zeros((len(target_indices), self.state.n_products))

        for i, t_idx in enumerate(target_indices):
            if t_idx in self._customer_dc_indices:
                # Customer DC: use max of inflow and expected throughput
                # v0.18.0: Expected throughput floor prevents cold-start under-ordering
                inflow_signal = inflow_demand[t_idx, :]
                expected = self._expected_throughput.get(t_idx, inflow_signal)
                avg_demand[i, :] = np.maximum(inflow_signal, expected)
            else:
                # Store: use max of 7-day inflow and current smoothed POS
                # This ensures we respond to POS spikes but don't drop to 0 on non-ordering days
                avg_demand[i, :] = np.maximum(inflow_demand[t_idx, :], pos_demand[t_idx, :])

        # Avoid division by zero or negative targets
        avg_demand = np.maximum(avg_demand, self.min_demand_floor)

        # 4. Calculate Targets (Vectorized)
        # v0.16.0: Variance-Aware Safety Stock Logic
        # ROP = LeadTime*Avg + Z*StdDev*sqrt(LeadTime)

        # Get config parameters
        replenishment_params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        lead_time = float(replenishment_params.get("lead_time_days", 3.0))
        min_ss_days = float(replenishment_params.get("min_safety_stock_days", 3.0))

        # Calculate Demand Standard Deviation for target nodes
        full_sigma = self.get_demand_std()
        sigma = full_sigma[target_idx_arr, :]

        # PERF: Vectorized Lead Time Stats Lookup
        # Build source index array for all targets, then do single array lookup
        n_targets = len(target_indices)
        source_idx_arr = np.zeros(n_targets, dtype=np.int32)
        has_source = np.zeros(n_targets, dtype=bool)

        for i, t_idx in enumerate(target_indices):
            target_id = self.state.node_idx_to_id[t_idx]
            source_id = self.store_supplier_map.get(target_id)
            if source_id:
                s_idx = self.state.node_id_to_idx.get(source_id)
                if s_idx is not None:
                    source_idx_arr[i] = s_idx
                    has_source[i] = True

        # Vectorized lookup from cache
        mu_L_vec = np.full((n_targets, 1), lead_time, dtype=np.float32)
        sigma_L_vec = np.zeros((n_targets, 1), dtype=np.float32)

        if np.any(has_source):
            valid_targets = target_idx_arr[has_source]
            valid_sources = source_idx_arr[has_source]
            mu_L_vec[has_source, 0] = self._lt_mu_cache[valid_targets, valid_sources]
            sigma_L_vec[has_source, 0] = self._lt_sigma_cache[valid_targets, valid_sources]

        # 1. Cycle Stock = Average Demand * Mean Lead Time
        # Using realized mean lead time instead of static config
        cycle_stock = avg_demand * mu_L_vec

        # 2. Safety Stock = Full Formula
        # SS = z * sqrt( mu_L * sigma_D^2 + mu_D^2 * sigma_L^2 )
        # Protects against both Demand Variability and Supply Variability

        # Check if we have enough history for sigma
        repl_config = self.config.get("simulation_parameters", {}).get("agents", {}).get("replenishment", {})
        min_history = int(repl_config.get("min_history_days", 7))
        use_variance_logic = self.history_idx >= min_history

        if use_variance_logic:
            # Variance of demand during lead time
            demand_risk_sq = mu_L_vec * (sigma**2)

            # Variance of supply (lead time) affecting total demand
            supply_risk_sq = (avg_demand**2) * (sigma_L_vec**2)

            # Combined Standard Deviation
            combined_sigma = np.sqrt(demand_risk_sq + supply_risk_sq)

            # Physics Overhaul Phase 3: Use Dynamic Z-Scores (ABC Segmentation)
            safety_stock = self.z_scores_vec * combined_sigma

            # Floor safety stock to minimum days coverage (hybrid approach)
            # This protects against ultra-low variance artifacts or zero sigma
            min_safety_stock = avg_demand * min_ss_days
            safety_stock = np.maximum(safety_stock, min_safety_stock)

            reorder_point = cycle_stock + safety_stock

            # Target Stock (Order-Up-To)
            # S = max(TargetDays * AvgDemand, ROP + Buffer)
            target_stock_days = self.target_days_vec[target_idx_arr]
            target_stock = np.maximum(
                avg_demand * target_stock_days,
                reorder_point + avg_demand  # Ensure at least 1 day gap
            )
        else:
            # Cold start fallback: Use legacy fixed-days logic
            rop_days = self.rop_vec[target_idx_arr]
            target_days = self.target_days_vec[target_idx_arr]

            reorder_point = avg_demand * rop_days
            target_stock = avg_demand * target_days

        batch_sz = self.batch_vec[target_idx_arr]
        min_qty = self.min_qty_vec[target_idx_arr]

        # 5. Determine Order Quantities (using Inventory Position for (s,S) decision)
        # v0.16.0: Compare IP against reorder point, order up to target minus IP
        # This prevents double-ordering when shipments are already in transit
        needs_order = inventory_position < reorder_point
        raw_qty = target_stock - inventory_position

        # --- v0.19.0 ECHELON INVENTORY LOGIC (Override for Customer DCs) ---
        # For Customer DCs, use ECHELON DEMAND (downstream POS) but LOCAL IP.
        #
        # BUG FIX (v0.19.1): Original implementation used Echelon IP which included
        # store inventory. This caused DCs to under-order because the system looked
        # "well-stocked" even as stores depleted. The DC's ability to ship depends
        # on its LOCAL inventory, not downstream inventory.
        #
        # Correct approach:
        # - Echelon Demand = M_E @ POS_Demand (aggregate downstream demand)
        # - Local IP = DC's own inventory + in-transit TO the DC
        # - Order = Target - Local IP (where Target is based on Echelon Demand)

        if self.echelon_matrix is not None and self.dc_idx_to_echelon_row:
            # Identify which targets are Customer DCs
            dc_target_indices = []
            echelon_rows = []

            for i, t_idx in enumerate(target_indices):
                if t_idx in self.dc_idx_to_echelon_row:
                    dc_target_indices.append(i)  # Index in 'target_indices' subset
                    echelon_rows.append(self.dc_idx_to_echelon_row[t_idx])

            if dc_target_indices:
                dc_indices = np.array(dc_target_indices)
                row_indices = np.array(echelon_rows)

                # 1. Calculate Echelon Demand (Aggregated downstream POS)
                # smoothed_demand is [n_nodes, n_products], stores have data, DCs ~0
                echelon_demand_all = self.echelon_matrix @ self.smoothed_demand

                # Extract relevant rows for the DCs currently being processed
                current_e_demand = echelon_demand_all[row_indices]

                # 2. Use LOCAL IP for the DC (NOT Echelon IP)
                # This is the DC's own inventory + in-transit to the DC
                # Already calculated above as inventory_position for all targets
                # dc_indices indexes into the target subset, so use inventory_position[dc_indices]
                local_ip = inventory_position[dc_indices]

                # 3. Calculate Echelon-based Target and ROP
                # Target = Echelon_Demand * TargetDays * SafetyMultiplier
                # This represents how much the DC needs to cover downstream demand
                dc_target_days = self.target_days_vec[target_idx_arr[dc_indices]]
                dc_rop_days = self.rop_vec[target_idx_arr[dc_indices]]

                # v0.19.2: Add safety multiplier to account for demand/lead time variance
                # at echelon level. This provides buffer beyond raw echelon demand.
                echelon_safety_multiplier = float(
                    params.get("echelon_safety_multiplier", 1.3)
                )
                echelon_target = current_e_demand * dc_target_days * echelon_safety_multiplier
                echelon_rop = current_e_demand * dc_rop_days * echelon_safety_multiplier

                # 4. Calculate Order Quantity using Local IP
                # Order = Target - Local IP
                echelon_qty = echelon_target - local_ip

                # 5. Order trigger: Local IP < Echelon ROP
                needs_echelon_order = local_ip < echelon_rop

                # Update the main arrays
                raw_qty[dc_indices] = echelon_qty
                needs_order[dc_indices] = needs_echelon_order

        # Use vectorized min_qty
        order_qty = np.where(needs_order, np.maximum(raw_qty, min_qty), 0.0)

        # v0.18.1: Explicitly zero out Ingredient orders in Replenisher
        # Sourcing/Procurement of ingredients is handled by MRPEngine
        order_qty[:, self.ingredient_mask] = 0.0

        # 6. Apply Batching
        batched_qty = np.ceil(order_qty / batch_sz) * batch_sz

        # 7. Create Orders
        rows, cols = np.nonzero(batched_qty)

        # Prepare Order Data
        orders_by_target: dict[int, dict[str, Any]] = {} # target_idx -> {lines: [], type: ...}

        for r, c in zip(rows, cols, strict=True):
            qty = batched_qty[r, c]
            if qty <= 0: continue

            t_idx = int(target_indices[r])
            p_idx = int(c)

            # v0.20.0: Deduplication check - skip if pending order exists
            target_id = target_ids[r]
            source_id = self.store_supplier_map.get(target_id)
            p_id = self.state.product_idx_to_id[p_idx]

            if source_id:
                pending_key = (source_id, target_id, p_id)
                if pending_key in self.pending_orders:
                    # Skip - already have pending order for this route/product
                    continue

            if t_idx not in orders_by_target:
                orders_by_target[t_idx] = {
                    "lines": [],
                    "days_supply_min": 999.0,
                    "promo_id": None
                }

            orders_by_target[t_idx]["lines"].append(OrderLine(p_id, qty))

            # Check days supply for Rush classification
            d_supply = on_hand_inv[r, c] / avg_demand[r, c]
            orders_by_target[t_idx]["days_supply_min"] = min(
                orders_by_target[t_idx]["days_supply_min"], d_supply
            )

            # Check Promo
            target_node = self.world.nodes.get(target_ids[r])
            if target_node and not orders_by_target[t_idx]["promo_id"]:
                for promo in active_promos:
                    affected_cats = promo.get("affected_categories", ["all"])
                    cat_match = (
                        "all" in affected_cats
                        or self.world.products[p_id].category.name
                        in affected_cats
                    )

                    affected_chans = promo.get("affected_channels")
                    chan_match = (
                        not affected_chans
                        or (
                            target_node.channel
                            and target_node.channel.name in affected_chans
                        )
                    )

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
            priority = OrderPriority.LOW
            if node.type == NodeType.DC:
                # DC orders are standard replenishment
                priority = OrderPriority.STANDARD

            if data["promo_id"]:
                o_type = OrderType.PROMOTIONAL
                priority = OrderPriority.HIGH
            elif data["days_supply_min"] < 2.0: # Critical low stock
                o_type = OrderType.RUSH
                priority = OrderPriority.RUSH

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

    # =========================================================================
    # v0.20.0: Pending Order Deduplication Methods
    # =========================================================================

    def record_fulfilled_orders(self, allocated_orders: list[Order]) -> None:
        """
        Remove fulfilled orders from pending tracking.

        Called after allocation to clear orders that were successfully allocated.
        This allows the store to generate new orders on the next cycle if still
        below reorder point (with fresh quantity calculations).
        """
        for order in allocated_orders:
            for line in order.lines:
                key = (order.source_id, order.target_id, line.product_id)
                self.pending_orders.pop(key, None)

    def record_unfulfilled_orders(
        self,
        raw_orders: list[Order],
        allocated_orders: list[Order],
        current_day: int,
    ) -> None:
        """
        Track orders that weren't fulfilled for deduplication.

        Orders that failed allocation are added to pending_orders to prevent
        duplicate ordering on the next cycle. The store will not generate a
        new order for the same (source, target, product) while one is pending.
        """
        allocated_ids = {o.id for o in allocated_orders}
        for order in raw_orders:
            if order.id not in allocated_ids:
                # This order was not allocated (source had no inventory)
                for line in order.lines:
                    key = (order.source_id, order.target_id, line.product_id)
                    # Only add if not already tracked (preserve original creation day)
                    if key not in self.pending_orders:
                        self.pending_orders[key] = (line.quantity, current_day)

    def expire_stale_pending_orders(
        self, current_day: int, timeout_days: int = 14
    ) -> None:
        """
        Remove pending orders older than timeout to allow retry.

        After timeout_days, a pending order is considered stale and removed.
        This allows the store to generate a fresh order with recalculated
        quantities based on current inventory position.

        Args:
            current_day: Current simulation day
            timeout_days: Days after which pending orders expire (default 14)
        """
        stale_keys = [
            key
            for key, (qty, created) in self.pending_orders.items()
            if current_day - created > timeout_days
        ]
        for key in stale_keys:
            del self.pending_orders[key]
