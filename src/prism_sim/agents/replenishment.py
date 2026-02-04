from collections import deque
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
    "MASS_RETAIL": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 500.0,
        "smoothing_factor": 0.3,
    },
    "GROCERY": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 400.0,
        "smoothing_factor": 0.3,
    },
    "CLUB": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 200.0,
        "smoothing_factor": 0.2,
    },
    "PHARMACY": {
        "target_days": 21.0,
        "reorder_point_days": 14.0,
        "batch_size": 100.0,
        "smoothing_factor": 0.2,
    },
    "DISTRIBUTOR": {
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
    "DTC": {
        "target_days": 10.0,
        "reorder_point_days": 5.0,
        "batch_size": 25.0,
        "smoothing_factor": 0.5,
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
        pos_engine: Any = None,
        warm_start_demand: float = 0.0,
        base_demand_matrix: np.ndarray | None = None,
    ) -> None:
        self.world = world
        self.state = state
        self.config = config
        self.pos_engine = pos_engine

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

        # v0.42.0: Load config-only channels not in defaults (e.g., DTC)
        default_fallback = DEFAULT_CHANNEL_POLICIES.get("default", {})
        for key, config_policy in config_policies.items():
            if key.upper() not in self.channel_policies and key != "default":
                self.channel_policies[key.upper()] = {
                    "target_days": config_policy.get(
                        "target_days", default_fallback.get("target_days", 12.0)
                    ),
                    "reorder_point_days": config_policy.get(
                        "reorder_point_days",
                        default_fallback.get("reorder_point_days", 8.0),
                    ),
                    "batch_size": config_policy.get(
                        "batch_size", default_fallback.get("batch_size", 100.0)
                    ),
                    "smoothing_factor": config_policy.get(
                        "smoothing_factor",
                        default_fallback.get("smoothing_factor", 0.2),
                    ),
                }

        # Config-driven thresholds (previously hardcoded)
        self.min_demand_floor = float(params.get("min_demand_floor", 0.1))
        self.default_min_qty = float(params.get("default_min_qty", 10.0))
        self.store_batch_size = float(params.get("store_batch_size_cases", 20.0))
        self.rush_threshold_days = float(params.get("rush_threshold_days", 2.0))

        # v0.45.0: Load format scale factors for store batch sizing
        demand_config = config.get("simulation_parameters", {}).get("demand", {})
        self.format_scale_factors: dict[str, float] = demand_config.get(
            "format_scale_factors", {}
        )

        # Optimization: Cache Store->Supplier map
        self.store_supplier_map = self._build_supplier_map()

        # State for Demand Smoothing (POS-based for stores)
        self.smoothed_demand: np.ndarray | None = None

        # v0.51.0: Config-driven history window (was hardcoded 7)
        self._history_window = int(params.get("outflow_history_days", 5))

        # v0.15.4: Allocation outflow tracking for customer DCs (derived demand)
        # Rolling N-day history of allocation outflow per node
        # Shape: [_history_window, n_nodes, n_products]
        self.outflow_history: np.ndarray | None = None
        self._outflow_ptr = 0

        # v0.15.9: Inflow tracking - orders RECEIVED by each node
        # This is the true demand signal (what downstream nodes requested)
        # vs outflow which is constrained by available inventory
        # Shape: [_history_window, n_nodes, n_products]
        self.inflow_history: np.ndarray | None = None
        self._inflow_ptr = 0

        # Cache which nodes are customer DCs (use derived demand, not POS)
        self._customer_dc_indices: set[int] = set()
        self._downstream_store_count: dict[int, int] = {}  # dc_idx -> store count
        self._cache_customer_dcs()

        # v0.18.0: Calculate expected throughput for customer DCs (physics-based floor)
        # This prevents cold-start under-ordering when stores haven't placed orders yet
        self._expected_throughput: dict[int, np.ndarray] = {}
        # v0.50.0: Store base demand matrix for absolute order cap reference (Fix 1)
        self._base_demand_matrix = base_demand_matrix
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
        # PERF: Sparse lead time storage (replaces dense 6126x6126x20 tensor)
        # Only ~6000 links exist in network - 99.99% of dense tensor is zeros.
        # Using dict[tuple[int, int], deque] saves ~3.5 GB memory.
        self.lt_history_len = int(params.get("lead_time_history_len", 20))
        self._lt_history: dict[tuple[int, int], deque[float]] = {}

        # PERF: Sparse caches - only store stats for active links
        # Default values used for links without history
        self._lt_mu_cache_sparse: dict[tuple[int, int], float] = {}
        self._lt_sigma_cache_sparse: dict[tuple[int, int], float] = {}
        self._lt_default_mu = float(params.get("lead_time_days", 3.0))
        self._lt_default_sigma = 0.0  # No variability assumed until history builds

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

        # v0.47.0: DC order suppression counter (Fix 1 + Fix 5 diagnostics)
        self._dc_order_suppression_count = 0

        # PERF: Pre-computed lookup caches for O(1) array access
        # Replaces 49M+ dict.get() calls with direct array indexing
        self._build_lookup_caches()

        # v0.21.0: Removed pending_orders dict (memory explosion fix)
        # Real retail systems don't track pending orders per-SKU. Instead, they:
        # 1. Use Inventory Position (on-hand + in-transit) for reorder decisions
        # 2. Recalculate requirements fresh each cycle
        # The IP logic at line ~727 already prevents double-ordering correctly.

    def record_lead_time(
        self, target_id: str, source_id: str, lead_time_days: float
    ) -> None:
        """
        Record a realized lead time for a specific link.
        Phase 1 of Physics Overhaul.

        PERF: Uses sparse dict storage with deque as circular buffer.
        """
        t_idx = self.state.node_id_to_idx.get(target_id)
        s_idx = self.state.node_id_to_idx.get(source_id)

        if t_idx is None or s_idx is None:
            return

        link_key = (t_idx, s_idx)

        # Initialize deque for this link if needed (lazy allocation)
        if link_key not in self._lt_history:
            self._lt_history[link_key] = deque(maxlen=self.lt_history_len)

        # Append to circular buffer (deque handles maxlen automatically)
        self._lt_history[link_key].append(lead_time_days)

        # PERF: Only mark THIS specific link as dirty (not all links)
        self._lt_dirty_links.add(link_key)

    def _update_lt_cache(self) -> None:
        """
        PERF: Incremental update of lead time stats cache.
        Only updates links that changed since last call.

        Uses sparse dict storage instead of dense arrays.
        """
        if not self._lt_dirty_links:
            return

        # Only update stats for dirty links (not all links)
        for link_key in self._lt_dirty_links:
            history = self._lt_history.get(link_key)
            if history and len(history) >= 2:  # noqa: PLR2004
                history_arr = np.array(history)
                self._lt_mu_cache_sparse[link_key] = float(np.mean(history_arr))
                sigma = float(np.std(history_arr, ddof=1))
                self._lt_sigma_cache_sparse[link_key] = sigma
            elif history and len(history) == 1:
                # Single sample: use it as mean, zero variance
                self._lt_mu_cache_sparse[link_key] = history[0]
                self._lt_sigma_cache_sparse[link_key] = 0.0

        self._lt_dirty_links.clear()

    def get_lead_time_stats(
        self, target_id: str, source_id: str
    ) -> tuple[float, float]:
        """
        Get Mean and StdDev of lead time for a link.
        Returns (mu_L, sigma_L).
        PERF: Now uses sparse cached values instead of dense arrays.
        """
        t_idx = self.state.node_id_to_idx.get(target_id)
        s_idx = self.state.node_id_to_idx.get(source_id)

        if t_idx is None or s_idx is None:
            return self._lt_default_mu, self._lt_default_sigma

        link_key = (t_idx, s_idx)
        mu = self._lt_mu_cache_sparse.get(link_key, self._lt_default_mu)
        sigma = self._lt_sigma_cache_sparse.get(link_key, self._lt_default_sigma)
        return mu, sigma

    def get_lead_time_stats_vectorized(
        self, target_indices: np.ndarray, source_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        PERF: Vectorized lead time stats lookup for multiple links.
        Returns (mu_L_vec, sigma_L_vec) arrays.

        Uses sparse dict storage with fallback to default values.
        """
        n = len(target_indices)
        mu_vec = np.full(n, self._lt_default_mu, dtype=np.float32)
        sigma_vec = np.full(n, self._lt_default_sigma, dtype=np.float32)

        for i in range(n):
            link_key = (int(target_indices[i]), int(source_indices[i]))
            if link_key in self._lt_mu_cache_sparse:
                mu_vec[i] = self._lt_mu_cache_sparse[link_key]
                sigma_vec[i] = self._lt_sigma_cache_sparse.get(
                    link_key, self._lt_default_sigma
                )

        return mu_vec, sigma_vec

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
            # v0.45.0: Scale batch/min by format_scale_factor (capped at 1.0)
            if node.type == NodeType.STORE:
                scale = 1.0
                if node.store_format:
                    scale = min(
                        self.format_scale_factors.get(
                            node.store_format.name, 1.0
                        ),
                        1.0,
                    )
                self.batch_vec[idx] = max(self.store_batch_size * scale, 5.0)
                self.min_qty_vec[idx] = max(self.default_min_qty * scale, 5.0)

    def _build_lookup_caches(self) -> None:
        """
        PERF: Build pre-computed lookup arrays for O(1) access.

        Replaces dict.get() calls with direct array indexing in hot paths.
        This eliminates ~49M dict lookups per day in _create_order_objects().

        Arrays built:
        - _product_id_arr: p_idx -> product_id (str)
        - _product_category_arr: p_idx -> category.name (str)
        - _node_id_arr: n_idx -> node_id (str)
        """
        n_products = self.state.n_products
        n_nodes = self.state.n_nodes

        # Product caches
        self._product_id_arr: np.ndarray = np.empty(n_products, dtype=object)
        self._product_category_arr: np.ndarray = np.empty(n_products, dtype=object)

        for p_idx in range(n_products):
            p_id = self.state.product_idx_to_id[p_idx]
            self._product_id_arr[p_idx] = p_id
            product = self.world.products.get(p_id)
            if product and product.category:
                self._product_category_arr[p_idx] = product.category.name
            else:
                self._product_category_arr[p_idx] = ""

        # Node caches
        self._node_id_arr: np.ndarray = np.empty(n_nodes, dtype=object)

        for n_idx in range(n_nodes):
            self._node_id_arr[n_idx] = self.state.node_idx_to_id[n_idx]

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
            # EXCLUDE ECOM_FC and DTC_FC - they're B2C nodes
            # B2C nodes should use POS demand, not outflow
            # ECOM/DTC FCs sell directly to consumers
            is_b2c_fc = node.store_format in (StoreFormat.ECOM_FC, StoreFormat.DTC_FC)
            if (
                node.type == NodeType.DC
                and not n_id.startswith("RDC-")
                and not is_b2c_fc
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
        hw = self._history_window
        self.outflow_history = np.zeros(
            (hw, self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        # v0.15.9: Also initialize inflow history (orders received)
        # In steady state, inflow ≈ outflow (demand matches fulfillment)
        self.inflow_history = np.zeros(
            (hw, self.state.n_nodes, self.state.n_products), dtype=np.float64
        )

        for dc_idx in self._customer_dc_indices:
            # Scale by downstream store count (more stores = more flow)
            store_count = self._downstream_store_count.get(dc_idx, 1)
            expected_flow = warm_start_demand * store_count

            # Set warm start for all products at this DC
            for i in range(hw):
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
        repl_config = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        min_history = int(repl_config.get("min_history_days", 7))
        if self.history_idx < min_history:
            # Fallback for cold start: assume zero std until we have history
            return np.zeros((self.state.n_nodes, self.state.n_products))

        n_samples = min(self.history_idx, self.variance_lookback)
        # Calculate std along time axis (axis 0)
        # Use ddof=1 for sample standard deviation
        return np.array(
            np.std(
                self.demand_history_buffer[:n_samples],
                axis=0,
                ddof=1,
            )
        )

    def record_outflow(self, allocation_matrix: np.ndarray) -> None:
        """
        Record allocation outflow for demand signal calculation.

        Called after allocation to update the rolling average of outflow
        per node. Customer DCs use this instead of POS demand for their
        replenishment calculations (derived demand from MRP theory).

        Args:
            allocation_matrix: Shape [n_nodes, n_products] - qty allocated out
        """
        hw = self._history_window
        if self.outflow_history is None:
            # Cold start fallback (shouldn't happen if warm_start_demand > 0)
            self.outflow_history = np.zeros(
                (hw, self.state.n_nodes, self.state.n_products), dtype=np.float64
            )
            # Seed all slots with current allocation to bootstrap
            for i in range(hw):
                self.outflow_history[i] = allocation_matrix

        # Update rolling history with actual allocation data
        self.outflow_history[self._outflow_ptr] = allocation_matrix
        self._outflow_ptr = (self._outflow_ptr + 1) % hw

    def get_outflow_demand(self) -> np.ndarray:
        """
        Get smoothed outflow-based demand signal for all nodes.

        Returns:
            Shape [n_nodes, n_products] - rolling N-day average of outflow
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

        PERF: Uses np.add.at() for efficient sparse accumulation instead of
        nested loops with individual element updates.

        Args:
            orders: List of orders generated by generate_orders()
        """
        hw = self._history_window
        if self.inflow_history is None:
            # Initialize inflow history
            self.inflow_history = np.zeros(
                (hw, self.state.n_nodes, self.state.n_products), dtype=np.float64
            )

        # Reset current day's slot before accumulating
        self.inflow_history[self._inflow_ptr] = 0

        # PERF: Pre-build coordinate arrays for scatter-add
        source_indices: list[int] = []
        product_indices: list[int] = []
        quantities: list[float] = []

        for order in orders:
            source_idx = self.state.node_id_to_idx.get(order.source_id)
            if source_idx is None:
                continue

            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    source_indices.append(source_idx)
                    product_indices.append(p_idx)
                    quantities.append(line.quantity)

        # PERF: Single scatter-add operation instead of N individual updates
        if source_indices:
            np.add.at(
                self.inflow_history[self._inflow_ptr],
                (np.array(source_indices, dtype=np.intp),
                 np.array(product_indices, dtype=np.intp)),
                np.array(quantities, dtype=np.float64),
            )

        # Advance pointer for circular buffer
        self._inflow_ptr = (self._inflow_ptr + 1) % self._history_window

    def get_inflow_demand(self) -> np.ndarray:
        """
        Get smoothed inflow-based demand signal for all nodes.

        This represents the TRUE demand (orders received) rather than
        constrained demand (orders fulfilled/shipped).

        Returns:
            Shape [n_nodes, n_products] - rolling N-day average of inflow
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

        v0.38.0: Now includes unmet demand in signal calculation and decays
        unmet demand after orders are placed to prevent accumulation.
        """
        self.current_day = day
        # 1. Identify Target Nodes
        target_indices, target_ids = self._identify_target_nodes(day)
        if not target_indices:
            return []

        # 2. Update Demand Smoothing
        self._update_demand_smoothing(demand_signal)

        # 3. Calculate Average Demand (now includes unmet demand)
        avg_demand = self._calculate_average_demand(target_indices)

        # 4. Calculate Base Order Logic (s,S)
        (
            needs_order, raw_qty, on_hand_inv, inventory_position
        ) = self._calculate_base_order_logic(
            target_indices, avg_demand
        )

        # 5. Apply Echelon Logic Override (MEIO)
        self._apply_echelon_logic(
            target_indices, inventory_position, needs_order, raw_qty
        )

        # 6. Finalize Quantities (Min Qty, Masking, Batching)
        batched_qty = self._finalize_quantities(target_indices, needs_order, raw_qty)

        # 7. Create Order Objects
        orders = self._create_order_objects(
            day, target_indices, target_ids, batched_qty, avg_demand, on_hand_inv
        )

        # v0.38.0: Decay unmet demand after orders placed to prevent accumulation.
        # Orders should capture the unmet demand, so we decay what was recorded.
        # v0.39.0: Slowed decay from 0.5 to 0.85 (15%/day) to preserve signal.
        # v0.39.5: Further slowed to 0.95 (5%/day) for C-item recovery.
        # v0.47.0: Config-driven decay (default 0.85, 15%/day). With 0.95 decay,
        # a 1-day shortage persists ~60 days → permanent surplus. 0.85 gives ~15
        # days persistence, still enough for C-item recovery with 1-day DC lead time.
        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        unmet_decay = float(params.get("unmet_demand_decay", 0.85))
        self.state.decay_unmet_demand(decay_factor=unmet_decay)

        return orders

    def _identify_target_nodes(self, day: int) -> tuple[list[int], list[str]]:
        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        order_cycle_days = int(params.get("order_cycle_days", 3))

        # v0.39.3: Emergency DOS threshold for bypassing stagger
        # When a node has ANY product with DOS < threshold, bypass stagger
        # to prevent stockout accumulation during off-schedule days
        emergency_dos_threshold = float(params.get("emergency_dos_threshold", 2.0))

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

                # v0.39.3 FIX: Emergency bypass for critical inventory
                #
                # BUG: When inventory = 0, replenishment still applies staggering.
                # Store waits 2-3 days for its "scheduled" order day while
                # customers see empty shelves. No demand signal flows upstream.
                #
                # INDUSTRY REALITY (Walmart, Target):
                # Emergency orders (expedited replenishment) bypass scheduling.
                # Zero/critical inventory triggers immediate action.
                #
                # FIX: Check minimum DOS at node. If < threshold, bypass stagger.
                is_emergency = False
                if node.type == NodeType.STORE:
                    # Get node inventory and demand signal
                    node_inv = self.state.actual_inventory[idx, :]
                    # Use smoothed demand if available, else use small floor
                    if self.smoothed_demand is not None:
                        node_demand = self.smoothed_demand[idx, :]
                    else:
                        node_demand = np.ones(self.state.n_products) * 0.1

                    # Calculate DOS for non-ingredient products
                    with np.errstate(divide="ignore", invalid="ignore"):
                        node_dos = np.where(
                            node_demand > self.min_demand_floor,
                            node_inv / node_demand,
                            np.inf,
                        )
                    # Mask out ingredients (they don't stockout at stores)
                    node_dos[self.ingredient_mask] = np.inf

                    # Emergency if ANY product has critical DOS
                    min_dos = np.min(node_dos)
                    is_emergency = min_dos < emergency_dos_threshold

                # Apply order staggering (unless emergency)
                if node.type == NodeType.STORE and not is_emergency:
                    order_day = hash(n_id) % order_cycle_days
                    if day % order_cycle_days != order_day:
                        continue

                target_indices.append(idx)
                target_ids.append(n_id)

        return target_indices, target_ids

    def _update_demand_smoothing(self, demand_signal: np.ndarray) -> None:
        if self.smoothed_demand is None:
            self.smoothed_demand = demand_signal.copy()
        else:
            self.smoothed_demand = (
                self.alpha_vec * demand_signal
                + (1.0 - self.alpha_vec) * self.smoothed_demand
            )

    def _calculate_average_demand(self, target_indices: list[int]) -> np.ndarray:
        inflow_demand = self.get_inflow_demand()
        # v0.52.0: Use outflow demand for DCs (consistent with throughput-based
        # echelon logic). Outflow = what DC actually ships to stores.
        outflow_demand = self.get_outflow_demand()
        pos_demand = self.smoothed_demand

        # v0.38.0: Get unmet demand to boost signal where allocation failed
        # This prevents "demand signal collapse" where shortages hide true demand
        unmet_demand = self.state.get_unmet_demand()

        # v0.36.0 Proactive Demand Sensing
        if hasattr(self, "pos_engine") and self.pos_engine is not None:
            # Look ahead by 14 days to capture upcoming structure
            forecast_horizon = 14
            proactive_matrix = self.pos_engine.get_deterministic_forecast(
                self.current_day, forecast_horizon, aggregated=False
            )
            # Planning rate = Average daily volume over horizon
            proactive_rate_matrix = proactive_matrix / forecast_horizon
        else:
            proactive_rate_matrix = None

        n_targets = len(target_indices)
        avg_demand = np.zeros((n_targets, self.state.n_products))

        # v0.51.0: Anti-windup floor gating — demand floor scales with
        # inventory deficit. At low DOS, full floor protects against death
        # spirals. At target DOS, floor disengages so system can self-correct.
        floor_gating_enabled = bool(
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
            .get("floor_gating_enabled", True)
        )

        # v0.52.0: Throughput floor for DCs (same as echelon logic)
        throughput_floor_pct = float(
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
            .get("throughput_floor_pct", 0.7)
        )

        for i, t_idx in enumerate(target_indices):
            if t_idx in self._customer_dc_indices:
                # v0.52.0: Use outflow (shipments to stores) as DC demand signal.
                # Previously used inflow (orders received), which is the echelon
                # demand signal and caused ordering-vs-shipping asymmetry.
                outflow_signal = outflow_demand[t_idx, :]
                expected = self._expected_throughput.get(t_idx, outflow_signal)

                # Floor at percentage of expected throughput for ramp protection
                base_signal = np.maximum(
                    outflow_signal, expected * throughput_floor_pct
                )

                if floor_gating_enabled:
                    # Anti-windup: floor engages proportionally to inventory deficit
                    # When DOS < target → full floor (prevent death spiral)
                    # When DOS >= target → floor disengages (allow drawdown)
                    local_inv = self.state.actual_inventory[t_idx, :]
                    local_demand = np.maximum(expected, 0.1)
                    local_dos = local_inv / local_demand
                    target_dos = self.target_days_vec[
                        np.array([t_idx])
                    ][0]

                    # Smooth ramp: floor_weight = clip((target - dos) / target, 0, 1)
                    # At DOS=0: weight=1.0 (full floor). At DOS=target: weight=0.0
                    floor_weight = np.clip(
                        (target_dos - local_dos) / np.maximum(target_dos, 1.0),
                        0.0,
                        1.0,
                    )
                    floor_signal = expected * floor_weight
                    base_signal = np.maximum(base_signal, floor_signal)
            else:
                base_signal = np.maximum(inflow_demand[t_idx, :], pos_demand[t_idx, :]) # type: ignore

            # v0.38.0: Add unmet demand to signal (weighted)
            # v0.39.0: Increased weight from 0.5 to 1.0 for full signal strength.
            # v0.47.0: Reduced to config-driven weight (default 0.5) to break
            # the ratchet effect where every shortage creates a permanent surplus.
            # C-items still get recovery via 0.5 weight + 0.85 decay (~7 days boost).
            unmet_weight = float(
                self.config.get("simulation_parameters", {})
                .get("agents", {})
                .get("replenishment", {})
                .get("unmet_demand_weight", 0.5)
            )
            # v0.50.0 Fix 2: Non-additive unmet signal. Previously additive (+=)
            # which meant signal = demand + 0.5*unmet → unbounded ratchet.
            # Now signal = max(demand, 0.5*unmet) → capped at whichever is larger.
            unmet_signal = unmet_demand[t_idx, :] * unmet_weight
            base_signal = np.maximum(base_signal, unmet_signal)

            # Blending logic: 50% Reactive (History) + 50% Proactive (Structure)
            if proactive_rate_matrix is not None:
                proactive_signal = proactive_rate_matrix[t_idx, :]
                avg_demand[i, :] = 0.5 * base_signal + 0.5 * proactive_signal
            else:
                avg_demand[i, :] = base_signal

        return np.maximum(avg_demand, self.min_demand_floor)

    def _calculate_base_order_logic(
        self, target_indices: list[int], avg_demand: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_idx_arr = np.array(target_indices)

        on_hand_inv = self.state.inventory[target_idx_arr, :]
        in_transit_matrix = self.state.get_in_transit_by_target()
        in_transit_inv = in_transit_matrix[target_idx_arr, :]
        inventory_position = on_hand_inv + in_transit_inv

        # Get ROP Parameters
        replenishment_params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        lead_time = float(replenishment_params.get("lead_time_days", 3.0))
        min_ss_days = float(replenishment_params.get("min_safety_stock_days", 3.0))

        # Variance Logic
        full_sigma = self.get_demand_std()
        sigma = full_sigma[target_idx_arr, :]

        # Lead Time Stats
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

        mu_L_vec = np.full((n_targets, 1), lead_time, dtype=np.float32)
        sigma_L_vec = np.zeros((n_targets, 1), dtype=np.float32)

        if np.any(has_source):
            valid_target_idx = target_idx_arr[has_source]
            valid_source_idx = source_idx_arr[has_source]
            # PERF: Use sparse cache via vectorized lookup method
            mu_vals, sigma_vals = self.get_lead_time_stats_vectorized(
                valid_target_idx, valid_source_idx
            )
            mu_L_vec[has_source, 0] = mu_vals
            sigma_L_vec[has_source, 0] = sigma_vals

        cycle_stock = avg_demand * mu_L_vec

        # Safety Stock Calculation
        repl_config = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        min_history = int(repl_config.get("min_history_days", 7))
        use_variance_logic = self.history_idx >= min_history

        if use_variance_logic:
            demand_risk_sq = mu_L_vec * (sigma**2)
            supply_risk_sq = (avg_demand**2) * (sigma_L_vec**2)
            combined_sigma = np.sqrt(demand_risk_sq + supply_risk_sq)
            safety_stock = self.z_scores_vec * combined_sigma
            min_safety_stock = avg_demand * min_ss_days
            safety_stock = np.maximum(safety_stock, min_safety_stock)

            reorder_point = cycle_stock + safety_stock

            target_stock_days = self.target_days_vec[target_idx_arr]
            target_stock = np.maximum(
                avg_demand * target_stock_days,
                reorder_point + avg_demand
            )
        else:
            rop_days = self.rop_vec[target_idx_arr]
            target_days = self.target_days_vec[target_idx_arr]
            reorder_point = avg_demand * rop_days
            target_stock = avg_demand * target_days

        needs_order = inventory_position < reorder_point
        raw_qty = target_stock - inventory_position

        # v0.39.0: Zero-inventory order generation to prevent signal collapse.
        # Stores with empty shelves must still generate orders, otherwise:
        # 1. No demand signal flows upstream
        # 2. MRP never sees the shortage
        # 3. C-items (low priority) get starved permanently
        # Force order generation for any product with zero on-hand inventory.
        zero_inventory_mask = on_hand_inv <= 0
        needs_order = needs_order | zero_inventory_mask

        # Ensure raw_qty is at least target_stock for zero-inventory items
        # (if inventory_position is negative from backorders, raw_qty may be high)
        raw_qty = np.where(
            zero_inventory_mask & (raw_qty < target_stock),
            target_stock,
            raw_qty
        )

        return needs_order, raw_qty, on_hand_inv, inventory_position

    def _apply_echelon_logic(
        self,
        target_indices: list[int],
        inventory_position: np.ndarray,
        needs_order: np.ndarray,
        raw_qty: np.ndarray
    ) -> None:
        if self.echelon_matrix is None or not self.dc_idx_to_echelon_row:
            return

        dc_target_indices = []

        for i, t_idx in enumerate(target_indices):
            if t_idx in self.dc_idx_to_echelon_row:
                dc_target_indices.append(i)

        if not dc_target_indices:
            return

        dc_indices = np.array(dc_target_indices)
        target_idx_arr = np.array(target_indices)

        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )

        # v0.52.0: Throughput-based DC ordering replaces echelon-demand ordering.
        # Root cause of +467% DC drift: DCs ordered based on echelon demand
        # (DC + 100 stores), but stores manage their own inventory independently.
        # DC ends up holding whatever stores don't pull → persistent accumulation.
        #
        # Fix: DC orders = outflow (what it ships to stores) + local buffer correction.
        # Matches real FMCG practice (Walmart DC orders ≈ throughput ± buffer).
        dc_actual_indices = target_idx_arr[dc_indices]

        # 1. Get outflow-based demand signal (rolling 5-day shipment average)
        dc_outflow = self.get_outflow_demand()[dc_actual_indices, :]

        # Floor at expected throughput for demand ramp protection
        throughput_floor_pct = float(params.get("throughput_floor_pct", 0.7))
        for i, dc_idx in enumerate(dc_actual_indices):
            expected = self._expected_throughput.get(int(dc_idx))
            if expected is not None:
                dc_outflow[i, :] = np.maximum(
                    dc_outflow[i, :], expected * throughput_floor_pct
                )

        # 2. Local buffer target based on outflow (not echelon demand)
        dc_buffer_days = float(params.get("dc_buffer_days", 10.0))
        dc_local_target = dc_outflow * dc_buffer_days

        # 3. Local inventory position (DC only, NOT echelon)
        dc_local_inv = self.state.actual_inventory[dc_actual_indices, :]
        dc_in_transit = self.state.get_in_transit_by_target()[dc_actual_indices, :]
        dc_local_ip = dc_local_inv + dc_in_transit

        # 4. Smooth correction toward local target
        dc_correction_days = float(params.get("dc_correction_days", 7.0))
        dc_correction_cap_pct = float(params.get("dc_correction_cap_pct", 0.5))
        local_correction = (dc_local_target - dc_local_ip) / dc_correction_days
        max_correction = dc_outflow * dc_correction_cap_pct
        local_correction = np.clip(local_correction, -max_correction, max_correction)

        # 5. DC order rate = outflow + correction
        dc_order_rate = np.maximum(dc_outflow + local_correction, 0.0)

        # Order when the rate-based qty is meaningful (>5% of outflow)
        rate_threshold = dc_outflow * 0.05
        needs_echelon_order = dc_order_rate > rate_threshold

        # Use rate-based qty instead of gap-to-target
        echelon_qty = dc_order_rate

        # v0.47.0 Fix 1: DC-Level DOS Cap — suppress ordering when local DC
        # inventory already exceeds ABC-differentiated threshold. This prevents
        # DCs from accumulating inventory beyond safety valve.
        dc_dos_cap_a = float(params.get("dc_dos_cap_a", 15.0))
        dc_dos_cap_b = float(params.get("dc_dos_cap_b", 20.0))
        dc_dos_cap_c = float(params.get("dc_dos_cap_c", 25.0))

        # Build per-product DOS cap vector based on ABC classification
        dc_dos_cap_vec = np.full(self.state.n_products, dc_dos_cap_c)
        # Use z_scores_vec as ABC proxy (A=2.33, B=1.65, C varies)
        dc_dos_cap_vec[self.z_scores_vec >= self.z_score_A] = dc_dos_cap_a
        dc_dos_cap_vec[
            (self.z_scores_vec >= self.z_score_B) & (self.z_scores_vec < self.z_score_A)
        ] = dc_dos_cap_b

        # Calculate LOCAL DC DOS using outflow demand (consistent with
        # throughput-based ordering). dc_local_inv and dc_actual_indices
        # already computed above.
        dc_local_demand = np.maximum(dc_outflow, 0.1)
        dc_local_dos = dc_local_inv / dc_local_demand

        # Suppress order for products where local DC DOS > cap
        dc_over_cap = dc_local_dos > dc_dos_cap_vec[np.newaxis, :]
        suppressed_count = int(np.sum(needs_echelon_order & dc_over_cap))
        needs_echelon_order = needs_echelon_order & ~dc_over_cap

        # Track suppression count for diagnostics (v0.47.0 Fix 5)
        self._dc_order_suppression_count = suppressed_count

        raw_qty[dc_indices] = echelon_qty
        needs_order[dc_indices] = needs_echelon_order

    def _finalize_quantities(
        self,
        target_indices: list[int],
        needs_order: np.ndarray,
        raw_qty: np.ndarray
    ) -> np.ndarray:
        target_idx_arr = np.array(target_indices)
        batch_sz = self.batch_vec[target_idx_arr]
        min_qty = self.min_qty_vec[target_idx_arr]

        order_qty = np.where(needs_order, np.maximum(raw_qty, min_qty), 0.0)
        order_qty[:, self.ingredient_mask] = 0.0

        # v0.50.0 Fix 1: Anchor order cap to EXOGENOUS base demand (POS).
        # Previously used get_inflow_demand() which is endogenous — when orders
        # inflate during bullwhip, the cap inflates with them (useless).
        # Base demand from POSEngine is the fixed reference point.
        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        target_days = self.target_days_vec[target_idx_arr]
        if self._base_demand_matrix is not None:
            base_ref = self._base_demand_matrix[target_idx_arr, :].copy()
            # For customer DCs: use expected throughput (aggregate downstream)
            for i, t_idx in enumerate(target_indices):
                if t_idx in self._customer_dc_indices:
                    expected = self._expected_throughput.get(t_idx)
                    if expected is not None:
                        base_ref[i, :] = expected
            base_ref = np.maximum(base_ref, self.min_demand_floor)
        else:
            # Fallback if no base demand matrix available
            base_ref = np.maximum(
                self.get_inflow_demand()[target_idx_arr, :], self.min_demand_floor
            )

        cap_mult = float(params.get("order_cap_base_demand_multiplier", 3.0))
        max_order = base_ref * target_days * cap_mult
        max_order = np.maximum(max_order, min_qty)
        order_qty = np.minimum(order_qty, max_order)

        batched_qty = np.ceil(order_qty / batch_sz) * batch_sz
        return np.asarray(batched_qty)

    def _create_order_objects(
        self,
        day: int,
        target_indices: list[int],
        target_ids: list[str],
        batched_qty: np.ndarray,
        avg_demand: np.ndarray,
        on_hand_inv: np.ndarray
    ) -> list[Order]:
        week = (day // 7) + 1
        active_promos = []
        promotions = self.config.get("promotions", [])
        for p in promotions:
            if p["start_week"] <= week <= p["end_week"]:
                active_promos.append(p)

        rows, cols = np.nonzero(batched_qty)
        orders_by_target: dict[int, dict[str, Any]] = {}

        for r, c in zip(rows, cols, strict=True):
            qty = batched_qty[r, c]
            if qty <= 0:
                continue

            t_idx = int(target_indices[r])
            p_idx = int(c)
            # PERF: Use cached array instead of dict lookup
            p_id = self._product_id_arr[p_idx]

            if t_idx not in orders_by_target:
                orders_by_target[t_idx] = {
                    "lines": [],
                    "days_supply_min": float("inf"),
                    "promo_id": None
                }

            orders_by_target[t_idx]["lines"].append(OrderLine(p_id, qty))

            # Days Supply Check
            if avg_demand[r, c] > 0:
                d_supply = on_hand_inv[r, c] / avg_demand[r, c]
            else:
                d_supply = float("inf")

            orders_by_target[t_idx]["days_supply_min"] = min(
                orders_by_target[t_idx]["days_supply_min"], d_supply
            )

            # Promo Logic
            target_node = self.world.nodes.get(target_ids[r])
            if target_node and not orders_by_target[t_idx]["promo_id"]:
                for promo in active_promos:
                    affected_cats = promo.get("affected_categories", ["all"])
                    # PERF: Use cached array instead of dict lookup
                    cat_match = (
                        "all" in affected_cats
                        or self._product_category_arr[p_idx] in affected_cats
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

        orders = []
        order_count = 0
        for t_idx, data in orders_by_target.items():
            # PERF: Use cached array instead of dict lookup
            target_id = self._node_id_arr[t_idx]
            source_id = self.store_supplier_map.get(target_id)
            if not source_id:
                continue

            target_node = self.world.nodes.get(target_id)

            o_type = OrderType.STANDARD
            priority = OrderPriority.LOW
            if target_node and target_node.type == NodeType.DC:
                priority = OrderPriority.STANDARD

            if data["promo_id"]:
                o_type = OrderType.PROMOTIONAL
                priority = OrderPriority.HIGH
            elif data["days_supply_min"] < self.rush_threshold_days:
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
    # v0.21.0: Removed Pending Order Deduplication Methods
    # =========================================================================
    # The pending_orders dict and associated methods were removed because:
    # 1. Real retail systems (Walmart, Target) don't track pending orders per-SKU
    # 2. They use Inventory Position (on-hand + in-transit) for reorder decisions
    # 3. The dict could grow to 3M+ entries (6000 stores x 500 SKUs) causing
    #    memory explosion in 365-day runs
    # 4. The IP logic at line ~727 already prevents double-ordering correctly
    #
    # Removed methods:
    # - record_fulfilled_orders()
    # - record_unfulfilled_orders()
    # - expire_stale_pending_orders()
