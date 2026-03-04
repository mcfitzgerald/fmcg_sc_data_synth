from collections import defaultdict, deque
from typing import Any

import numpy as np

from prism_sim.network.core import (
    Link,
    NodeType,
    Order,
    OrderBatch,
    OrderPriority,
    StoreFormat,
)
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

        # v0.89.0: DRP suppresses DC→RDC pull orders when enabled
        drp_cfg = params.get("drp_distribution", {})
        self._drp_suppresses_dc_pull = bool(drp_cfg.get("suppress_dc_pull", False))

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

        # v0.81.0: Proactive demand sensing lookahead window
        self._forecast_horizon = int(
            params.get("proactive_demand_horizon_days", 14)
        )

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

        # PERF v0.69.3: Day-level cache for get_demand_std
        # (invalidated in record_demand)
        self._demand_std_cache: np.ndarray | None = None
        self._min_history_days = int(params.get("min_history_days", 7))

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

        # v0.86.0: Welford accumulators per link — [count, mean, M2]
        self._lt_welford: dict[tuple[int, int], list[float]] = {}

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

        # Non-FG mask (True = skip in Replenishment)
        # Includes raw materials AND bulk intermediates
        self.ingredient_mask = np.zeros(self.state.n_products, dtype=bool)
        for p_id, product in self.world.products.items():
            if not product.is_finished_good:
                p_idx = self.state.product_id_to_idx.get(p_id)
                if p_idx is not None:
                    self.ingredient_mask[p_idx] = True

        # v0.47.0: DC order suppression counter (Fix 1 + Fix 5 diagnostics)
        self._dc_order_suppression_count = 0

        # PERF: Pre-computed lookup caches for O(1) array access
        # Replaces 49M+ dict.get() calls with direct array indexing
        self._build_lookup_caches()

        # v0.83.0: Multi-source replenishment (secondary source probabilistic routing)
        enrichment = config.get("network_enrichment", {})
        self._secondary_order_fraction = float(
            enrichment.get("secondary_source_order_fraction", 0.0)
        )
        self._secondary_rng = np.random.default_rng(seed=777)

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

        PERF v0.86.0: Welford's online algorithm — O(1) update of running
        mean and variance.  Eliminates np.array() + np.std() per dirty link.
        The deque is kept for warm-start serialization compatibility.
        """
        t_idx = self.state.node_id_to_idx.get(target_id)
        s_idx = self.state.node_id_to_idx.get(source_id)

        if t_idx is None or s_idx is None:
            return

        link_key = (t_idx, s_idx)

        # Initialize deque for this link if needed (warm-start serialization)
        if link_key not in self._lt_history:
            self._lt_history[link_key] = deque(maxlen=self.lt_history_len)

        dq = self._lt_history[link_key]
        val = float(lead_time_days)

        # Windowed Welford: remove oldest sample if window full
        welford = self._lt_welford.get(link_key)
        if welford is None:
            welford = [0, 0.0, 0.0]  # [count, mean, M2]
            self._lt_welford[link_key] = welford

        if len(dq) == dq.maxlen:
            # Remove oldest value from running stats
            old_val = dq[0]
            n = welford[0]
            if n > 1:
                old_mean = welford[1]
                new_n = n - 1
                new_mean = (old_mean * n - old_val) / new_n
                welford[2] -= (old_val - old_mean) * (old_val - new_mean)
                welford[1] = new_mean
                welford[0] = new_n
            else:
                welford[0] = 0
                welford[1] = 0.0
                welford[2] = 0.0

        # Add new value (standard Welford update)
        dq.append(val)
        welford[0] += 1
        n = welford[0]
        delta = val - welford[1]
        welford[1] += delta / n
        delta2 = val - welford[1]
        welford[2] += delta * delta2

        # Update cache directly — no dirty set needed
        self._lt_mu_cache_sparse[link_key] = welford[1]
        if n >= 2:  # noqa: PLR2004
            # Safety floor: M2 can go slightly negative from fp drift
            m2 = max(0.0, welford[2])
            self._lt_sigma_cache_sparse[link_key] = float(
                np.sqrt(m2 / (n - 1))
            )
        else:
            self._lt_sigma_cache_sparse[link_key] = 0.0

    def _update_lt_cache(self) -> None:
        """No-op — stats are now maintained inline by record_lead_time().

        Kept for API compatibility (called by generate_orders).
        """
        pass

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
        - _target_source_idx: n_idx -> primary source node_idx (PERF v0.87.0)
        - _target_lead_time_arr: n_idx -> primary link lead time (PERF v0.87.0)
        - _target_has_secondary: n_idx -> has secondary source (PERF v0.87.0)
        - _secondary_source_idx_arr: n_idx -> secondary source node_idx (PERF v0.87.0)
        - _secondary_lead_time_arr: n_idx -> secondary lead time (PERF v0.87.0)
        - _target_channel_arr: n_idx -> channel.name string (PERF v0.87.0)
        """
        n_products = self.state.n_products
        n_nodes = self.state.n_nodes

        # Product caches
        self._product_id_arr: np.ndarray = np.empty(n_products, dtype=object)
        self._product_category_arr: np.ndarray = np.empty(n_products, dtype=object)
        self._price_arr: np.ndarray = np.zeros(n_products, dtype=np.float64)

        for p_idx in range(n_products):
            p_id = self.state.product_idx_to_id[p_idx]
            self._product_id_arr[p_idx] = p_id
            product = self.world.products.get(p_id)
            if product and product.category:
                self._product_category_arr[p_idx] = product.category.name
                self._price_arr[p_idx] = product.price_per_case
            else:
                self._product_category_arr[p_idx] = ""

        # Node caches
        self._node_id_arr: np.ndarray = np.empty(n_nodes, dtype=object)

        for n_idx in range(n_nodes):
            self._node_id_arr[n_idx] = self.state.node_idx_to_id[n_idx]

        # PERF v0.87.0: Pre-computed source/lead-time/channel arrays for
        # vectorized _create_order_batch (replaces per-target dict lookups)
        _nid_to_idx = self.state.node_id_to_idx
        self._target_source_idx = np.full(n_nodes, -1, dtype=np.int32)
        self._target_lead_time_arr = np.full(n_nodes, 3.0, dtype=np.float32)
        self._target_has_secondary = np.zeros(n_nodes, dtype=bool)
        self._secondary_source_idx_arr = np.full(n_nodes, -1, dtype=np.int32)
        self._secondary_lead_time_arr = np.full(n_nodes, 3.0, dtype=np.float32)
        self._target_channel_arr: np.ndarray = np.empty(n_nodes, dtype=object)

        for n_idx in range(n_nodes):
            n_id = self._node_id_arr[n_idx]
            # Primary source
            src_id = self.store_supplier_map.get(n_id)
            if src_id:
                self._target_source_idx[n_idx] = _nid_to_idx.get(src_id, -1)
                self._target_lead_time_arr[n_idx] = self._link_lead_time.get(
                    n_id, 3.0
                )
            # Secondary source
            sec_id = self._secondary_source_map.get(n_id)
            if sec_id:
                self._target_has_secondary[n_idx] = True
                self._secondary_source_idx_arr[n_idx] = _nid_to_idx.get(
                    sec_id, -1
                )
                self._secondary_lead_time_arr[n_idx] = (
                    self._secondary_lead_time.get(n_id, 3.0)
                )
            # Channel name
            node = self.world.nodes.get(n_id)
            if node and node.channel:
                self._target_channel_arr[n_idx] = node.channel.name
            else:
                self._target_channel_arr[n_idx] = ""

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
        # PERF v0.69.3: Invalidate demand std cache
        self._demand_std_cache = None

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

        PERF v0.69.3: Day-level cache invalidated by record_demand().
        Removed np.array() wrapper (np.std already returns ndarray).

        Returns:
            Shape [n_nodes, n_products] - Standard deviation of demand
        """
        # Return cached value if available (invalidated in record_demand)
        if self._demand_std_cache is not None:
            return self._demand_std_cache

        # Need minimum history to calculate meaningful variance
        min_history = self._min_history_days
        if self.history_idx < min_history:
            # Fallback for cold start: assume zero std until we have history
            result = np.zeros((self.state.n_nodes, self.state.n_products))
            self._demand_std_cache = result
            return result

        n_samples = min(self.history_idx, self.variance_lookback)
        # Calculate std along time axis (axis 0)
        # Use ddof=1 for sample standard deviation
        # PERF v0.69.3: np.std already returns ndarray — removed np.array() wrapper
        result = np.std(
            self.demand_history_buffer[:n_samples],
            axis=0,
            ddof=1,
        )
        self._demand_std_cache = result
        return result

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

        # PERF v0.69.3: Pre-allocate arrays for scatter-add, use cached indices
        # Estimate upper bound: ~30 lines per order on average
        n_est = len(orders) * 30
        src_arr = np.empty(n_est, dtype=np.intp)
        prod_arr = np.empty(n_est, dtype=np.intp)
        qty_arr = np.empty(n_est, dtype=np.float64)
        n = 0

        for order in orders:
            source_idx = (
                order.source_idx if order.source_idx >= 0
                else self.state.node_id_to_idx.get(order.source_id)
            )
            if source_idx is None:
                continue

            for line in order.lines:
                p_idx = (
                    line.product_idx if line.product_idx >= 0
                    else self.state.product_id_to_idx.get(line.product_id)
                )
                if p_idx is not None and p_idx >= 0:
                    if n >= n_est:
                        # Rare: grow arrays
                        n_est *= 2
                        src_arr = np.resize(src_arr, n_est)
                        prod_arr = np.resize(prod_arr, n_est)
                        qty_arr = np.resize(qty_arr, n_est)
                    src_arr[n] = source_idx
                    prod_arr[n] = p_idx
                    qty_arr[n] = line.quantity
                    n += 1

        # PERF: Single scatter-add operation instead of N individual updates
        if n > 0:
            np.add.at(
                self.inflow_history[self._inflow_ptr],
                (src_arr[:n], prod_arr[:n]),
                qty_arr[:n],
            )

        # Advance pointer for circular buffer
        self._inflow_ptr = (self._inflow_ptr + 1) % self._history_window

    def record_inflow_batch(self, batch: OrderBatch) -> None:
        """Record orders received BY each node — vectorized OrderBatch path.

        PERF v0.86.0: Single np.add.at() call replacing nested Order→OrderLine loop.
        """
        hw = self._history_window
        if self.inflow_history is None:
            self.inflow_history = np.zeros(
                (hw, self.state.n_nodes, self.state.n_products), dtype=np.float64
            )

        self.inflow_history[self._inflow_ptr] = 0
        np.add.at(
            self.inflow_history[self._inflow_ptr],
            (batch.source_idx, batch.product_idx),
            batch.quantity,
        )
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
        """Builds a lookup map for Store -> Source ID (primary = shortest link).

        Also builds secondary source maps for multi-source DCs.
        """
        # Group links by target, sort by distance to identify primary (shortest)
        target_links: dict[str, list[Link]] = defaultdict(list)
        for link in self.world.links.values():
            target_links[link.target_id].append(link)

        mapping: dict[str, str] = {}
        self._link_lead_time: dict[str, float] = {}
        self._secondary_source_map: dict[str, str] = {}
        self._secondary_lead_time: dict[str, float] = {}

        for target_id, links in target_links.items():
            links.sort(key=lambda lk: lk.distance_km)
            # Primary = shortest distance
            mapping[target_id] = links[0].source_id
            self._link_lead_time[target_id] = links[0].lead_time_days
            # Secondary = second shortest (if exists)
            if len(links) >= 2:  # noqa: PLR2004
                self._secondary_source_map[target_id] = links[1].source_id
                self._secondary_lead_time[target_id] = links[1].lead_time_days

        return mapping

    def generate_orders(
        self, day: int, demand_signal: np.ndarray
    ) -> OrderBatch | None:
        """
        Generates replenishment orders for Retail Stores and downstream DCs.
        Uses exponential smoothing on the demand signal to dampen bullwhip.
        Includes order staggering to prevent synchronized ordering waves.

        v0.38.0: Now includes unmet demand in signal calculation and decays
        unmet demand after orders are placed to prevent accumulation.

        v0.86.0 PERF: Returns OrderBatch (parallel numpy arrays) instead of
        list[Order].  Eliminates ~50K OrderLine + ~6.5K Order object creation.
        """
        self.current_day = day
        # 1. Identify Target Nodes
        target_indices, target_ids = self._identify_target_nodes(day)
        if not target_indices:
            return None

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

        # 7. Create OrderBatch (vectorized — no Order/OrderLine objects)
        order_batch = self._create_order_batch(
            day, target_indices, target_ids, batched_qty, avg_demand, on_hand_inv
        )

        # v0.38.0: Decay unmet demand after orders placed to prevent accumulation.
        params = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("replenishment", {})
        )
        unmet_decay = float(params.get("unmet_demand_decay", 0.85))
        self.state.decay_unmet_demand(decay_factor=unmet_decay)

        return order_batch

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
                or (
                    node.type == NodeType.DC
                    and "RDC" not in node.id
                    and not self._drp_suppresses_dc_pull
                )
            ):
                idx = self.state.node_id_to_idx.get(n_id)
                if idx is None:
                    continue

                # v0.69.2 PERF: Check stagger day BEFORE computing emergency DOS.
                # Emergency DOS computation is expensive (numpy ops per store).
                # Only compute it for stores whose stagger day doesn't match today.
                if node.type == NodeType.STORE:
                    order_day = hash(n_id) % order_cycle_days
                    if day % order_cycle_days != order_day:
                        # Not this store's scheduled day — check emergency bypass
                        node_inv = self.state.actual_inventory[idx, :]
                        if self.smoothed_demand is not None:
                            node_demand = self.smoothed_demand[idx, :]
                        else:
                            node_demand = np.ones(self.state.n_products) * 0.1

                        with np.errstate(divide="ignore", invalid="ignore"):
                            node_dos = np.where(
                                node_demand > self.min_demand_floor,
                                node_inv / node_demand,
                                np.inf,
                            )
                        node_dos[self.ingredient_mask] = np.inf
                        min_dos = np.min(node_dos)
                        if min_dos >= emergency_dos_threshold:
                            continue  # Not emergency, skip this store

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
            forecast_horizon = self._forecast_horizon
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

        for i, t_idx in enumerate(target_indices):
            if t_idx in self._customer_dc_indices:
                # v0.52.0: Use outflow (shipments to stores) as DC demand signal.
                # Previously used inflow (orders received), which is the echelon
                # demand signal and caused ordering-vs-shipping asymmetry.
                # No hard throughput floor here — echelon logic handles ordering.
                # Anti-windup gating provides conditional protection only.
                outflow_signal = outflow_demand[t_idx, :]
                expected = self._expected_throughput.get(t_idx, outflow_signal)
                base_signal = outflow_signal.copy()

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
        dc_actual_outflow = self.get_outflow_demand()[dc_actual_indices, :].copy()

        # Apply inventory-gated throughput floor (anti-windup pattern).
        # Floor engages when DC inventory is LOW (cold start / death spiral).
        # Floor disengages when inventory >= target (allows drawdown).
        # Without gating, the floor causes persistent over-ordering during
        # seasonal troughs when actual outflow < expected.
        throughput_floor_pct = float(params.get("throughput_floor_pct", 0.7))
        dc_buffer_days = float(params.get("dc_buffer_days", 7.0))
        dc_local_inv = self.state.actual_inventory[dc_actual_indices, :]

        dc_outflow = dc_actual_outflow.copy()
        for i, dc_idx in enumerate(dc_actual_indices):
            expected = self._expected_throughput.get(int(dc_idx))
            if expected is not None:
                # Gating: floor_weight = clip(
                #   (target_dos - local_dos) / target_dos, 0, 1)
                # DOS=0 → weight=1.0; DOS=target → weight=0.0
                local_demand = np.maximum(expected, 0.1)
                local_dos = dc_local_inv[i, :] / local_demand
                floor_weight = np.clip(
                    (dc_buffer_days - local_dos) / np.maximum(dc_buffer_days, 1.0),
                    0.0,
                    1.0,
                )
                floor = expected * throughput_floor_pct * floor_weight
                dc_outflow[i, :] = np.maximum(dc_actual_outflow[i, :], floor)

        # 2. Local buffer target based on ACTUAL outflow
        dc_local_target = dc_actual_outflow * dc_buffer_days

        # Floor target at gated throughput level for cold-start protection
        for i, dc_idx in enumerate(dc_actual_indices):
            expected = self._expected_throughput.get(int(dc_idx))
            if expected is not None:
                local_demand = np.maximum(expected, 0.1)
                local_dos = dc_local_inv[i, :] / local_demand
                floor_weight = np.clip(
                    (dc_buffer_days - local_dos) / np.maximum(dc_buffer_days, 1.0),
                    0.0,
                    1.0,
                )
                min_target = (
                    expected * throughput_floor_pct * floor_weight * dc_buffer_days
                )
                dc_local_target[i, :] = np.maximum(
                    dc_local_target[i, :], min_target
                )

        # 3. Local inventory position (DC only, NOT echelon)
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

        # v0.54.1: Physics-derived DC DOS caps from dc_buffer_days x ABC multiplier.
        # With buffer=7: A~10.5, B=14, C=17.5. Self-adjusting if buffer changes.
        dc_dos_cap_base = dc_buffer_days
        dc_dos_cap_a = dc_dos_cap_base * float(params.get("dc_dos_cap_mult_a", 1.5))
        dc_dos_cap_b = dc_dos_cap_base * float(params.get("dc_dos_cap_mult_b", 2.0))
        dc_dos_cap_c = dc_dos_cap_base * float(params.get("dc_dos_cap_mult_c", 2.5))

        # Build per-product DOS cap vector based on ABC classification
        dc_dos_cap_vec = np.full(self.state.n_products, dc_dos_cap_c)
        # Use z_scores_vec as ABC proxy (A=2.33, B=1.65, C varies)
        dc_dos_cap_vec[self.z_scores_vec >= self.z_score_A] = dc_dos_cap_a
        dc_dos_cap_vec[
            (self.z_scores_vec >= self.z_score_B) & (self.z_scores_vec < self.z_score_A)
        ] = dc_dos_cap_b

        # Calculate LOCAL DC DOS using ACTUAL outflow (NOT floored).
        # Using floored outflow deflates DOS, preventing the cap from firing
        # when actual outflow << expected throughput.
        dc_local_demand = np.maximum(dc_actual_outflow, 0.1)
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

    def _create_order_batch(
        self,
        day: int,
        target_indices: list[int],
        target_ids: list[str],
        batched_qty: np.ndarray,
        avg_demand: np.ndarray,
        on_hand_inv: np.ndarray,
    ) -> OrderBatch | None:
        """Build an OrderBatch from the non-zero entries of *batched_qty*.

        PERF v0.86.0: Replaces _create_order_objects.  Zero Python object
        creation for the hot path — all data lives in parallel numpy arrays.
        """
        rows, cols = np.nonzero(batched_qty)
        if len(rows) == 0:
            return None

        # --- 1. Build per-line arrays directly from nonzero results -----------
        product_idx_arr = cols.astype(np.int32)

        # PERF v0.86.0: Vectorized fancy-indexing replaces Python loop
        ti_arr = np.array(target_indices, dtype=np.int32)
        target_idx_arr = ti_arr[rows]
        quantity_arr = batched_qty[rows, cols].astype(np.float64)
        unit_price_arr = self._price_arr[cols]

        # Filter out zero/negative
        alive = quantity_arr > 0
        if not np.any(alive):
            return None
        if not np.all(alive):
            target_idx_arr = target_idx_arr[alive]
            product_idx_arr = product_idx_arr[alive]
            quantity_arr = quantity_arr[alive]
            unit_price_arr = unit_price_arr[alive]
            rows = rows[alive]
            cols = cols[alive]

        # --- 2. Per-target metadata (order type, priority, source, promo) -----
        # PERF v0.87.0: Fully vectorized — replaces per-target Python loop.
        # Unique targets in this batch (preserves order)
        unique_targets, inverse = np.unique(target_idx_arr, return_inverse=True)
        n_targets = len(unique_targets)

        # -- 2a. Vectorized source lookup (primary / secondary) ----------------
        tgt_source_idx = self._target_source_idx[unique_targets].copy()
        tgt_lead_time = self._target_lead_time_arr[unique_targets].copy()

        if self._secondary_order_fraction > 0:
            has_sec = self._target_has_secondary[unique_targets]
            rng_vals = self._secondary_rng.random(n_targets)
            use_sec = has_sec & (rng_vals < self._secondary_order_fraction)
            if np.any(use_sec):
                tgt_source_idx[use_sec] = self._secondary_source_idx_arr[
                    unique_targets[use_sec]
                ]
                tgt_lead_time[use_sec] = self._secondary_lead_time_arr[
                    unique_targets[use_sec]
                ]

        # Build source_id strings and order IDs (unavoidable string ops)
        _nid_lookup = self.state.node_idx_to_id
        _node_id_arr = self._node_id_arr
        tgt_source_id: list[str] = [
            _nid_lookup[int(si)] if si >= 0 else ""
            for si in tgt_source_idx
        ]
        tgt_order_id: list[str] = [
            f"ORD-{day}-{_node_id_arr[int(unique_targets[gi])]}-{gi + 1}"
            for gi in range(n_targets)
        ]
        tgt_requested_date = day + tgt_lead_time.astype(np.int32)

        # -- 2b. Vectorized rush detection via np.minimum.reduceat -------------
        with np.errstate(divide="ignore", invalid="ignore"):
            dos_per_line = np.where(
                avg_demand[rows, cols] > 0,
                on_hand_inv[rows, cols] / avg_demand[rows, cols],
                1e30,  # large sentinel (reduceat-safe, no inf)
            )

        # Sort lines by target group for reduceat
        sort_order = np.argsort(inverse, kind="mergesort")
        sorted_dos = dos_per_line[sort_order]
        sorted_inverse = inverse[sort_order]
        _, group_starts = np.unique(sorted_inverse, return_index=True)

        min_dos_per_target = np.minimum.reduceat(sorted_dos, group_starts)

        # -- 2c. Vectorized promo detection ------------------------------------
        week = (day // 7) + 1
        active_promos: list[dict[str, Any]] = []
        promotions = self.config.get("promotions", [])
        for p in promotions:
            if p["start_week"] <= week <= p["end_week"]:
                active_promos.append(p)

        is_promo = np.zeros(n_targets, dtype=bool)
        tgt_promo_ids: dict[int, str] = {}

        if active_promos:
            tgt_channels = self._target_channel_arr[unique_targets]
            sorted_prod_idx = product_idx_arr[sort_order]
            sorted_cats = self._product_category_arr[sorted_prod_idx]

            for promo in active_promos:
                remaining = ~is_promo
                if not np.any(remaining):
                    break
                affected_cats = promo.get("affected_categories", ["all"])
                affected_chans = promo.get("affected_channels")

                # Channel match (vectorized)
                if affected_chans:
                    chan_match = np.array(
                        [cn in affected_chans for cn in tgt_channels],
                        dtype=bool,
                    )
                else:
                    chan_match = np.ones(n_targets, dtype=bool)

                candidates = chan_match & remaining
                if not np.any(candidates):
                    continue

                if "all" in affected_cats:
                    new_promo = candidates
                else:
                    # Per-line category match → reduceat to per-target "any"
                    line_cat_match = np.array(
                        [c in affected_cats for c in sorted_cats],
                        dtype=np.int8,
                    )
                    any_cat = np.maximum.reduceat(
                        line_cat_match, group_starts
                    ).astype(bool)
                    new_promo = candidates & any_cat

                if np.any(new_promo):
                    is_promo |= new_promo
                    code = promo["code"]
                    for gi in np.where(new_promo)[0]:
                        tgt_promo_ids[int(unique_targets[gi])] = code

        # -- 2d. Vectorized order type + priority assignment -------------------
        is_rush = (
            ~is_promo
            & (min_dos_per_target < self.rush_threshold_days)
        )

        tgt_order_type = np.full(n_targets, 3, dtype=np.int8)   # STANDARD
        tgt_order_type[is_promo] = 2   # PROMOTIONAL
        tgt_order_type[is_rush] = 1    # RUSH

        tgt_priority = np.full(
            n_targets, int(OrderPriority.STANDARD), dtype=np.int8
        )
        tgt_priority[is_promo] = int(OrderPriority.HIGH)
        tgt_priority[is_rush] = int(OrderPriority.RUSH)

        # --- 3. Broadcast per-target metadata to per-line arrays ---------------
        source_idx_arr = tgt_source_idx[inverse]
        order_type_arr = tgt_order_type[inverse]
        priority_arr = tgt_priority[inverse]
        requested_date_arr = tgt_requested_date[inverse]
        inverse_arr = inverse  # line → group_idx mapping for string ID lookup

        # Filter lines whose target has no source
        has_source = source_idx_arr >= 0
        if not np.all(has_source):
            target_idx_arr = target_idx_arr[has_source]
            product_idx_arr = product_idx_arr[has_source]
            quantity_arr = quantity_arr[has_source]
            unit_price_arr = unit_price_arr[has_source]
            source_idx_arr = source_idx_arr[has_source]
            order_type_arr = order_type_arr[has_source]
            priority_arr = priority_arr[has_source]
            requested_date_arr = requested_date_arr[has_source]
            inverse_arr = inverse_arr[has_source]

        if len(quantity_arr) == 0:
            return None

        return OrderBatch(
            source_idx=source_idx_arr,
            target_idx=target_idx_arr,
            product_idx=product_idx_arr,
            quantity=quantity_arr,
            unit_price=unit_price_arr,
            order_type=order_type_arr,
            priority=priority_arr,
            creation_day=day,
            requested_date=requested_date_arr,
            promo_ids=tgt_promo_ids,
            _tgt_source_ids=tgt_source_id,
            _tgt_order_ids=tgt_order_id,
            _tgt_inverse=inverse_arr,
        )

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
