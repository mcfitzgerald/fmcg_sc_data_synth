"""
MRP Engine: Translates Distribution Requirements (DRP) into Production Orders.

[Task 5.1] [Intent: 4. Architecture - Phase 2: The Time Loop]

This module monitors RDC inventory levels and generates Production Orders
for Plants when stock falls below reorder points.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prism_sim.network.core import (
    NodeType,
    Order,
    OrderLine,
    ProductionOrder,
    ProductionOrderStatus,
    Shipment,
)
from prism_sim.product.core import ProductCategory
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

# Diagnostic logger for MRP signal tracing
mrp_logger = logging.getLogger("prism_sim.mrp.diagnostics")


@dataclass
class MRPDiagnostics:
    """Diagnostic snapshot of MRP state for debugging starvation issues."""

    day: int = 0
    # Demand signals
    expected_daily_demand: float = 0.0
    actual_demand_signal: float = 0.0
    pos_demand_total: float = 0.0
    demand_signal_ratio: float = 0.0  # actual / expected
    # Production
    production_orders_generated: float = 0.0
    production_floor_applied: bool = False
    smoothing_cap_applied: bool = False
    rate_based_mode: bool = False
    # Inventory position
    total_rdc_inventory: float = 0.0
    total_in_transit: float = 0.0
    total_in_production: float = 0.0
    avg_dos: float = 0.0
    # Ingredients
    ingredient_orders_generated: float = 0.0
    total_backlog: float = 0.0
    backlog_vs_expected_ratio: float = 0.0
    # Detailed breakdown by product (top 5)
    top_products: list[dict[str, Any]] = field(default_factory=list)

    def log(self) -> None:
        """Log diagnostics at INFO level."""
        mrp_logger.info(
            f"Day {self.day}: "
            f"Demand[exp={self.expected_daily_demand:,.0f} "
            f"act={self.actual_demand_signal:,.0f} "
            f"ratio={self.demand_signal_ratio:.1%}] "
            f"Prod[gen={self.production_orders_generated:,.0f} "
            f"floor={self.production_floor_applied} "
            f"cap={self.smoothing_cap_applied} "
            f"rate={self.rate_based_mode}] "
            f"Inv[rdc={self.total_rdc_inventory:,.0f} "
            f"transit={self.total_in_transit:,.0f} "
            f"dos={self.avg_dos:.1f}d] "
            f"Ing[orders={self.ingredient_orders_generated:,.0f} "
            f"backlog_ratio={self.backlog_vs_expected_ratio:.1%}]"
        )


class MRPEngine:
    """
    Material Requirements Planning engine that generates Production Orders.

    Monitors RDC inventory levels (DRP) and creates Production Orders
    for Plants to maintain target stock levels.
    """

    def __init__(
        self,
        world: World,
        state: StateManager,
        config: dict[str, Any],
        base_demand_matrix: np.ndarray | None = None,
    ) -> None:
        self.world = world
        self.state = state
        self.config = config
        self.base_demand_matrix = base_demand_matrix

        # Extract manufacturing config
        mrp_config = config.get("simulation_parameters", {}).get("manufacturing", {})
        self.target_days_supply = mrp_config.get("target_days_supply", 14.0)
        self.reorder_point_days = mrp_config.get("reorder_point_days", 7.0)
        self.min_production_qty = mrp_config.get("min_production_qty", 100.0)
        self.min_ingredient_moq = mrp_config.get("min_ingredient_moq", 100.0)
        self.production_lead_time = mrp_config.get("production_lead_time_days", 3)

        # MRP threshold parameters (previously hardcoded)
        mrp_thresholds = mrp_config.get("mrp_thresholds", {})
        self.demand_signal_collapse_pct = mrp_thresholds.get(
            "demand_signal_collapse_pct", 0.4
        )
        self.velocity_trend_threshold_pct = mrp_thresholds.get(
            "velocity_trend_threshold_pct", 0.6
        )
        self.production_floor_pct = mrp_thresholds.get("production_floor_pct", 0.3)
        self.min_production_cap_pct = mrp_thresholds.get("min_production_cap_pct", 0.5)

        # v0.19.9: Rate-based production parameters (Option C fix)
        # When enabled, production always runs at expected demand rate.
        # DOS only modulates ABOVE baseline (catch-up mode), never below.
        self.rate_based_production = mrp_thresholds.get(
            "rate_based_production", False
        )
        # Only throttle production if DOS exceeds this many days
        self.inventory_cap_dos = mrp_thresholds.get("inventory_cap_dos", 45.0)

        # Diagnostics storage
        self._diagnostics: MRPDiagnostics | None = None
        self._diagnostics_enabled = mrp_thresholds.get("diagnostics_enabled", True)

        # Pre-calculate Policy Vectors
        self._build_policy_vectors(mrp_config)

        # Cache RDC and Plant node indices
        self._rdc_ids: list[str] = []
        self._plant_ids: list[str] = []
        self._finished_product_ids: list[str] = []

        # Map of plant_id -> list of supported category names
        self.plant_capabilities: dict[str, list[str]] = {}
        self._load_plant_capabilities(mrp_config)

        self._cache_node_info()

        # v0.19.8: Cache max daily plant capacity for ingredient ordering
        # Must be after _cache_node_info() to have _plant_ids populated
        self._max_daily_capacity = self._calculate_max_daily_capacity()

        # C.1 FIX: Cache expected daily demand for fallback (prevents death spiral)
        self._build_expected_demand_vector()

        # v0.22.0: Calculate sustainable demand (capacity-aware)
        # DEATH SPIRAL FIX: When expected demand exceeds plant capacity,
        # MRP floors must be scaled down to match what plants can actually produce.
        # Otherwise, production orders pile up, changeovers increase, and
        # actual throughput drops - creating a death spiral.
        self._build_sustainable_demand_vector()

        # Demand history for moving average [Products]
        # v0.15.6: Extended from 7 to 14 days for smoother signal
        # v0.19.9: Warm start with expected demand to prevent Day 1 dip
        self.demand_history = np.tile(
            self.expected_daily_demand, (14, 1)
        ).astype(np.float64)
        self._history_ptr = 0

        # C.5 FIX: Production order history for smoothing (reduces volatility)
        # v0.15.6: Extended from 7 to 14 days
        # v0.19.9: Warm start with total expected production
        total_expected_production = np.sum(self.expected_daily_demand)
        self.production_order_history = np.full(
            14, total_expected_production, dtype=np.float64
        )
        self._prod_hist_ptr = 0

        # v0.15.6: Velocity tracking - detect declining demand trends
        self._week1_demand_sum = 0.0
        self._week2_demand_sum = 0.0

        # Production Order counter for unique IDs
        self._po_counter = 0

        # Phase 2: ABC Prioritization
        # Classify products and build ROP multiplier vector
        self.abc_rop_multiplier = np.ones(self.state.n_products, dtype=np.float64)
        # v0.19.10: Track ABC class per product for production differentiation
        # 0 = A-item, 1 = B-item, 2 = C-item
        self.abc_class = np.full(self.state.n_products, 2, dtype=np.int8)  # Default C
        self._classify_products_abc()

        # v0.19.10: ABC Production Strategy parameters
        # Differentiate production by ABC class to fix product mix mismatch
        self.abc_production_enabled = mrp_thresholds.get(
            "abc_production_enabled", True
        )
        # A-items: buffer above max(expected, actual) to never stockout
        self.a_production_buffer = mrp_thresholds.get("a_production_buffer", 1.1)
        # C-items: produce at reduced rate to minimize SLOB
        self.c_production_factor = mrp_thresholds.get("c_production_factor", 0.6)
        # C-items: when using actual demand, apply this factor
        self.c_demand_factor = mrp_thresholds.get("c_demand_factor", 0.8)

    def _build_policy_vectors(self, mrp_config: dict[str, Any]) -> None:
        """Pre-calculate ROP and Target vectors for all products."""
        policies = mrp_config.get("inventory_policies", {})
        default_rop = mrp_config.get("reorder_point_days", 14.0)
        default_target = mrp_config.get("target_days_supply", 28.0)

        n_products = self.state.n_products
        self.rop_vector = np.full(n_products, default_rop, dtype=np.float64)
        self.target_vector = np.full(n_products, default_target, dtype=np.float64)

        spof_id = mrp_config.get("spof", {}).get("ingredient_id", "")

        for p_id, p in self.world.products.items():
            p_idx = self.state.product_id_to_idx.get(p_id)
            if p_idx is None:
                continue

            # Determine Policy Key
            key = "DEFAULT"
            if p_id == spof_id:
                key = "SPOF"
            elif "ACT-CHEM" in p_id:
                key = "ACTIVE_CHEM"
            elif "PKG" in p_id:
                key = "PACKAGING"
            elif p.category == ProductCategory.INGREDIENT:
                key = "INGREDIENT"

            policy = policies.get(key, policies.get("DEFAULT", {}))

            self.rop_vector[p_idx] = policy.get("reorder_point_days", default_rop)
            self.target_vector[p_idx] = policy.get("target_days_supply", default_target)

    def _build_expected_demand_vector(self) -> None:
        """
        C.1 FIX: Build expected daily demand vector.

        v0.19.6: Use base_demand_matrix (Zipf-aware) if available, otherwise
        fallback to config-based estimation.
        """
        if self.base_demand_matrix is not None:
            # Sum demand across all nodes for each product
            # This captures Zipfian distribution correctly
            self.expected_daily_demand = np.sum(self.base_demand_matrix, axis=0)
            return

        # Fallback to legacy config-based logic (Zipf-blind)
        demand_config = self.config.get("simulation_parameters", {}).get("demand", {})
        cat_profiles = demand_config.get("category_profiles", {})

        # Count stores (nodes that consume - typically stores/DCs)
        n_stores = sum(
            1 for node in self.world.nodes.values()
            if node.type == NodeType.STORE
        )
        # Fallback if no stores found
        if n_stores == 0:
            mfg_config = self.config.get(
                "simulation_parameters", {}
            ).get("manufacturing", {})
            n_stores = int(mfg_config.get("default_store_count", 100))

        self.expected_daily_demand = np.zeros(self.state.n_products, dtype=np.float64)

        for p_id, product in self.world.products.items():
            p_idx = self.state.product_id_to_idx.get(p_id)
            if p_idx is None:
                continue

            # Skip ingredients - they don't have consumer demand
            if product.category == ProductCategory.INGREDIENT:
                continue

            # Get category profile
            cat_key = product.category.name  # e.g., "ORAL_CARE"
            profile = cat_profiles.get(cat_key, {})
            base_demand = profile.get("base_daily_demand", 7.0)

            # Expected demand = base per store * number of stores
            self.expected_daily_demand[p_idx] = base_demand * n_stores

    def _build_sustainable_demand_vector(self) -> None:
        """
        v0.22.0: Calculate sustainable demand that respects plant capacity.

        DEATH SPIRAL ROOT CAUSE:
        When expected_daily_demand exceeds plant capacity, MRP floors
        (70-90% of expected) create impossible production targets.
        Example: 21M expected demand × 0.8 floor = 16.8M requested,
        but plants can only produce 8-9M. The backlog accumulates,
        causing more changeovers and less actual production time.

        FIX: Scale expected demand proportionally when it exceeds capacity.
        This ensures MRP never requests more than plants can deliver.
        """
        total_expected = np.sum(self.expected_daily_demand)

        if total_expected > 0 and self._max_daily_capacity > 0:
            capacity_ratio = self._max_daily_capacity / total_expected

            if capacity_ratio < 1.0:
                # Capacity is less than expected demand - scale down
                self.sustainable_daily_demand = (
                    self.expected_daily_demand * capacity_ratio
                )
                print(
                    f"MRPEngine: Capacity constraint detected!\n"
                    f"  Expected demand: {total_expected:,.0f} cases/day\n"
                    f"  Plant capacity:  {self._max_daily_capacity:,.0f} cases/day\n"
                    f"  Capacity ratio:  {capacity_ratio:.1%}\n"
                    f"  Sustainable demand scaled to: "
                    f"{np.sum(self.sustainable_daily_demand):,.0f} cases/day"
                )
            else:
                # Capacity exceeds demand - use expected as-is
                self.sustainable_daily_demand = self.expected_daily_demand.copy()
        else:
            # Fallback
            self.sustainable_daily_demand = self.expected_daily_demand.copy()

    def _classify_products_abc(self) -> None:
        """
        Classify products into A/B/C categories within each category.

        Phase 2 ABC Prioritization:
        - Classify products separately within each category (ORAL, WASH, HOME)
          to ensure every category has A/B/C items.
        - A-items (Top 80% volume): 1.2x ROP multiplier
        - B-items (Next 15% volume): 1.0x ROP multiplier
        - C-items (Bottom 5% volume): 0.8x ROP multiplier
        """
        # Get abc config or use defaults
        abc_config = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("abc_prioritization", {})
        )
        enabled = abc_config.get("enabled", True)

        if not enabled:
            return

        # Multipliers
        mult_a = abc_config.get("a_rop_multiplier", 1.2)
        mult_b = abc_config.get("b_rop_multiplier", 1.0)
        mult_c = abc_config.get("c_rop_multiplier", 0.8)

        # Thresholds
        thresh_a = abc_config.get("a_threshold_pct", 0.80)
        thresh_b = abc_config.get("b_threshold_pct", 0.95)

        # Iterate by category to ensure per-category ABC
        # This prevents high-volume categories (Oral Care) from dominating A-class
        categories = set(p.category for p in self.world.products.values())

        total_a, total_b, total_c = 0, 0, 0

        for category in categories:
            if category == ProductCategory.INGREDIENT:
                continue

            # Find all products in this category
            cat_p_indices = []
            cat_demands = []

            for p_id, product in self.world.products.items():
                if product.category == category:
                    p_idx = self.state.product_id_to_idx.get(p_id)
                    if p_idx is not None:
                        cat_p_indices.append(p_idx)
                        cat_demands.append(self.expected_daily_demand[p_idx])

            if not cat_p_indices:
                continue

            cat_p_indices = np.array(cat_p_indices)
            cat_demands = np.array(cat_demands)

            cat_total_volume = np.sum(cat_demands)
            if cat_total_volume == 0:
                continue

            # Sort within category
            sorted_local_indices = np.argsort(cat_demands)[::-1]
            sorted_global_indices = cat_p_indices[sorted_local_indices]

            cumulative_volume = np.cumsum(cat_demands[sorted_local_indices])

            # Determine cutoffs
            idx_a = np.searchsorted(
                cumulative_volume, cat_total_volume * thresh_a, side='right'
            )
            idx_b = np.searchsorted(
                cumulative_volume, cat_total_volume * thresh_b, side='right'
            )

            # Apply multipliers and track class
            # A-items
            a_indices = sorted_global_indices[:idx_a]
            self.abc_rop_multiplier[a_indices] = mult_a
            self.abc_class[a_indices] = 0
            total_a += len(a_indices)

            # B-items
            b_indices = sorted_global_indices[idx_a:idx_b]
            self.abc_rop_multiplier[b_indices] = mult_b
            self.abc_class[b_indices] = 1
            total_b += len(b_indices)

            # C-items
            c_indices = sorted_global_indices[idx_b:]
            self.abc_rop_multiplier[c_indices] = mult_c
            self.abc_class[c_indices] = 2
            total_c += len(c_indices)

        # Log Alignment Verification
        print(
            "MRPEngine: Initialized Category-Level ABC Classification"
        )
        print(f"  A-Items: {total_a} SKUs (Multiplier {mult_a}x)")
        print(f"  B-Items: {total_b} SKUs (Multiplier {mult_b}x)")
        print(f"  C-Items: {total_c} SKUs (Multiplier {mult_c}x)")

    def _calculate_max_daily_capacity(self) -> float:
        """
        Calculate maximum daily production capacity across all plants.

        v0.19.8: Used to cap ingredient ordering at plant capacity,
        breaking the feedback loop where low historical production
        led to under-ordering ingredients.

        v0.22.0: Now includes production_rate_multiplier to match
        actual production capacity in TransformEngine.

        Returns:
            Total cases/day the network can produce at max capacity.
        """
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        plant_params = mfg_config.get("plant_parameters", {})
        hours_per_day = mfg_config.get("production_hours_per_day", 24.0)
        global_efficiency = mfg_config.get("efficiency_factor", 0.85)
        global_downtime = mfg_config.get("unplanned_downtime_pct", 0.05)
        # v0.22.0: Include production_rate_multiplier (simulates multiple lines)
        rate_multiplier = mfg_config.get("production_rate_multiplier", 1.0)

        total_capacity = 0.0

        for plant_id in self._plant_ids:
            # Get plant-specific config
            p_config = plant_params.get(plant_id, {})
            efficiency = p_config.get("efficiency_factor", global_efficiency)
            downtime = p_config.get("unplanned_downtime_pct", global_downtime)

            # Effective hours = hours * (1 - downtime) * efficiency
            effective_hours = hours_per_day * (1.0 - downtime) * efficiency

            # Get average run rate for products this plant can make
            # Use the recipes to find run rates for supported categories
            supported_cats = p_config.get("supported_categories", [])
            run_rates = []

            for product_id, recipe in self.world.recipes.items():
                product = self.world.products.get(product_id)
                if product is None:
                    continue

                # If plant has restrictions, check category
                if supported_cats:
                    if product.category.name not in supported_cats:
                        continue

                run_rates.append(recipe.run_rate_cases_per_hour)

            # Use average run rate for this plant
            if run_rates:
                avg_run_rate = sum(run_rates) / len(run_rates)
                plant_capacity = effective_hours * avg_run_rate
                total_capacity += plant_capacity

        # v0.22.0: Apply rate multiplier (simulates multiple production lines)
        return total_capacity * rate_multiplier

    def _cache_node_info(self) -> None:
        """Cache RDC and Plant node IDs for efficient lookups.

        Note: Only manufacturer-controlled RDCs (RDC-*) are included,
        not customer DCs (RET-DC-*, DIST-DC-*, ECOM-*, etc.) which
        represent customer inventory, not manufacturer inventory position.
        """
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.DC:
                # Only include manufacturer RDCs, not customer DCs
                if node_id.startswith("RDC-"):
                    self._rdc_ids.append(node_id)
            elif node.type == NodeType.PLANT:
                self._plant_ids.append(node_id)

        # Cache finished product IDs (non-ingredients)
        for product_id, product in self.world.products.items():
            if product.category != ProductCategory.INGREDIENT:
                self._finished_product_ids.append(product_id)

    def _calculate_inventory_position(
        self,
        product_id: str,
        in_transit_qty: dict[str, float],
        in_production_qty: dict[str, float],
    ) -> float:
        """Calculate total Inventory Position for a product."""
        on_hand_inventory = 0.0
        for rdc_id in self._rdc_ids:
            on_hand_inventory += self.state.get_inventory(rdc_id, product_id)

        for plant_id in self._plant_ids:
            on_hand_inventory += self.state.get_inventory(plant_id, product_id)

        return (
            on_hand_inventory
            + in_transit_qty.get(product_id, 0.0)
            + in_production_qty.get(product_id, 0.0)
        )

    def generate_production_orders(  # noqa: PLR0912, PLR0915
        self,
        current_day: int,
        rdc_shipments: list[Shipment],  # FIX 3: Accept Shipments instead of Orders
        active_production_orders: list[ProductionOrder],
        pos_demand: np.ndarray | None = None,  # v0.19.1: POS demand floor
    ) -> list[ProductionOrder]:
        """
        Generate Production Orders based on RDC inventory and demand signals.

        v0.19.9: Added rate-based production mode (Option C).
        When enabled, production always runs at expected demand rate.
        DOS only modulates ABOVE baseline (catch-up mode), never below.

        v0.19.1: Added POS demand as a floor to prevent demand signal collapse.
        When the order-based demand signal declines (because downstream is starving),
        we use actual consumer demand (POS) to maintain production levels.
        """
        production_orders: list[ProductionOrder] = []

        # Initialize diagnostics
        diag = MRPDiagnostics(day=current_day) if self._diagnostics_enabled else None
        floor_applied = False
        cap_applied = False

        # v0.20.0: Use order-based signal as primary (prevents death spiral)
        # Shipment signal removed - orders (recorded via record_order_demand()) are
        # the primary demand signal. Shipments collapse when constrained, but orders
        # reflect true demand. The fallback logic below (lines 520-538) is kept as
        # a safety net in case order signal also collapses.
        # OLD: self._update_demand_history(current_day, rdc_shipments)

        # Calculate Moving Average Demand from order/shipment history
        avg_daily_demand_vec = np.mean(self.demand_history, axis=0)

        # v0.19.1: Use POS demand as floor for demand signal
        pos_demand_total = 0.0
        if pos_demand is not None:
            pos_demand_by_product = np.sum(pos_demand, axis=0)
            pos_demand_total = float(np.sum(pos_demand_by_product))
            avg_daily_demand_vec = np.maximum(
                avg_daily_demand_vec, pos_demand_by_product
            )

        # v0.15.6: Calculate demand velocity (week-over-week trend)
        week1_avg = np.mean(self.demand_history[:7], axis=0)
        week2_avg = np.mean(self.demand_history[7:], axis=0)
        self._week1_demand_sum = float(np.sum(week1_avg))
        self._week2_demand_sum = float(np.sum(week2_avg))

        # C.1 FIX: Fallback to prevent death spiral
        total_signal = np.sum(avg_daily_demand_vec)
        expected_total = np.sum(self.expected_daily_demand)

        use_fallback = False
        if expected_total > 0:
            if total_signal < expected_total * self.demand_signal_collapse_pct:
                use_fallback = True
            elif (
                self._week2_demand_sum > 0
                and self._week1_demand_sum
                < self._week2_demand_sum * self.velocity_trend_threshold_pct
            ):
                use_fallback = True

        if use_fallback:
            avg_daily_demand_vec = np.maximum(
                avg_daily_demand_vec, self.expected_daily_demand
            )

        # 2. Calculate In-Production qty per product
        lookahead_horizon = self.reorder_point_days
        in_production_qty: dict[str, float] = {}
        for po in active_production_orders:
            if po.status != ProductionOrderStatus.COMPLETE:
                if po.due_day <= current_day + lookahead_horizon:
                    remaining = po.quantity_cases - po.produced_quantity
                    in_production_qty[po.product_id] = (
                        in_production_qty.get(po.product_id, 0.0) + remaining
                    )

        # Calculate In-Transit qty per product (to RDCs)
        in_transit_qty: dict[str, float] = {}
        for shipment in self.state.active_shipments:
            if shipment.target_id in self._rdc_ids:
                for line in shipment.lines:
                    in_transit_qty[line.product_id] = (
                        in_transit_qty.get(line.product_id, 0.0) + line.quantity
                    )

        # Collect inventory diagnostics
        total_rdc_inv = 0.0
        total_in_transit = sum(in_transit_qty.values())
        total_in_prod = sum(in_production_qty.values())
        for rdc_id in self._rdc_ids:
            for product_id in self._finished_product_ids:
                total_rdc_inv += self.state.get_inventory(rdc_id, product_id)

        # ================================================================
        # v0.19.9: RATE-BASED PRODUCTION (Option C)
        # v0.19.11: POS-driven production (physically correct approach)
        # ================================================================
        if self.rate_based_production:
            # v0.19.11: Pass POS demand directly for closed-loop control
            pos_demand_vec = None
            if pos_demand is not None:
                pos_demand_vec = np.sum(pos_demand, axis=0)

            production_orders = self._generate_rate_based_orders(
                current_day,
                avg_daily_demand_vec,
                in_transit_qty,
                in_production_qty,
                total_rdc_inv,
                pos_demand_vec,
            )
            if diag:
                diag.rate_based_mode = True

        else:
            # ================================================================
            # LEGACY DOS-TRIGGERED PRODUCTION (with floors)
            # ================================================================
            for product_id in self._finished_product_ids:
                p_idx = self.state.product_id_to_idx.get(product_id)
                if p_idx is None:
                    continue

                inventory_position = self._calculate_inventory_position(
                    product_id, in_transit_qty, in_production_qty
                )
                avg_daily_demand = max(float(avg_daily_demand_vec[p_idx]), 1.0)

                if avg_daily_demand > 0:
                    dos_position = inventory_position / avg_daily_demand
                else:
                    dos_position = float("inf")

                # v0.19.3: Apply ABC multiplier to ROP
                effective_rop = self.rop_vector[p_idx] * self.abc_rop_multiplier[p_idx]

                if dos_position < effective_rop:
                    target_inventory = avg_daily_demand * self.target_vector[p_idx]
                    net_requirement = target_inventory - inventory_position

                    if net_requirement > 0:
                        demand_based_min = avg_daily_demand * 7.0
                        mfg_config = self.config.get(
                            "simulation_parameters", {}
                        ).get("manufacturing", {})
                        absolute_min = float(
                            mfg_config.get("min_batch_size_absolute", 1000.0)
                        )
                        order_qty = max(
                            net_requirement, demand_based_min, absolute_min
                        )

                        plant_id = self._select_plant(product_id)
                        po = ProductionOrder(
                            id=self._generate_po_id(current_day),
                            plant_id=plant_id,
                            product_id=product_id,
                            quantity_cases=order_qty,
                            creation_day=current_day,
                            due_day=current_day + self.production_lead_time,
                            status=ProductionOrderStatus.PLANNED,
                            planned_start_day=current_day + 1,
                        )
                        production_orders.append(po)

            # v0.15.6: Minimum production floor
            total_orders_today = sum(po.quantity_cases for po in production_orders)
            expected_production = np.sum(self.expected_daily_demand)
            min_production_floor = expected_production * self.production_floor_pct

            if total_orders_today < min_production_floor and expected_production > 0:
                floor_applied = True
                shortfall = min_production_floor - total_orders_today
                if production_orders:
                    scale_factor = min_production_floor / max(total_orders_today, 1.0)
                    for po in production_orders:
                        po.quantity_cases = po.quantity_cases * scale_factor
                    total_orders_today = min_production_floor
                else:
                    top_products = np.argsort(self.expected_daily_demand)[-10:][::-1]
                    qty_per_product = shortfall / len(top_products)
                    for p_idx in top_products:
                        if self.expected_daily_demand[p_idx] > 0:
                            product_id = self.state.product_idx_to_id[p_idx]
                            plant_id = self._select_plant(product_id)
                            po = ProductionOrder(
                                id=self._generate_po_id(current_day),
                                plant_id=plant_id,
                                product_id=product_id,
                                quantity_cases=qty_per_product,
                                creation_day=current_day,
                                due_day=current_day + self.production_lead_time,
                                status=ProductionOrderStatus.PLANNED,
                                planned_start_day=current_day + 1,
                            )
                            production_orders.append(po)
                    total_orders_today = sum(
                        po.quantity_cases for po in production_orders
                    )

        # C.5 FIX: Smooth production orders to reduce volatility
        avg_recent = np.mean(self.production_order_history)
        expected_production = np.sum(self.expected_daily_demand)
        smoothing_baseline = max(avg_recent, expected_production)

        total_orders_today = sum(po.quantity_cases for po in production_orders)
        if smoothing_baseline > 0 and total_orders_today > smoothing_baseline * 1.5:
            cap_applied = True
            scale_factor = (smoothing_baseline * 1.5) / total_orders_today
            for po in production_orders:
                po.quantity_cases = float(po.quantity_cases * scale_factor)

        # Update production history
        actual_total = sum(po.quantity_cases for po in production_orders)
        self.production_order_history[self._prod_hist_ptr] = actual_total
        self._prod_hist_ptr = (self._prod_hist_ptr + 1) % 14

        # Collect and log diagnostics
        if diag:
            diag.expected_daily_demand = expected_total
            diag.actual_demand_signal = total_signal
            diag.pos_demand_total = pos_demand_total
            diag.demand_signal_ratio = (
                total_signal / expected_total if expected_total > 0 else 0.0
            )
            diag.production_orders_generated = actual_total
            diag.production_floor_applied = floor_applied
            diag.smoothing_cap_applied = cap_applied
            diag.total_rdc_inventory = total_rdc_inv
            diag.total_in_transit = total_in_transit
            diag.total_in_production = total_in_prod
            diag.avg_dos = (
                total_rdc_inv / expected_total if expected_total > 0 else 0.0
            )
            self._diagnostics = diag
            diag.log()

        return production_orders

    def _generate_rate_based_orders(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        current_day: int,
        avg_daily_demand_vec: np.ndarray,
        in_transit_qty: dict[str, float],
        in_production_qty: dict[str, float],
        total_rdc_inv: float,
        pos_demand_vec: np.ndarray | None = None,
    ) -> list[ProductionOrder]:
        """
        v0.19.11: POS-driven production (physically correct approach).

        Key insight: Production should TRACK actual consumer demand (POS),
        not expected demand or derived signals. This creates a closed-loop
        system that self-corrects based on actual inventory and demand.

        Physics principle: At steady state, Production = Demand.
        ABC differentiation is in RESPONSE DYNAMICS, not baseline rates.

        Production logic:
        1. PRIMARY SIGNAL: Use POS demand (actual consumer sales)
        2. INVENTORY FEEDBACK: Adjust based on actual DOS
        3. ABC RESPONSE: Different catch-up/draw-down speeds per class
        """
        production_orders: list[ProductionOrder] = []

        for product_id in self._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue

            # Get expected demand as fallback
            expected_demand = float(self.expected_daily_demand[p_idx])
            if expected_demand <= 0:
                continue

            # ============================================================
            # v0.19.11: PRIMARY SIGNAL - Use POS demand (actual consumer sales)
            # ============================================================
            # POS demand is the TRUE demand signal - what consumers actually bought
            # This is more reliable than order-based signals which collapse
            if pos_demand_vec is not None:
                pos_demand = float(pos_demand_vec[p_idx])
            else:
                # Fallback to avg demand if POS not available
                pos_demand = float(avg_daily_demand_vec[p_idx])

            # Use POS demand directly as the primary signal
            # v0.22.0 DEATH SPIRAL FIX: Remove the 70% floor!
            # The floor was inflating demand for ALL SKUs, causing:
            # - Production orders for items that don't need them
            # - 500 orders/day → massive changeover overhead
            #
            # NEW LOGIC: Use actual POS demand. If stores are stocking out,
            # the DOS-based control will trigger catch-up production.
            # Only apply a floor for A-items to protect high-runners.
            abc = self.abc_class[p_idx]
            if abc == 0:  # A-item only: use floor to prevent stockouts
                sustainable_floor = float(self.sustainable_daily_demand[p_idx]) * 0.5
                actual_demand = max(pos_demand, sustainable_floor)
            else:
                # B and C items: use POS demand directly
                actual_demand = max(pos_demand, 1.0)  # Avoid division by zero

            # Calculate inventory position and DOS based on POS demand
            inventory_position = self._calculate_inventory_position(
                product_id, in_transit_qty, in_production_qty
            )

            # DOS calculated against actual POS demand (not derived signals)
            if actual_demand > 0:
                dos_position = inventory_position / actual_demand
            else:
                dos_position = float("inf")

            # abc already assigned above at line 826

            # ============================================================
            # v0.19.11: CLOSED-LOOP CONTROL
            # Production = Demand + Inventory Correction
            # ABC determines HOW FAST we correct, not baseline rate
            # ============================================================

            if abc == 0:  # A-item: Fast response, high service target
                # Target DOS for A-items: 14-21 days
                target_dos = 17.0
                if dos_position < 10:  # noqa: PLR2004
                    # Critical low - aggressive catch-up (130% of demand)
                    production_qty = actual_demand * 1.3
                elif dos_position < target_dos:
                    # Below target - moderate catch-up (115% of demand)
                    production_qty = actual_demand * 1.15
                elif dos_position < 30:  # noqa: PLR2004
                    # At/above target - maintain (100% of demand)
                    production_qty = actual_demand
                else:
                    # Overstocked - slight reduction (90% of demand)
                    production_qty = actual_demand * 0.9

            elif abc == 1:  # B-item: Balanced response
                # Target DOS for B-items: 21 days
                target_dos = 21.0
                if dos_position < 14:  # noqa: PLR2004
                    # Low - catch-up (110% of demand)
                    production_qty = actual_demand * 1.1
                elif dos_position < 35:  # noqa: PLR2004
                    # Normal - match demand
                    production_qty = actual_demand
                else:
                    # High - reduce (85% of demand)
                    production_qty = actual_demand * 0.85

            else:  # C-item: Slow response, let inventory self-correct
                # Target DOS for C-items: 28 days (we can afford more buffer)
                target_dos = 28.0
                if dos_position < 14:  # noqa: PLR2004
                    # Low - match demand (no aggressive catch-up for C)
                    production_qty = actual_demand
                elif dos_position < 45:  # noqa: PLR2004
                    # Normal - slight underproduce (90%)
                    production_qty = actual_demand * 0.9
                elif dos_position < 90:  # noqa: PLR2004
                    # High - reduce significantly (60%)
                    production_qty = actual_demand * 0.6
                else:
                    # Very high (SLOB territory) - minimal production (30%)
                    production_qty = actual_demand * 0.3

            # ============================================================
            # SAFETY BOUNDS
            # ============================================================
            # v0.20.0: SLOB throttling applied FIRST, then floor (floor is absolute)
            # This prevents SLOB from overriding safety floors and causing starvation

            # SLOB Throttling - apply BEFORE floor so floor always wins
            # Only throttle in extreme SLOB territory (90+ DOS = 3 months inventory)
            if dos_position > 90.0:  # noqa: PLR2004
                production_qty = min(production_qty, actual_demand * 0.7)
            elif dos_position > 75.0:  # noqa: PLR2004
                production_qty = min(production_qty, actual_demand * 0.85)

            # Floor: Only apply when DOS is critically low (< 7 days)
            # v0.22.0 DEATH SPIRAL FIX: Remove blanket daily floors!
            # With 500 SKUs, daily floors for ALL SKUs cause:
            # - 500 production orders/day
            # - Massive changeover time (25+ hours)
            # - Changeovers exceed daily capacity → production collapse
            #
            # NEW LOGIC: Only boost production when DOS is truly critical.
            # Let the DOS-based control logic (lines 835-878) handle normal cases.
            # This reduces changeover overhead dramatically.
            if dos_position < 7.0:  # noqa: PLR2004
                # Critical shortage - apply minimum floor
                abc_floors = {0: 0.7, 1: 0.5, 2: 0.3}
                floor_pct = abc_floors.get(abc, 0.3)
                sustainable_demand = float(self.sustainable_daily_demand[p_idx])
                production_qty = max(production_qty, sustainable_demand * floor_pct)

            # Cap: Never produce more than 150% of sustainable (prevent runaway)
            sustainable_demand = float(self.sustainable_daily_demand[p_idx])
            production_qty = min(production_qty, sustainable_demand * 1.5)

            order_qty = production_qty

            # Apply minimum batch size
            # v0.19.9: For rate-based production, use a lower minimum to avoid
            # filtering out C-items entirely. C-items with Zipfian distribution
            # average ~450 cases/day, so use 100 as the floor.
            mfg_config = self.config.get(
                "simulation_parameters", {}
            ).get("manufacturing", {})
            # Use a lower min batch for rate-based mode to capture C-items
            absolute_min = float(mfg_config.get("rate_based_min_batch", 100.0))

            # Only create order if meaningful quantity
            if order_qty >= absolute_min:
                plant_id = self._select_plant(product_id)
                po = ProductionOrder(
                    id=self._generate_po_id(current_day),
                    plant_id=plant_id,
                    product_id=product_id,
                    quantity_cases=order_qty,
                    creation_day=current_day,
                    due_day=current_day + self.production_lead_time,
                    status=ProductionOrderStatus.PLANNED,
                    planned_start_day=current_day + 1,
                )
                production_orders.append(po)

        # ================================================================
        # v0.22.0 DEATH SPIRAL FIX: Cap total orders at plant capacity
        # ================================================================
        # Individual SKU caps don't guarantee total stays within capacity.
        # When many SKUs request catch-up (130% of demand), total can exceed
        # what plants can produce. Backlog accumulates → more changeovers
        # → less production → death spiral.
        #
        # FIX: If total orders exceed 95% of capacity, scale down proportionally.
        # The 95% buffer leaves room for catch-up without creating backlog.
        total_orders = sum(po.quantity_cases for po in production_orders)
        capacity_threshold = self._max_daily_capacity * 0.95

        if total_orders > capacity_threshold and total_orders > 0:
            scale_factor = capacity_threshold / total_orders
            for po in production_orders:
                po.quantity_cases = po.quantity_cases * scale_factor

        return production_orders

    def get_diagnostics(self) -> MRPDiagnostics | None:
        """Return the most recent diagnostics snapshot."""
        return self._diagnostics

    def _update_demand_history(self, day: int, shipments: list[Shipment]) -> None:
        """Update demand history with actual shipment quantities (FIX 3.3)."""
        daily_vol = np.zeros(self.state.n_products)
        for shipment in shipments:
            for line in shipment.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    daily_vol[p_idx] += line.quantity

        self.demand_history[self._history_ptr] = daily_vol
        # v0.15.6: Extended history from 7 to 14 days
        self._history_ptr = (self._history_ptr + 1) % 14

    def record_order_demand(self, orders: list[Order]) -> None:
        """
        v0.20.0: Record order-based demand as PRIMARY signal.

        This captures the TRUE demand signal - what customer DCs requested from
        RDCs, before allocation constrains it. Orders reflect actual demand
        while shipments collapse when inventory is constrained.

        v0.20.0 CHANGE: This is now the PRIMARY signal for MRP (not blended).
        The shipment-based update was removed to prevent death spiral where
        low shipments → low production → lower shipments.

        Args:
            orders: Orders to RDCs (source_id is RDC, target_id is customer DC)
        """
        daily_vol = np.zeros(self.state.n_products)
        for order in orders:
            for line in order.lines:
                p_idx = self.state.product_id_to_idx.get(line.product_id)
                if p_idx is not None:
                    daily_vol[p_idx] += line.quantity

        # v0.19.8: Clamp demand signal to prevent panic-order poisoning
        # Cap at 4x expected demand (allows for promos but stops 30x bullwhip)
        if self.expected_daily_demand is not None:
            max_signal = self.expected_daily_demand * 4.0
            daily_vol = np.minimum(daily_vol, max_signal)

            # v0.20.0: Also use expected demand as FLOOR to prevent death spiral
            # Order signal can collapse due to deduplication or allocation constraints
            # Expected demand ensures production matches at least baseline needs
            daily_vol = np.maximum(daily_vol, self.expected_daily_demand)

        # v0.20.0: REPLACE slot directly (no blending with shipments)
        # This is now the primary signal - advance pointer after writing
        self.demand_history[self._history_ptr] = daily_vol
        self._history_ptr = (self._history_ptr + 1) % 14

    def _estimate_demand(self, product_idx: int, daily_demand: np.ndarray) -> float:
        """
        Estimate average daily demand for a product.

        Uses the current day's demand as a proxy (could be enhanced with
        moving average or forecast in future milestones).
        """
        # Sum demand across all nodes for this product
        total_demand = float(np.sum(daily_demand[:, product_idx]))
        return max(total_demand, 1.0)  # Avoid division by zero

    def generate_purchase_orders(
        self,
        current_day: int,
        active_production_orders: list[ProductionOrder],
    ) -> list[Order]:
        """
        Generate Purchase Orders for ingredients at plants using Vectorized MRP.

        Uses PRODUCTION-BASED signal (not POS demand) to ensure ingredient
        replenishment matches actual consumption. This prevents ingredient
        exhaustion when bullwhip causes production > demand.

        1. Calculate production quantities from active orders
        2. Calculate Ingredient Requirement (Req = Production @ R)
        3. Check Inventory Position vs ROP
        4. Generate Orders

        v0.19.9: Enhanced diagnostics for ingredient ordering.
        """
        purchase_orders: list[Order] = []
        if not self._plant_ids:
            return purchase_orders

        # v0.19.8 FIX: Decouple ingredient ordering from historical production.
        # This breaks the feedback loop where low production → fewer ingredients
        # → more shortages → even lower production (starvation spiral).
        #
        # NEW LOGIC: Order ingredients based on SCHEDULED DEMAND (active backlog)
        # constrained by PLANT CAPACITY, NOT historical throughput.

        # 1. Calculate active backlog from production orders (what we NEED to produce)
        # Shape: [n_products]
        production_by_product = np.zeros(self.state.n_products, dtype=np.float64)
        for po in active_production_orders:
            if po.status != ProductionOrderStatus.COMPLETE:
                p_idx = self.state.product_id_to_idx.get(po.product_id)
                if p_idx is not None:
                    # Count remaining quantity to be produced
                    remaining = po.quantity_cases - po.produced_quantity
                    production_by_product[p_idx] += remaining

        total_backlog = np.sum(production_by_product)
        expected_daily = np.sum(self.expected_daily_demand)

        # 2. Determine target daily production rate
        # v0.19.8: Always order ingredients for at least expected demand.
        # The original fix only used backlog, which created a feedback loop
        # when production orders were at floor level (30%).
        #
        # NEW LOGIC:
        # - If backlog > expected: use backlog mix (catch up mode)
        # - If backlog <= expected: use expected demand mix (steady state)
        # - Cap at plant capacity
        expected_daily = np.sum(self.expected_daily_demand)

        if total_backlog > expected_daily:
            # Catch-up mode: order for the backlog using backlog mix
            target_daily_rate = min(total_backlog, self._max_daily_capacity)
            mix = production_by_product / total_backlog
            daily_production = mix * target_daily_rate
        else:
            # Steady-state: order for expected demand using expected mix
            # This ensures we always have ingredients for full production
            daily_production = self.expected_daily_demand.copy()

        # 3. Apply floors and caps for robustness
        # Floor: never order less than min_production_cap_pct of expected demand
        # This prevents complete ingredient starvation during demand lulls
        daily_production = np.maximum(
            daily_production, self.expected_daily_demand * self.min_production_cap_pct
        )

        # Cap: prevent bullwhip-driven explosion (max 2x expected demand)
        max_daily = self.expected_daily_demand * 2.0
        daily_production = np.minimum(daily_production, max_daily)

        # Distribute to plants (Fair Share assumption)
        n_plants = len(self._plant_ids)
        plant_production_share = daily_production / n_plants

        # 2. Calculate Ingredient Requirements Vector
        # Req[j] = Sum(PlantProduction[i] * R[i, j])
        # Vector-Matrix multiplication: production @ R
        ingredient_reqs = plant_production_share @ self.state.recipe_matrix

        # 3. Calculate Targets & ROPs
        # Target Inventory = Daily Req * Target Days
        target_levels = ingredient_reqs * self.target_vector
        rop_levels = ingredient_reqs * self.rop_vector

        # v0.15.4: Cap ingredient order quantities to prevent explosion
        # Max order per ingredient per day = daily requirement * target days * 2
        max_order_per_ingredient = ingredient_reqs * self.target_vector * 2.0

        # 4. Build Pipeline Vector (In-Transit to Plants)
        # Shape: [n_nodes, n_products] - but we only care about plants
        pipeline = np.zeros(
            (self.state.n_nodes, self.state.n_products), dtype=np.float64
        )
        for shipment in self.state.active_shipments:
            target_idx = self.state.node_id_to_idx.get(shipment.target_id)
            if target_idx is not None and shipment.target_id in self._plant_ids:
                for line in shipment.lines:
                    p_idx = self.state.product_id_to_idx.get(line.product_id)
                    if p_idx is not None:
                        pipeline[target_idx, p_idx] += line.quantity

        # 5. Process Plants (Vectorized check per plant)
        for plant_id in self._plant_ids:
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            if plant_idx is None:
                continue

            # Inventory Position = On Hand + Pipeline
            on_hand = self.state.inventory[plant_idx]
            in_transit = pipeline[plant_idx]
            inv_position = on_hand + in_transit

            # Mask: Only consider items where we have a requirement (Ingredients)
            # and where IP < ROP
            needs_ordering = (inv_position < rop_levels) & (ingredient_reqs > 0)

            # Get indices of items to order
            # This returns a tuple of arrays, we want the first (and only) dimension
            order_indices = np.where(needs_ordering)[0]

            for p_idx in order_indices:
                qty_needed = target_levels[p_idx] - inv_position[p_idx]

                # Apply MOQ
                # Ideally MOQ should be per ingredient from config/product
                # Using global min_production_qty as proxy or 1 pallet
                qty_to_order = max(qty_needed, self.min_ingredient_moq)

                # v0.15.4: Cap order quantity to prevent explosion
                qty_to_order = min(qty_to_order, max_order_per_ingredient[p_idx])

                ing_id = self.state.product_idx_to_id[p_idx]
                supplier_id = self._find_supplier_for_ingredient(plant_id, ing_id)

                if supplier_id:
                    order_id = f"PO-ING-{current_day:03d}-{len(purchase_orders):06d}"
                    purchase_order = Order(
                        id=order_id,
                        source_id=supplier_id,
                        target_id=plant_id,
                        creation_day=current_day,
                        lines=[OrderLine(ing_id, float(qty_to_order))],
                        status="OPEN",
                    )
                    purchase_orders.append(purchase_order)

        # Update diagnostics with ingredient ordering info
        if self._diagnostics:
            total_ing_orders = sum(
                line.quantity for po in purchase_orders for line in po.lines
            )
            self._diagnostics.ingredient_orders_generated = total_ing_orders
            self._diagnostics.total_backlog = total_backlog
            self._diagnostics.backlog_vs_expected_ratio = (
                total_backlog / expected_daily if expected_daily > 0 else 0.0
            )

        return purchase_orders

    def _find_supplier_for_ingredient(self, plant_id: str, ing_id: str) -> str | None:
        """Find a supplier that provides the ingredient to the plant."""
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        spof_config = mfg_config.get("spof", {})

        # C.2 FIX: Check SPOF ingredient - but verify link exists!
        if ing_id == spof_config.get("ingredient_id"):
            primary = spof_config.get("primary_supplier_id")
            if primary:
                # Only use primary supplier if valid link exists to this plant
                has_link = any(
                    link.source_id == primary and link.target_id == plant_id
                    for link in self.world.links.values()
                )
                if has_link:
                    return str(primary)
                # Fall through to generic case if no link exists

        # Generic case: find any supplier linked to this plant
        for link in self.world.links.values():
            if link.target_id == plant_id:
                source_node = self.world.nodes.get(link.source_id)
                if source_node and source_node.type == NodeType.SUPPLIER:
                    return source_node.id

        return None

    def _load_plant_capabilities(self, mfg_config: dict[str, Any]) -> None:
        """Load supported categories for each plant from config."""
        plant_params = mfg_config.get("plant_parameters", {})
        for plant_id, params in plant_params.items():
            cats = params.get("supported_categories", [])
            if cats:
                self.plant_capabilities[plant_id] = cats

    def _select_plant(self, product_id: str) -> str:
        """
        Select a plant for production based on product category capabilities.
        Falls back to round-robin if no specific capabilities defined.
        """
        if not self._plant_ids:
            raise ValueError("No plants available for production")

        product = self.world.products.get(product_id)
        if not product:
            # Fallback
            plant_idx = self._po_counter % len(self._plant_ids)
            return self._plant_ids[plant_idx]

        cat_name = product.category.name

        # Filter plants that support this category
        eligible_plants = []
        for pid in self._plant_ids:
            # If plant has specific capabilities defined, check them
            # If not defined, assume it can produce everything (legacy behavior)
            caps = self.plant_capabilities.get(pid)
            if caps is None or cat_name in caps:
                eligible_plants.append(pid)

        if not eligible_plants:
            # If no plant explicitly supports it, fallback to all (or raise error?)
            # For robustness, fallback to all but log/warn ideally.
            # We'll stick to robust fallback.
            eligible_plants = self._plant_ids

        # Round-robin based on order counter within eligible plants
        plant_idx = self._po_counter % len(eligible_plants)
        return eligible_plants[plant_idx]

    def _generate_po_id(self, current_day: int) -> str:
        """Generate unique Production Order ID."""
        self._po_counter += 1
        return f"PO-{current_day:03d}-{self._po_counter:06d}"
