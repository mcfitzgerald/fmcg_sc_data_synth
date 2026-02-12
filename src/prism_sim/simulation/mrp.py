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
        pos_engine: Any = None,
        base_demand_matrix: np.ndarray | None = None,
    ) -> None:
        self.world = world
        self.state = state
        self.config = config
        self.pos_engine = pos_engine
        self.base_demand_matrix = base_demand_matrix

        # v0.48.0: DRP planner for forward-netting production targets
        # Set externally by orchestrator after both MRP and DRP are initialized
        self.drp_planner: Any = None
        self._active_production_orders_cache: list[ProductionOrder] = []

        # Extract manufacturing config
        mrp_config = config.get("simulation_parameters", {}).get("manufacturing", {})
        self.target_days_supply = mrp_config.get("target_days_supply", 14.0)
        self.reorder_point_days = mrp_config.get("reorder_point_days", 7.0)
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

        # v0.64.0: Migrated hardcodes to config
        self._history_days = int(
            mrp_thresholds.get("demand_history_lookback_days", 14)
        )
        self._demand_based_min_days = float(
            mrp_thresholds.get("demand_based_min_days", 7.0)
        )
        self._production_smoothing_cap_mult = float(
            mrp_thresholds.get("production_smoothing_cap_multiplier", 1.5)
        )

        # v0.19.9: Rate-based production parameters (Option C fix)
        # When enabled, production always runs at expected demand rate.
        # DOS only modulates ABOVE baseline (catch-up mode), never below.
        self.rate_based_production = mrp_thresholds.get(
            "rate_based_production", False
        )
        # v0.46.0: ABC-differentiated inventory DOS caps (negative feedback loop)
        # When a product's DOS exceeds its cap, skip production until consumed.
        self.inventory_cap_dos_a = mrp_thresholds.get("inventory_cap_dos_a", 25.0)
        self.inventory_cap_dos_b = mrp_thresholds.get("inventory_cap_dos_b", 35.0)
        self.inventory_cap_dos_c = mrp_thresholds.get("inventory_cap_dos_c", 45.0)

        # v0.28.0: Seasonality parameters for demand-aware MRP
        # Without this, MRP uses flat expected demand for DOS/batch calculations,
        # causing overproduction during troughs → inventory ages → SLOB accumulation
        demand_config = config.get("simulation_parameters", {}).get("demand", {})
        season_config = demand_config.get("seasonality", {})
        self._seasonality_amplitude = season_config.get("amplitude", 0.12)
        self._seasonality_phase_shift = season_config.get("phase_shift_days", 150)
        self._seasonality_cycle_days = season_config.get("cycle_days", 365)
        # v0.29.0: Capacity amplitude for flexible production capacity
        self._capacity_amplitude = season_config.get("capacity_amplitude", 0.0)

        # v0.46.0: Seasonal floor minimum percentage
        # During seasonal troughs, the demand floor tracks seasonality but never
        # drops below this fraction of expected demand (prevents death spiral).
        self._seasonal_floor_min_pct = mrp_thresholds.get(
            "seasonal_floor_min_pct", 0.85
        )

        # v0.53.0: MRP floor gating — gate seasonal floor on inventory position.
        # When inventory is at or above target, the floor disengages so production
        # can suppress. Prevents unconditional 85% floor from causing +92% drift.
        self._mrp_floor_gating_enabled = mrp_thresholds.get(
            "mrp_floor_gating_enabled", False
        )

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
            self.expected_daily_demand, (self._history_days, 1)
        ).astype(np.float64)
        self._history_ptr = 0

        # v0.39.2: Consumption tracking for demand signal calibration (SLOB fix)
        # Tracks actual consumption (what was sold) vs expected demand.
        # This feedback allows MRP to adjust production to match reality,
        # preventing over-production when service level < 100%.
        self._consumption_history = np.zeros(
            (self._history_days, self.state.n_products), dtype=np.float64
        )
        self._consumption_ptr = 0

        # C.5 FIX: Production order history for smoothing (reduces volatility)
        # v0.15.6: Extended from 7 to 14 days
        # v0.19.9: Warm start with total expected production
        total_expected_production = np.sum(self.expected_daily_demand)
        self.production_order_history = np.full(
            self._history_days, total_expected_production, dtype=np.float64
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
        # v0.39.3: c_production_factor removed - C-items use longer horizons instead
        # of smaller batches (see production_horizon_days_c config)
        self.c_production_factor = mrp_thresholds.get("c_production_factor", 0.6)
        # v0.42.0: A-item capacity share for Phase 4 ABC-aware clipping
        self.a_capacity_share = mrp_thresholds.get("a_capacity_share", 0.65)
        # v0.39.3: c_demand_factor removed - was dead code (never used in calculations)

        # v0.46.0: SLOB production dampening
        # When a product's weighted inventory age exceeds its ABC-class SLOB
        # threshold, reduce batch size to slow further accumulation of aged stock.
        validation_config = config.get("simulation_parameters", {}).get(
            "validation", {}
        )
        slob_abc = validation_config.get("slob_abc_thresholds", {})
        self.slob_threshold_a = float(slob_abc.get("A", 60.0))
        self.slob_threshold_b = float(slob_abc.get("B", 90.0))
        self.slob_threshold_c = float(slob_abc.get("C", 120.0))
        # v0.56.0: Graduated SLOB dampening (replaces binary slob_dampening_factor)
        # Floor = minimum production rate for aged products (was 0.25 binary)
        # Ramp multiplier = ramp length as fraction of threshold
        self.slob_dampening_floor = mrp_thresholds.get(
            "slob_dampening_floor", 0.50
        )
        self.slob_ramp_multiplier = mrp_thresholds.get(
            "slob_dampening_ramp_multiplier", 1.0
        )
        # v0.56.0: Demand smoothing weight for DOS calculation
        # Blend smoothed expected demand (weight) + raw POS (1-weight)
        self.demand_smoothing_weight = mrp_thresholds.get(
            "demand_smoothing_weight", 0.7
        )

        # v0.23.0: Campaign Batching parameters
        # DEATH SPIRAL FIX: Instead of producing all 500 SKUs daily (25h changeover),
        # produce larger batches for fewer SKUs (campaign runs).
        # Only trigger production when DOS drops below threshold.
        # When triggered, produce enough to cover the full horizon.
        campaign_config = mrp_thresholds.get("campaign_batching", {})
        self.campaign_batching_enabled = campaign_config.get("enabled", True)
        self.production_horizon_days = campaign_config.get(
            "production_horizon_days", 14
        )
        # Different DOS triggers by ABC class (A needs tighter coverage)
        self.trigger_dos_a = campaign_config.get("trigger_dos_a", 10)
        self.trigger_dos_b = campaign_config.get("trigger_dos_b", 7)
        self.trigger_dos_c = campaign_config.get("trigger_dos_c", 5)
        # Hard cap on SKUs per plant per day to limit changeover overhead
        self.max_skus_per_plant_per_day = campaign_config.get(
            "max_skus_per_plant_per_day", 25
        )
        # v0.39.1: ABC-differentiated production horizons
        # C-items use shorter horizon to prevent inventory buildup → SLOB
        self.production_horizon_days_a = campaign_config.get(
            "production_horizon_days_a", 14
        )
        self.production_horizon_days_b = campaign_config.get(
            "production_horizon_days_b", 10
        )
        self.production_horizon_days_c = campaign_config.get(
            "production_horizon_days_c", 5
        )
        # v0.42.0: Config-driven ABC slot percentages for campaign scheduling
        self.a_slot_pct = campaign_config.get("a_slot_pct", 0.60)
        self.b_slot_pct = campaign_config.get("b_slot_pct", 0.25)
        self.c_slot_pct = campaign_config.get("c_slot_pct", 0.15)

        # v0.37.0: PO Consolidation parameters
        # Consolidate ingredient POs to improve inbound fill rate
        po_consolidation_config = mrp_thresholds.get("po_consolidation", {})
        self.po_consolidation_enabled = po_consolidation_config.get("enabled", False)
        self.po_window_days = po_consolidation_config.get("window_days", 2)
        self.po_min_weight_kg = po_consolidation_config.get("min_weight_kg", 5000)
        self.po_critical_dos = po_consolidation_config.get("critical_dos_threshold", 3)

        # PO consolidation state: supplier_id -> list of pending OrderLines
        self._pending_po_lines: dict[str, list[OrderLine]] = {}
        self._po_window_start: dict[str, int] = {}  # supplier_id -> start day

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
        Example: 21M expected demand x 0.8 floor = 16.8M requested,
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

            cat_p_arr = np.array(cat_p_indices)
            cat_demands_arr = np.array(cat_demands)

            cat_total_volume = np.sum(cat_demands_arr)
            if cat_total_volume == 0:
                continue

            # Sort within category
            sorted_local_indices = np.argsort(cat_demands_arr)[::-1]
            sorted_global_indices = cat_p_arr[sorted_local_indices]

            cumulative_volume = np.cumsum(cat_demands_arr[sorted_local_indices])

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

        v0.32.0: Updated to use line-based capacity logic (num_lines).
        Replaces the legacy rate_multiplier hack with explicit line counts.

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

        default_num_lines = mfg_config.get("default_num_lines", 4)
        # Multiplier should be 1.0 now, but kept for legacy compat if needed
        rate_multiplier = mfg_config.get("production_rate_multiplier", 1.0)

        total_capacity = 0.0

        for plant_id in self._plant_ids:
            # Get plant-specific config
            p_config = plant_params.get(plant_id, {})
            efficiency = p_config.get("efficiency_factor", global_efficiency)
            downtime = p_config.get("unplanned_downtime_pct", global_downtime)
            num_lines = p_config.get("num_lines", default_num_lines)

            # Effective hours PER LINE = hours * (1 - downtime) * efficiency
            effective_hours_per_line = hours_per_day * (1.0 - downtime) * efficiency

            # Get average run rate for products this plant can make
            supported_cats = p_config.get("supported_categories", [])
            run_rates = []

            for product_id, recipe in self.world.recipes.items():
                product = self.world.products.get(product_id)
                if product is None:
                    continue

                if supported_cats:
                    if product.category.name not in supported_cats:
                        continue

                run_rates.append(recipe.run_rate_cases_per_hour)

            if run_rates:
                avg_run_rate = sum(run_rates) / len(run_rates)
                # Plant capacity = capacity per line * num_lines
                plant_capacity = effective_hours_per_line * avg_run_rate * num_lines
                total_capacity += plant_capacity

        return float(total_capacity * rate_multiplier)

    def _cache_node_info(self) -> None:
        """Cache RDC, Plant, and plant-direct DC node IDs.

        Note: Only manufacturer-controlled RDCs (RDC-*) are included,
        not customer DCs (RET-DC-*, DIST-DC-*, ECOM-*, etc.) which
        represent customer inventory, not manufacturer inventory position.

        v0.55.0: Also identifies plant-direct DCs for pipeline IP expansion.
        """
        # Build upstream map for plant-direct DC identification
        upstream_map: dict[str, str] = {}
        for link in self.world.links.values():
            upstream_map[link.target_id] = link.source_id

        self._plant_direct_dc_ids: set[str] = set()

        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.DC:
                # Only include manufacturer RDCs, not customer DCs
                if node_id.startswith("RDC-"):
                    self._rdc_ids.append(node_id)
                elif upstream_map.get(node_id, "").startswith("PLANT-"):
                    # Plant-direct DC: upstream link from a plant
                    self._plant_direct_dc_ids.add(node_id)
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
        """Calculate pipeline Inventory Position for a product.

        v0.55.0: Plant FG + transit + in-production. Plant FG provides
        natural MRP backpressure: products with excess FG get suppressed,
        freeing capacity for products that need it.

        IP = plant_fg + in-transit + in-production.
        """
        # Sum FG across all plants for this product
        plant_fg = 0.0
        p_idx = self.state.product_id_to_idx.get(product_id)
        if p_idx is not None:
            for plant_id in self._plant_ids:
                pi = self.state.node_id_to_idx.get(plant_id)
                if pi is not None:
                    plant_fg += max(
                        0.0, float(self.state.actual_inventory[pi, p_idx])
                    )

        return (
            plant_fg
            + in_transit_qty.get(product_id, 0.0)
            + in_production_qty.get(product_id, 0.0)
        )

    def _get_seasonal_factor(self, day: int) -> float:
        """
        v0.28.0: Calculate seasonal demand factor for a given day.

        Returns a multiplier (e.g., 0.88 for trough, 1.12 for peak) that
        represents actual demand relative to expected annual average.

        This matches the seasonality calculation in POSEngine.generate_demand().
        """
        phase = (
            (day - self._seasonality_phase_shift)
            / self._seasonality_cycle_days
        )
        return float(
            1.0 + self._seasonality_amplitude
            * np.sin(2 * np.pi * phase)
        )

    def _get_daily_capacity(self, day: int) -> float:
        """
        v0.29.0: Get production capacity for a specific day,
        accounting for seasonal flex.

        Mirrors demand seasonality so MRP can plan with accurate capacity knowledge:
        - Peak demand → higher capacity available (overtime, extra shifts)
        - Trough demand → lower capacity available (reduced shifts, maintenance)

        This allows MRP to generate appropriately-sized production orders that
        TransformEngine can actually execute within the day's capacity.

        Args:
            day: Current simulation day

        Returns:
            Total network capacity in cases/day, seasonally adjusted
        """
        if self._capacity_amplitude == 0:
            return self._max_daily_capacity

        # Apply same sinusoidal pattern as demand seasonality
        phase = (
            (day - self._seasonality_phase_shift)
            / self._seasonality_cycle_days
        )
        capacity_factor = (
            1.0 + self._capacity_amplitude
            * np.sin(2 * np.pi * phase)
        )
        return float(
            self._max_daily_capacity * capacity_factor
        )

    def generate_production_orders(
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
        half = self._history_days // 2
        week1_avg = np.mean(self.demand_history[:half], axis=0)
        week2_avg = np.mean(self.demand_history[half:], axis=0)
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
        # v0.55.0: Include all plant-sourced in-transit (RDCs + DCs).
        # Deployment sends to both; MRP needs the full pipeline picture.
        # Combined with plant FG in IP, this gives accurate backpressure.
        in_transit_qty: dict[str, float] = {}
        plant_id_set = set(self._plant_ids)
        for shipment in self.state.active_shipments:
            if shipment.source_id in plant_id_set:
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

        # v0.48.0: Cache active POs for DRP planner access
        self._active_production_orders_cache = active_production_orders

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
                        demand_based_min = (
                            avg_daily_demand * self._demand_based_min_days
                        )
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
        smoothing_cap = smoothing_baseline * self._production_smoothing_cap_mult
        if smoothing_baseline > 0 and total_orders_today > smoothing_cap:
            cap_applied = True
            scale_factor = smoothing_cap / total_orders_today
            for po in production_orders:
                po.quantity_cases = float(po.quantity_cases * scale_factor)

        # Update production history
        actual_total = sum(po.quantity_cases for po in production_orders)
        self.production_order_history[self._prod_hist_ptr] = actual_total
        self._prod_hist_ptr = (self._prod_hist_ptr + 1) % self._history_days

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

    def _generate_rate_based_orders(
        self,
        current_day: int,
        avg_daily_demand_vec: np.ndarray,
        in_transit_qty: dict[str, float],
        in_production_qty: dict[str, float],
        total_rdc_inv: float,
        pos_demand_vec: np.ndarray | None = None,
    ) -> list[ProductionOrder]:
        """
        v0.23.0: Campaign Batching production (DEATH SPIRAL FIX).

        Key insight: Instead of producing all 500 SKUs daily (causing 25h+ of
        changeovers), produce larger batches for fewer SKUs. This matches how
        real FMCG plants operate with "campaign runs".

        Production logic:
        1. TRIGGER: Only produce when DOS drops below threshold (not daily)
        2. BATCH SIZE: Produce horizon_days worth (not daily demand)
        3. PRIORITY: Lowest DOS first (most critical items)
        4. CAP: Max SKUs per plant per day (limits changeover overhead)

        Math example:
        - 500 SKUs, produce 14 days' worth when DOS < 7
        - Each SKU produced every ~7 days = 500/7 ≈ 71 SKUs/day
        - Split across 4 plants = ~18 SKUs/plant/day
        - 18 x 0.05h changeover = 0.9h/plant (vs 25h before)
        """
        # ================================================================
        # PHASE 1: Build candidate list - only SKUs that need production
        # ================================================================
        # Candidate tuple: (product_id, plant_id, dos, batch_qty, abc_class)
        candidates: list[tuple[str, str, float, float, int]] = []

        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        absolute_min = float(mfg_config.get("rate_based_min_batch", 100.0))

        # v0.36.0 Demand Sensing: Cache future deterministic
        # forecast for the planning horizon.
        # This includes deterministic seasonality and promos.
        if self.pos_engine is not None:
            forecast_duration = self.production_horizon_days
            network_forecast = (
                self.pos_engine.get_deterministic_forecast(
                    current_day, forecast_duration
                )
            )
            # Planning daily rate = Total Forecast over horizon / Duration
            planning_daily_rate_vec = network_forecast / forecast_duration
        else:
            planning_daily_rate_vec = None

        # v0.48.0: Get DRP daily targets for B/C items (forward-netting)
        drp_targets: np.ndarray | None = None
        if self.drp_planner is not None:
            drp_targets = self.drp_planner.plan_requirements(
                current_day,
                self.abc_class,
                self._active_production_orders_cache,
            )

        # v0.46.0: Pre-compute weighted inventory ages for SLOB dampening
        product_ages = self.state.get_weighted_age_by_product()

        for product_id in self._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue

            # Get expected demand for batch sizing
            expected_demand = float(self.expected_daily_demand[p_idx])
            if expected_demand <= 0:
                continue

            # v0.39.1: Get ABC class early for horizon selection
            abc = int(self.abc_class[p_idx])
            if abc == 0:  # A-item
                abc_horizon = self.production_horizon_days_a
            elif abc == 1:  # B-item
                abc_horizon = self.production_horizon_days_b
            else:  # C-item
                abc_horizon = self.production_horizon_days_c

            expected = float(self.expected_daily_demand[p_idx])

            inventory_position = self._calculate_inventory_position(
                product_id, in_transit_qty, in_production_qty
            )

            if pos_demand_vec is not None:
                # v0.56.0: Blend expected demand + raw POS to reduce noise
                # while retaining responsiveness to genuine demand shifts.
                # Single-day POS spikes/drops cause false DOS cap hits and
                # volatile batch sizing.
                pos_today = max(float(pos_demand_vec[p_idx]), 1.0)
                demand_for_dos = max(
                    self.demand_smoothing_weight * expected
                    + (1.0 - self.demand_smoothing_weight) * pos_today,
                    1.0,
                )
            elif planning_daily_rate_vec is not None:
                demand_for_dos = max(expected, float(planning_daily_rate_vec[p_idx]))
            else:
                demand_for_dos = max(expected, 1.0)

            # v0.39.2 FIX: Decouple batch sizing from forecast totals
            # Simple formula: rate x horizon (no recycling of 14-day totals)
            batch_qty_base = demand_for_dos * abc_horizon

            # Calculate DOS
            dos_position = inventory_position / max(demand_for_dos, 0.1)

            # v0.46.0: DOS cap guard — skip production when inventory is sufficient.
            # This is the missing negative feedback loop: when DOS exceeds the
            # ABC-differentiated cap, production stops until inventory is consumed.
            if abc == 0:
                cap_dos = self.inventory_cap_dos_a
            elif abc == 1:
                cap_dos = self.inventory_cap_dos_b
            else:
                cap_dos = self.inventory_cap_dos_c
            if dos_position > cap_dos:
                continue

            # Get trigger threshold for this ABC class (abc already computed above)
            if abc == 0:  # A-item
                trigger_dos = self.trigger_dos_a
            elif abc == 1:  # B-item
                trigger_dos = self.trigger_dos_b
            else:  # C-item
                trigger_dos = self.trigger_dos_c

            # ============================================================
            # PHASE 1: ABC-branched candidate selection
            # ============================================================

            # v0.46.0: Pre-compute product age for SLOB dampening
            product_age = float(product_ages[p_idx])
            if abc == 0:
                slob_thresh = self.slob_threshold_a
            elif abc == 1:
                slob_thresh = self.slob_threshold_b
            else:
                slob_thresh = self.slob_threshold_c

            if abc == 0:
                # --------------------------------------------------------
                # A-ITEMS: Net-requirement scheduling (MPS-style)
                # --------------------------------------------------------
                # Instead of trigger-based feast/famine, compute the gap
                # between target inventory and current inventory position.
                # Each A-item gets a small, demand-matched batch every
                # ~2.2 days (310 items / 140 A-slots), keeping plants
                # steadily loaded and inventory near target.
                target_inventory = (
                    demand_for_dos * abc_horizon * self.a_production_buffer
                )
                net_requirement = target_inventory - inventory_position

                if net_requirement < absolute_min:
                    continue  # At or above target — no production needed

                batch_qty = net_requirement

                # v0.56.0: Graduated SLOB dampening (replaces binary 0.25 cut)
                batch_qty = self._apply_slob_dampening(
                    batch_qty, product_age, slob_thresh
                )

                plant_id = self._select_plant(product_id)
                candidates.append(
                    (product_id, plant_id, dos_position, batch_qty, abc)
                )

            elif drp_targets is not None:
                # --------------------------------------------------------
                # B/C ITEMS: DRP-driven production (v0.48.0)
                # --------------------------------------------------------
                # Instead of binary trigger (dos < threshold → produce
                # horizon days' worth), use DRP's forward-netted daily
                # target. This produces smaller, more frequent batches
                # that match actual net requirements.
                drp_daily = float(drp_targets[p_idx])
                if drp_daily < absolute_min * 0.1:
                    continue  # DRP says no production needed

                # Scale DRP daily target by ABC horizon to get batch qty
                # that integrates naturally with the slot allocation system
                batch_qty = drp_daily * abc_horizon

                b_production_buffer = float(
                    self.config.get("simulation_parameters", {})
                    .get("manufacturing", {})
                    .get("mrp_thresholds", {})
                    .get("b_production_buffer", 1.1)
                )

                if abc == 1:  # B-item
                    batch_qty *= b_production_buffer
                elif abc == 2:  # C-item  # noqa: PLR2004
                    batch_qty *= self.c_production_factor

                # v0.56.0: Graduated SLOB dampening (replaces binary 0.25 cut)
                batch_qty = self._apply_slob_dampening(
                    batch_qty, product_age, slob_thresh
                )

                if batch_qty < absolute_min:
                    continue

                plant_id = self._select_plant(product_id)
                candidates.append(
                    (product_id, plant_id, dos_position, batch_qty, abc)
                )

            elif dos_position < trigger_dos:
                # --------------------------------------------------------
                # B/C ITEMS: Fallback campaign trigger (no DRP planner)
                # --------------------------------------------------------
                batch_qty = batch_qty_base

                b_production_buffer = float(
                    self.config.get("simulation_parameters", {})
                    .get("manufacturing", {})
                    .get("mrp_thresholds", {})
                    .get("b_production_buffer", 1.1)
                )

                if abc == 1:  # B-item
                    batch_qty *= b_production_buffer
                elif abc == 2:  # C-item  # noqa: PLR2004
                    batch_qty *= self.c_production_factor

                # DOS throttling for B-items only (high inventory levels).
                abc_class_c = 2
                if abc != abc_class_c:  # B-items only
                    if dos_position > 60.0:  # noqa: PLR2004
                        batch_qty *= 0.5
                    elif dos_position > 45.0:  # noqa: PLR2004
                        batch_qty *= 0.7

                # v0.56.0: Graduated SLOB dampening (replaces binary 0.25 cut)
                batch_qty = self._apply_slob_dampening(
                    batch_qty, product_age, slob_thresh
                )

                if batch_qty < absolute_min:
                    continue

                plant_id = self._select_plant(product_id)
                candidates.append(
                    (product_id, plant_id, dos_position, batch_qty, abc)
                )

        # ================================================================
        # PHASE 2: Sort by priority (lowest Critical Ratio first)
        # ================================================================
        # CRITICAL FIX (v0.36.1): Use Critical Ratio (DOS / Trigger) for sorting.
        # Previously sorted by (ABC, DOS), which meant A-items ALWAYS beat C-items.
        # This caused "Starvation" where C-items were never produced because A-items
        # constantly filled the daily SKU slots.
        #
        # New Sort: ratio = dos / trigger_dos
        # - Ratio < 1.0 means below trigger (urgent)
        # - Ratio << 1.0 means imminent stockout (most urgent)
        # - This creates a "Fairness" mechanic where a C-item at 10% inventory
        #   prioritizes above an A-item at 90% inventory.

        def get_trigger(abc_code: int) -> float:
            if abc_code == 0:
                return float(self.trigger_dos_a)
            if abc_code == 1:
                return float(self.trigger_dos_b)
            return float(self.trigger_dos_c)

        # v0.36.2: Shuffle to break ties (e.g. multiple items with 0 inventory)
        # Without shuffle, the same "tail" items always lose
        # the tie-break and never run.
        np.random.shuffle(candidates)

        # Sort by Critical Ratio ascending (lowest ratio = most critical)
        candidates.sort(key=lambda x: x[2] / get_trigger(x[4]))

        # ================================================================
        # PHASE 3: Select top N per plant with ABC slot reservation
        # ================================================================
        # v0.38.0 FIX: Reserve slots per ABC class to prevent C-item starvation.
        # Previously, sorting all SKUs by critical ratio caused A-items to
        # dominate (lower DOS = higher priority), starving C-items which never
        # got produced. This caused ~200 C-items to accumulate old inventory
        # that was flagged as SLOB.
        #
        # Reserve proportional slots for each ABC class (config-driven):
        # - A-items get most slots (highest velocity, need frequent production)
        # - B-items get medium share
        # - C-items get smallest share (ensures coverage despite low velocity)
        #
        # Within each class, items are still sorted by critical ratio.
        #
        # This improves OEE (more SKUs run → more run time) and reduces SLOB
        # (C-items get produced before inventory ages out).
        a_slot_pct = self.a_slot_pct
        b_slot_pct = self.b_slot_pct
        c_slot_pct = self.c_slot_pct

        # Group by plant first
        plant_orders: dict[str, list[tuple[str, str, float, float, int]]] = {}
        for cand in candidates:
            plant_id = cand[1]
            if plant_id not in plant_orders:
                plant_orders[plant_id] = []
            plant_orders[plant_id].append(cand)

        production_orders: list[ProductionOrder] = []

        for plant_id, plant_candidates in plant_orders.items():
            max_skus = self.max_skus_per_plant_per_day

            # Split candidates by ABC class
            # Candidates already sorted by critical ratio (asc)
            abc_a, abc_b, abc_c = 0, 1, 2
            a_items = [c for c in plant_candidates if c[4] == abc_a]
            b_items = [c for c in plant_candidates if c[4] == abc_b]
            c_items = [c for c in plant_candidates if c[4] == abc_c]

            # Calculate slot allocation
            a_slots = int(max_skus * a_slot_pct)
            b_slots = int(max_skus * b_slot_pct)
            c_slots = int(max_skus * c_slot_pct)

            # Select top N from each class (already sorted by critical ratio)
            selected_a = a_items[:a_slots]
            selected_b = b_items[:b_slots]
            selected_c = c_items[:c_slots]

            # If one class has fewer candidates than slots, redistribute
            # unused slots to other classes (A gets priority, then B)
            unused_a = max(0, a_slots - len(selected_a))
            unused_b = max(0, b_slots - len(selected_b))
            unused_c = max(0, c_slots - len(selected_c))
            total_unused = unused_a + unused_b + unused_c

            if total_unused > 0:
                # Redistribute unused slots
                remaining_a = a_items[len(selected_a):]
                remaining_b = b_items[len(selected_b):]
                remaining_c = c_items[len(selected_c):]

                # Pool remaining candidates and sort by critical ratio
                remaining_all = remaining_a + remaining_b + remaining_c
                remaining_all.sort(key=lambda x: x[2] / get_trigger(x[4]))

                # Take up to unused slots from remaining
                extra = remaining_all[:total_unused]
                selected = selected_a + selected_b + selected_c + extra
            else:
                selected = selected_a + selected_b + selected_c

            for product_id, _, _dos, batch_qty, _ in selected:
                po = ProductionOrder(
                    id=self._generate_po_id(current_day),
                    plant_id=plant_id,
                    product_id=product_id,
                    quantity_cases=batch_qty,
                    creation_day=current_day,
                    due_day=current_day + self.production_lead_time,
                    status=ProductionOrderStatus.PLANNED,
                    planned_start_day=current_day + 1,
                )
                production_orders.append(po)

        # PHASE 4: ABC-aware capacity clipping (v0.42.0)
        self._apply_abc_capacity_clipping(production_orders, current_day)

        return production_orders

    def get_diagnostics(self) -> MRPDiagnostics | None:
        """Return the most recent diagnostics snapshot."""
        return self._diagnostics

    def _get_abc_class(self, product_id: str) -> int:
        """Return ABC class for a product: 0=A, 1=B, 2=C."""
        p_idx = self.state.product_id_to_idx.get(product_id)
        if p_idx is not None:
            return int(self.abc_class[p_idx])
        return 2  # Default to C-item if unknown

    def _apply_abc_capacity_clipping(
        self,
        production_orders: list[ProductionOrder],
        current_day: int,
    ) -> None:
        """Apply ABC-aware capacity clipping to production orders (Phase 4).

        A-items (demand-matched, small batches) are protected up to
        a_capacity_share of capacity. B/C campaign batches absorb clipping
        first since they have more fill-rate headroom.
        """
        total_orders = sum(po.quantity_cases for po in production_orders)
        daily_capacity = self._get_daily_capacity(current_day)
        capacity_threshold = daily_capacity * 0.98

        if total_orders <= capacity_threshold or total_orders <= 0:
            return

        a_orders = [
            po for po in production_orders
            if self._get_abc_class(po.product_id) == 0
        ]
        bc_orders = [
            po for po in production_orders
            if self._get_abc_class(po.product_id) != 0
        ]

        a_total = sum(po.quantity_cases for po in a_orders)
        bc_total = sum(po.quantity_cases for po in bc_orders)
        a_cap = capacity_threshold * self.a_capacity_share

        if a_total <= a_cap:
            # A-items fit within their share — clip only B/C
            remaining = capacity_threshold - a_total
            if bc_total > remaining and bc_total > 0:
                bc_scale = remaining / bc_total
                for po in bc_orders:
                    po.quantity_cases *= bc_scale
        else:
            # A-items exceed their share — clip both, A less aggressively
            a_scale = a_cap / a_total
            for po in a_orders:
                po.quantity_cases *= a_scale
            remaining = capacity_threshold - a_cap
            if bc_total > remaining and bc_total > 0:
                bc_scale = remaining / bc_total
                for po in bc_orders:
                    po.quantity_cases *= bc_scale

    def _apply_slob_dampening(
        self, batch_qty: float, product_age: float, slob_thresh: float
    ) -> float:
        """v0.56.0: Graduated SLOB dampening — linear ramp from 1.0 to floor.

        Replaces the binary 0.25 cut that caused a death spiral:
        products just crossing the threshold were cut 75%, falling below
        absolute_min and being silently dropped from production.

        Now: at threshold → 1.0 (no dampening), at threshold + ramp → floor.
        The ramp length = slob_thresh * slob_ramp_multiplier.
        """
        if product_age <= slob_thresh:
            return batch_qty
        ramp_end = slob_thresh * self.slob_ramp_multiplier
        age_ratio = min((product_age - slob_thresh) / max(ramp_end, 1.0), 1.0)
        dampening = 1.0 - age_ratio * (1.0 - self.slob_dampening_floor)
        return batch_qty * dampening

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
        self._history_ptr = (self._history_ptr + 1) % self._history_days

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
        self._history_ptr = (self._history_ptr + 1) % self._history_days

    def record_consumption(self, actual_sales: np.ndarray) -> None:
        """
        v0.39.2: Record actual consumption for demand signal calibration.

        This tracks what was actually sold (constrained by inventory availability),
        not what was demanded. The gap between demand and consumption represents
        lost sales due to stockouts.

        Production should track actual consumption, not inflated demand expectations.
        This prevents the over-production that causes SLOB accumulation.

        Args:
            actual_sales: Shape [n_nodes, n_products] - actual sales per node/product
        """
        # Sum consumption across all nodes for each product
        daily_consumption = np.sum(actual_sales, axis=0)
        self._consumption_history[self._consumption_ptr] = daily_consumption
        self._consumption_ptr = (self._consumption_ptr + 1) % self._history_days

    def get_actual_daily_demand(self) -> np.ndarray:
        """
        v0.39.2: Return 14-day average of actual consumption per product.

        This provides a realistic demand signal that accounts for service level.
        When service is 89%, actual consumption is 89% of theoretical demand.

        Returns:
            np.ndarray: Shape [n_products] - average daily consumption
        """
        result: np.ndarray = np.mean(self._consumption_history, axis=0)
        return result

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

        v0.37.0: Added PO consolidation to improve inbound fill rate.
        When enabled, POs are batched by supplier until weight or window
        criteria are met, resulting in fuller trucks.

        1. Calculate production quantities from active orders
        2. Calculate Ingredient Requirement (Req = Production @ R)
        3. Check Inventory Position vs ROP
        4. Generate Orders (with optional consolidation)
        """
        if not self._plant_ids:
            return []

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

        # 5. Process Plants - collect candidate order lines
        # candidate_lines: list of (plant_id, supplier_id, ing_id, qty, dos)
        candidate_lines: list[tuple[str, str, str, float, float]] = []

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
            order_indices = np.where(needs_ordering)[0]

            for p_idx in order_indices:
                qty_needed = target_levels[p_idx] - inv_position[p_idx]

                # Apply MOQ
                qty_to_order = max(qty_needed, self.min_ingredient_moq)

                # v0.15.4: Cap order quantity to prevent explosion
                qty_to_order = min(qty_to_order, max_order_per_ingredient[p_idx])

                ing_id = self.state.product_idx_to_id[p_idx]
                supplier_id = self._find_supplier_for_ingredient(plant_id, ing_id)

                if supplier_id:
                    # Calculate DOS for critical ingredient detection
                    daily_req = ingredient_reqs[p_idx]
                    dos = inv_position[p_idx] / daily_req if daily_req > 0 else 999.0
                    candidate_lines.append(
                        (plant_id, supplier_id, ing_id, float(qty_to_order), dos)
                    )

        # 6. Apply consolidation logic (v0.37.0)
        purchase_orders = self._apply_po_consolidation(
            current_day, candidate_lines, ingredient_reqs
        )

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

    def _apply_po_consolidation(
        self,
        current_day: int,
        candidate_lines: list[tuple[str, str, str, float, float]],
        ingredient_reqs: np.ndarray,
    ) -> list[Order]:
        """
        v0.37.0: Apply PO consolidation to improve inbound fill rate.

        Consolidation logic:
        1. Add new candidate lines to pending pool by supplier
        2. Check release criteria for each supplier:
           - Window expired (2 days default)
           - Weight threshold met (5000 kg default)
           - Critical DOS detected (< 3 days)
        3. Release consolidated POs when criteria met

        Args:
            current_day: Current simulation day
            candidate_lines: List of (plant_id, supplier_id, ing_id, qty, dos)
            ingredient_reqs: Daily ingredient requirement vector

        Returns:
            List of (possibly consolidated) Purchase Orders
        """
        if not self.po_consolidation_enabled:
            # Immediate mode: create individual POs
            return self._create_immediate_pos(current_day, candidate_lines)

        # Add new lines to pending pool by supplier
        for plant_id, supplier_id, ing_id, qty, _dos in candidate_lines:
            # Create a supplier key that includes plant for proper routing
            supplier_key = f"{supplier_id}|{plant_id}"

            if supplier_key not in self._pending_po_lines:
                self._pending_po_lines[supplier_key] = []
                self._po_window_start[supplier_key] = current_day

            self._pending_po_lines[supplier_key].append(
                OrderLine(ing_id, qty)
            )

        # Check release criteria for each supplier
        released_pos: list[Order] = []

        for supplier_key in list(self._pending_po_lines.keys()):
            pending_lines = self._pending_po_lines[supplier_key]
            window_start = self._po_window_start[supplier_key]

            # Parse supplier key back to IDs
            supplier_id, plant_id = supplier_key.split("|")

            # Calculate total weight
            total_weight = sum(
                line.quantity * self._get_product_weight(line.product_id)
                for line in pending_lines
            )

            # Check if any ingredient is critical
            has_critical = any(
                self._get_ingredient_dos(line.product_id, ingredient_reqs)
                < self.po_critical_dos
                for line in pending_lines
            )

            # Release conditions
            window_expired = (current_day - window_start) >= self.po_window_days
            weight_met = total_weight >= self.po_min_weight_kg

            if has_critical or window_expired or weight_met:
                # Consolidate into single PO with multiple lines
                consolidated_po = self._consolidate_po_lines(
                    current_day, supplier_id, plant_id, pending_lines
                )
                released_pos.append(consolidated_po)

                # Clear pending for this supplier
                del self._pending_po_lines[supplier_key]
                del self._po_window_start[supplier_key]

        return released_pos

    def _create_immediate_pos(
        self,
        current_day: int,
        candidate_lines: list[tuple[str, str, str, float, float]],
    ) -> list[Order]:
        """Create immediate POs without consolidation (legacy behavior)."""
        purchase_orders: list[Order] = []

        for i, (plant_id, supplier_id, ing_id, qty, _dos) in enumerate(candidate_lines):
            order_id = f"PO-ING-{current_day:03d}-{i:06d}"
            purchase_order = Order(
                id=order_id,
                source_id=supplier_id,
                target_id=plant_id,
                creation_day=current_day,
                lines=[OrderLine(ing_id, qty)],
                status="OPEN",
            )
            purchase_orders.append(purchase_order)

        return purchase_orders

    def _consolidate_po_lines(
        self,
        current_day: int,
        supplier_id: str,
        plant_id: str,
        lines: list[OrderLine],
    ) -> Order:
        """Merge multiple order lines into a single consolidated PO."""
        # Merge duplicate ingredients by summing quantities
        merged: dict[str, float] = {}
        for line in lines:
            merged[line.product_id] = merged.get(line.product_id, 0.0) + line.quantity

        consolidated_lines = [
            OrderLine(prod_id, qty) for prod_id, qty in merged.items()
        ]

        order_id = f"PO-CONS-{current_day:03d}-{supplier_id[-3:]}"
        return Order(
            id=order_id,
            source_id=supplier_id,
            target_id=plant_id,
            creation_day=current_day,
            lines=consolidated_lines,
            status="OPEN",
        )

    def _get_product_weight(self, product_id: str) -> float:
        """Get weight per case for a product (kg)."""
        product = self.world.products.get(product_id)
        if product and product.weight_kg:
            return product.weight_kg
        # Default: 10 kg per case for ingredients
        return 10.0

    def _get_ingredient_dos(
        self, product_id: str, ingredient_reqs: np.ndarray
    ) -> float:
        """Get current days of supply for an ingredient across all plants."""
        p_idx = self.state.product_id_to_idx.get(product_id)
        if p_idx is None:
            return 999.0

        # Sum inventory across all plants
        total_inv = 0.0
        for plant_id in self._plant_ids:
            total_inv += self.state.get_inventory(plant_id, product_id)

        daily_req = float(ingredient_reqs[p_idx]) * len(self._plant_ids)
        if daily_req > 0:
            return total_inv / daily_req
        return 999.0

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

        # Round-robin based on per-category counter within eligible plants
        # Bug fix (v0.35.3): Use a separate counter per category to ensure even
        # distribution across plants. Previously, all products in a category
        # went to the same plant because _po_counter wasn't incremented until
        # _generate_po_id was called (much later in campaign batching).
        if not hasattr(self, "_plant_selection_counters"):
            self._plant_selection_counters: dict[str, int] = {}

        counter_key = cat_name
        if counter_key not in self._plant_selection_counters:
            self._plant_selection_counters[counter_key] = 0

        plant_idx = self._plant_selection_counters[counter_key] % len(eligible_plants)
        self._plant_selection_counters[counter_key] += 1

        return eligible_plants[plant_idx]

    def _generate_po_id(self, current_day: int) -> str:
        """Generate unique Production Order ID."""
        self._po_counter += 1
        return f"PO-{current_day:03d}-{self._po_counter:06d}"
