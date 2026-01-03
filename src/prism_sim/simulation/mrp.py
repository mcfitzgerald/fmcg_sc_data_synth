"""
MRP Engine: Translates Distribution Requirements (DRP) into Production Orders.

[Task 5.1] [Intent: 4. Architecture - Phase 2: The Time Loop]

This module monitors RDC inventory levels and generates Production Orders
for Plants when stock falls below reorder points.
"""

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
        self._classify_products_abc()

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

    def _classify_products_abc(self) -> None:
        """
        Classify products into A/B/C categories based on expected demand volume.
        
        Phase 2 ABC Prioritization:
        - A-items (Top 80% volume): 1.2x ROP multiplier (Prioritize availability)
        - B-items (Next 15% volume): 1.0x ROP multiplier (Standard)
        - C-items (Bottom 5% volume): 0.8x ROP multiplier (Deprioritize/Just-in-Time)
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

        # Calculate total volume and sort indices
        total_volume = np.sum(self.expected_daily_demand)
        if total_volume == 0:
            return

        sorted_indices = np.argsort(self.expected_daily_demand)[::-1]
        cumulative_volume = np.cumsum(self.expected_daily_demand[sorted_indices])

        # Determine cutoffs
        # side='right' ensures boundary items are included in the higher priority category
        idx_a = np.searchsorted(
            cumulative_volume, total_volume * thresh_a, side='right'
        )
        idx_b = np.searchsorted(
            cumulative_volume, total_volume * thresh_b, side='right'
        )

        # Apply multipliers
        # A-items
        self.abc_rop_multiplier[sorted_indices[:idx_a]] = mult_a
        # B-items
        self.abc_rop_multiplier[sorted_indices[idx_a:idx_b]] = mult_b
        # C-items
        self.abc_rop_multiplier[sorted_indices[idx_b:]] = mult_c

        # Log Alignment Verification
        n_a = idx_a
        n_b = idx_b - idx_a
        n_c = len(sorted_indices) - idx_b

        # Filter out zero-demand items from count (usually ingredients)
        # We only care about products with demand
        active_indices = [
            i for i in sorted_indices if self.expected_daily_demand[i] > 0
        ]
        n_active = len(active_indices)

        # Recalculate counts for active items only for clearer logging
        # (This is just for logging display, the multipliers are already applied)
        print(
            f"MRPEngine: Initialized ABC Classification "
            f"(Total Vol: {total_volume:,.0f})"
        )
        print(
            f"  A-Items (Top {thresh_a*100:.0f}%): {n_a} SKUs "
            f"(Multiplier {mult_a}x)"
        )
        print(
            f"  B-Items (Next {(thresh_b-thresh_a)*100:.0f}%): {n_b} SKUs "
            f"(Multiplier {mult_b}x)"
        )
        print(f"  C-Items (Tail): {n_c} SKUs (Multiplier {mult_c}x)")

    def _calculate_max_daily_capacity(self) -> float:
        """
        Calculate maximum daily production capacity across all plants.

        v0.19.8: Used to cap ingredient ordering at plant capacity,
        breaking the feedback loop where low historical production
        led to under-ordering ingredients.

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

        return total_capacity

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

    def generate_production_orders(
        self,
        current_day: int,
        rdc_shipments: list[Shipment],  # FIX 3: Accept Shipments instead of Orders
        active_production_orders: list[ProductionOrder],
        pos_demand: np.ndarray | None = None,  # v0.19.1: POS demand floor
    ) -> list[ProductionOrder]:
        """
        Generate Production Orders based on RDC inventory and demand signals.

        v0.19.1: Added POS demand as a floor to prevent demand signal collapse.
        When the order-based demand signal declines (because downstream is starving),
        we use actual consumer demand (POS) to maintain production levels.
        """
        production_orders: list[ProductionOrder] = []

        # 1. Update Demand History with daily shipment volume (The "Lumpy" Signal)
        self._update_demand_history(current_day, rdc_shipments)

        # Calculate Moving Average Demand from order/shipment history
        avg_daily_demand_vec = np.mean(self.demand_history, axis=0)

        # v0.19.1: Use POS demand as floor for demand signal
        # This is the TRUE consumer demand, not constrained by inventory availability
        # Sum POS across all nodes to get network-wide demand per product
        if pos_demand is not None:
            pos_demand_by_product = np.sum(pos_demand, axis=0)
            # Use maximum of order-based signal and POS demand
            avg_daily_demand_vec = np.maximum(
                avg_daily_demand_vec, pos_demand_by_product
            )

        # v0.15.6: Calculate demand velocity (week-over-week trend)
        # Detect declining trends before full collapse
        week1_avg = np.mean(self.demand_history[:7], axis=0)
        week2_avg = np.mean(self.demand_history[7:], axis=0)
        self._week1_demand_sum = float(np.sum(week1_avg))
        self._week2_demand_sum = float(np.sum(week2_avg))

        # C.1 FIX: Fallback to prevent death spiral
        # v0.15.6: Raised threshold from 10% to 40% AND added velocity check
        # When shipment signal is low OR declining rapidly, use expected demand floor
        total_signal = np.sum(avg_daily_demand_vec)
        expected_total = np.sum(self.expected_daily_demand)

        use_fallback = False
        if expected_total > 0:
            # Condition 1: Signal below threshold of expected
            if total_signal < expected_total * self.demand_signal_collapse_pct:
                use_fallback = True
            # Condition 2: Velocity declining - week1 < threshold of week2 (rapid decline)
            elif (
                self._week2_demand_sum > 0
                and self._week1_demand_sum
                < self._week2_demand_sum * self.velocity_trend_threshold_pct
            ):
                use_fallback = True

        if use_fallback:
            # Signal has collapsed or declining - use maximum of actual signal and expected
            avg_daily_demand_vec = np.maximum(
                avg_daily_demand_vec, self.expected_daily_demand
            )

        # 2. Calculate In-Production qty per product
        lookahead_horizon = self.reorder_point_days
        in_production_qty: dict[str, float] = {}
        for po in active_production_orders:
            if po.status != ProductionOrderStatus.COMPLETE:
                # Only count if due within horizon
                if po.due_day <= current_day + lookahead_horizon:
                    # Count remaining qty
                    remaining = po.quantity_cases - po.produced_quantity
                    in_production_qty[po.product_id] = (
                        in_production_qty.get(po.product_id, 0.0) + remaining
                    )

        # 2. Calculate In-Transit qty per product (to RDCs)
        in_transit_qty: dict[str, float] = {}
        for shipment in self.state.active_shipments:
            # Only count shipments heading to an RDC
            if shipment.target_id in self._rdc_ids:
                for line in shipment.lines:
                    in_transit_qty[line.product_id] = (
                        in_transit_qty.get(line.product_id, 0.0) + line.quantity
                    )

        # 3. Process each finished product
        for product_id in self._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue

            # Calculate total Inventory Position
            inventory_position = self._calculate_inventory_position(
                product_id, in_transit_qty, in_production_qty
            )

            # Use Moving Average Demand
            avg_daily_demand = max(float(avg_daily_demand_vec[p_idx]), 1.0)

            # Calculate days of supply based on Inventory Position
            if avg_daily_demand > 0:
                dos_position = inventory_position / avg_daily_demand
            else:
                dos_position = float("inf")

            # Check if we need to order
            # v0.19.3: Apply ABC multiplier to ROP (Phase 2)
            # A-items get higher ROP (earlier ordering), C-items get lower
            effective_rop = self.rop_vector[p_idx] * self.abc_rop_multiplier[p_idx]

            if dos_position < effective_rop:
                # Calculate quantity needed to reach target days supply
                target_inventory = avg_daily_demand * self.target_vector[p_idx]
                net_requirement = target_inventory - inventory_position

                if net_requirement > 0:
                    # v0.19.2: Use demand-proportional minimum batch size
                    # instead of fixed minimum to prevent SLOB accumulation.
                    # Min batch = max of:
                    #   1. Net requirement (what we actually need)
                    #   2. 7 days of demand (cover lead time)
                    #   3. Absolute floor of 1000 cases (avoid tiny batches)
                    demand_based_min = avg_daily_demand * 7.0  # 7 days coverage

                    mfg_config = self.config.get(
                        "simulation_parameters", {}
                    ).get("manufacturing", {})
                    absolute_min = float(
                        mfg_config.get("min_batch_size_absolute", 1000.0)
                    )  # Minimum viable batch
                    order_qty = max(
                        net_requirement, demand_based_min, absolute_min
                    )

                    # Assign to a plant (simple round-robin for now)
                    plant_id = self._select_plant(product_id)

                    # Create Production Order
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

        # v0.15.6: Minimum production floor - never drop below configured % of expected
        # This prevents complete production shutdown when demand signal dampens
        total_orders_today = sum(po.quantity_cases for po in production_orders)
        expected_production = np.sum(self.expected_daily_demand)
        min_production_floor = expected_production * self.production_floor_pct

        if total_orders_today < min_production_floor and expected_production > 0:
            # Production is too low - boost up to minimum floor
            # Create additional orders to fill the gap
            shortfall = min_production_floor - total_orders_today
            # Distribute shortfall across existing orders or create new ones
            if production_orders:
                # Scale up existing orders proportionally
                scale_factor = min_production_floor / max(total_orders_today, 1.0)
                for po in production_orders:
                    po.quantity_cases = po.quantity_cases * scale_factor
                total_orders_today = min_production_floor
            else:
                # No orders exist - create minimum orders for top products
                # Select top products by expected demand
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
                total_orders_today = sum(po.quantity_cases for po in production_orders)

        # C.5 FIX: Smooth production orders to reduce volatility
        # v0.19.8: Use max(history, expected) as baseline to prevent feedback loop.
        # Previously, using only avg_recent created a death spiral where low
        # production → lower cap → even lower production.
        avg_recent = np.mean(self.production_order_history)
        expected_production = np.sum(self.expected_daily_demand)

        # Use the higher of recent history or expected as the baseline
        # This ensures we can always scale up to meet expected demand
        smoothing_baseline = max(avg_recent, expected_production)

        if smoothing_baseline > 0 and total_orders_today > smoothing_baseline * 1.5:
            # Scale down all orders proportionally (prevent bullwhip spikes)
            scale_factor = (smoothing_baseline * 1.5) / total_orders_today
            for po in production_orders:
                po.quantity_cases = float(po.quantity_cases * scale_factor)

        # Update production history with ACTUAL (post-scaled) total
        actual_total = sum(po.quantity_cases for po in production_orders)
        self.production_order_history[self._prod_hist_ptr] = actual_total
        # v0.15.6: Extended history from 7 to 14 days
        self._prod_hist_ptr = (self._prod_hist_ptr + 1) % 14

        return production_orders

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
        v0.15.9: Record order-based demand signal (pre-allocation).

        This captures the TRUE demand signal - what customer DCs requested from
        RDCs, before allocation constrains it. Used to prevent demand signal
        attenuation when DCs are short on inventory.

        Unlike shipment-based demand (what was shipped), order-based demand
        reflects what was actually needed. This creates a stronger, more accurate
        signal for production planning.

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

        # Blend with existing history - use max to not lose shipment signal
        # This ensures we capture the higher of (orders, shipments) as demand
        current_slot = (self._history_ptr - 1) % 14
        self.demand_history[current_slot] = np.maximum(
            self.demand_history[current_slot], daily_vol
        )

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
