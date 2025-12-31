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
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        # Extract manufacturing config
        mrp_config = config.get("simulation_parameters", {}).get("manufacturing", {})
        self.target_days_supply = mrp_config.get("target_days_supply", 14.0)
        self.reorder_point_days = mrp_config.get("reorder_point_days", 7.0)
        self.min_production_qty = mrp_config.get("min_production_qty", 100.0)
        self.min_ingredient_moq = mrp_config.get("min_ingredient_moq", 100.0)
        self.production_lead_time = mrp_config.get("production_lead_time_days", 3)
        
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
        
        # Demand history for moving average [Products]
        # v0.15.6: Extended from 7 to 14 days for smoother signal
        self.demand_history = np.zeros((14, self.state.n_products), dtype=np.float64)
        self._history_ptr = 0

        # C.1 FIX: Cache expected daily demand for fallback (prevents death spiral)
        self._build_expected_demand_vector()

        # C.5 FIX: Production order history for smoothing (reduces volatility)
        # v0.15.6: Extended from 7 to 14 days
        self.production_order_history = np.zeros(14, dtype=np.float64)
        self._prod_hist_ptr = 0

        # v0.15.6: Velocity tracking - detect declining demand trends
        self._week1_demand_sum = 0.0
        self._week2_demand_sum = 0.0

        # Production Order counter for unique IDs
        self._po_counter = 0

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
        C.1 FIX: Build expected daily demand vector from config.

        Used as fallback when shipment signals collapse to prevent death spiral.
        Expected demand = base_daily_demand * n_stores (per product category).
        """
        demand_config = self.config.get("simulation_parameters", {}).get("demand", {})
        cat_profiles = demand_config.get("category_profiles", {})

        # Count stores (nodes that consume - typically stores/DCs)
        n_stores = sum(
            1 for node in self.world.nodes.values()
            if node.type == NodeType.STORE
        )
        # Fallback if no stores found
        if n_stores == 0:
            n_stores = 100

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
    ) -> list[ProductionOrder]:
        """
        Generate Production Orders based on RDC inventory and lumpy shipment signals.
        """
        production_orders: list[ProductionOrder] = []

        # 1. Update Demand History with daily shipment volume (The "Lumpy" Signal)
        self._update_demand_history(current_day, rdc_shipments)

        # Calculate Moving Average Demand
        avg_daily_demand_vec = np.mean(self.demand_history, axis=0)

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
            # Condition 1: Signal below 40% of expected (raised from 10%)
            if total_signal < expected_total * 0.4:
                use_fallback = True
            # Condition 2: Velocity declining - week1 < 60% of week2 (rapid decline)
            elif self._week2_demand_sum > 0 and self._week1_demand_sum < self._week2_demand_sum * 0.6:
                use_fallback = True

        if use_fallback:
            # Signal has collapsed or declining - use maximum of actual signal and expected
            avg_daily_demand_vec = np.maximum(avg_daily_demand_vec, self.expected_daily_demand)

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
            if dos_position < self.rop_vector[p_idx]:
                # Calculate quantity needed to reach target days supply
                target_inventory = avg_daily_demand * self.target_vector[p_idx]
                net_requirement = target_inventory - inventory_position

                if net_requirement > 0:
                    # Increase minimum batch size to amplify Bullwhip
                    order_qty = max(net_requirement, self.min_production_qty * 2.0)

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

        # v0.15.6: Minimum production floor - never drop below 30% of expected
        # This prevents complete production shutdown when demand signal dampens
        total_orders_today = sum(po.quantity_cases for po in production_orders)
        expected_production = np.sum(self.expected_daily_demand)
        min_production_floor = expected_production * 0.3

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
        # Cap total daily orders at 1.5x rolling average (after warmup)
        avg_recent = np.mean(self.production_order_history)

        if avg_recent > 0 and total_orders_today > avg_recent * 1.5:
            # Scale down all orders proportionally
            scale_factor = (avg_recent * 1.5) / total_orders_today
            for po in production_orders:
                po.quantity_cases = po.quantity_cases * scale_factor

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

        # 1. Calculate production signal from active production orders
        # This is what we're ACTUALLY producing, not downstream demand
        # Shape: [n_products]
        production_by_product = np.zeros(self.state.n_products, dtype=np.float64)
        for po in active_production_orders:
            p_idx = self.state.product_id_to_idx.get(po.product_id)
            if p_idx is not None:
                production_by_product[p_idx] += po.quantity_cases

        # Use 7-day average of production history as signal (smoothed)
        # Fall back to expected demand during cold start
        avg_production = np.mean(self.production_order_history)
        if avg_production > 0:
            # Scale production_by_product to daily average
            # (production orders may span multiple days)
            daily_production = production_by_product / max(1, len(active_production_orders) // 10 + 1)
        else:
            # Cold start: use expected demand as floor
            daily_production = self.expected_daily_demand.copy()

        # Ensure minimum floor to prevent zero-ordering
        daily_production = np.maximum(daily_production, self.expected_daily_demand * 0.5)

        # v0.15.4: Cap daily production estimate to prevent bullwhip-driven explosion
        # Max ingredient ordering = 2x expected demand (reasonable buffer)
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
        # Max order per ingredient per day = daily requirement × target days × 2
        max_order_per_ingredient = ingredient_reqs * self.target_vector * 2.0

        # 4. Build Pipeline Vector (In-Transit to Plants)
        # Shape: [n_nodes, n_products] - but we only care about plants
        pipeline = np.zeros((self.state.n_nodes, self.state.n_products), dtype=np.float64)
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
                    po = Order(
                        id=order_id,
                        source_id=supplier_id,
                        target_id=plant_id,
                        creation_day=current_day,
                        lines=[OrderLine(ing_id, float(qty_to_order))],
                        status="OPEN",
                    )
                    purchase_orders.append(po)

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