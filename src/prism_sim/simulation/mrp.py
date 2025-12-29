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

    def _cache_node_info(self) -> None:
        """Cache RDC and Plant node IDs for efficient lookups."""
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.DC:
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
        daily_demand: np.ndarray,
        active_production_orders: list[ProductionOrder],
    ) -> list[ProductionOrder]:
        """
        Generate Production Orders based on RDC inventory and demand.

        Uses Inventory Position:
        IP = On Hand + On Order (In-Transit) + In-Production (WIP/Planned)

        Args:
            current_day: Current simulation day
            daily_demand: Demand tensor [nodes, products] for demand estimation
            active_production_orders: Currently active/planned production orders

        Returns:
            List of Production Orders to be processed by TransformEngine
        """
        production_orders: list[ProductionOrder] = []

        # 1. Calculate In-Production qty per product
        # FIX: Only count production that will be ready within the planning horizon (look-ahead)
        lookahead_horizon = self.reorder_point_days  # Use ROP window as the relevant horizon
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

            # Estimate average daily demand for this product
            avg_daily_demand = self._estimate_demand(p_idx, daily_demand)

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
                    # Round up to minimum batch size
                    order_qty = max(net_requirement, self.min_production_qty)

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

        return production_orders

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
        daily_demand: np.ndarray,
    ) -> list[Order]:
        """
        Generate Purchase Orders for ingredients at plants using Vectorized MRP.

        1. Estimate Plant Demand Share (D_plant)
        2. Calculate Ingredient Requirement (Req = D_plant @ R)
        3. Check Inventory Position vs ROP
        4. Generate Orders
        """
        purchase_orders: list[Order] = []
        if not self._plant_ids:
            return purchase_orders

        # 1. Estimate Aggregate Demand per Product
        # Shape: [n_products]
        total_demand = np.sum(daily_demand, axis=0)
        
        # Avoid zero demand issues for initial priming/cold start
        # Use a minimum demand floor (e.g. 1.0) to ensure we stock something
        total_demand = np.maximum(total_demand, 1.0)

        # Distribute to plants (Fair Share assumption for replenishment planning)
        # Future: Use historical production share per plant
        n_plants = len(self._plant_ids)
        plant_demand_share = total_demand / n_plants

        # 2. Calculate Ingredient Requirements Vector
        # Req[j] = Sum(PlantDemand[i] * R[i, j])
        # Vector-Matrix multiplication: d @ R
        ingredient_reqs = plant_demand_share @ self.state.recipe_matrix

        # 3. Calculate Targets & ROPs
        # Target Inventory = Daily Req * Target Days
        target_levels = ingredient_reqs * self.target_vector
        rop_levels = ingredient_reqs * self.rop_vector

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
                qty_to_order = max(qty_needed, 100.0) # Simple MOQ

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
        # Special case for SPOF specialty ingredient
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        spof_config = mfg_config.get("spof", {})
        if ing_id == spof_config.get("ingredient_id"):
            val = spof_config.get("primary_supplier_id")
            return str(val) if val else None

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
