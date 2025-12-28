"""
MRP Engine: Translates Distribution Requirements (DRP) into Production Orders.

[Task 5.1] [Intent: 4. Architecture - Phase 2: The Time Loop]

This module monitors RDC inventory levels and generates Production Orders
for Plants when stock falls below reorder points.
"""

from typing import Any

import numpy as np

from prism_sim.network.core import NodeType, ProductionOrder, ProductionOrderStatus, Order, OrderLine
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

    def generate_production_orders(
        self, 
        current_day: int, 
        daily_demand: np.ndarray,
        active_production_orders: list[ProductionOrder]
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
        in_production_qty: dict[str, float] = {}
        for po in active_production_orders:
            if po.status != ProductionOrderStatus.COMPLETE:
                # Count remaining qty
                remaining = po.quantity_cases - po.produced_quantity
                in_production_qty[po.product_id] = in_production_qty.get(po.product_id, 0.0) + remaining

        # 2. Calculate In-Transit qty per product (to RDCs)
        in_transit_qty: dict[str, float] = {}
        for shipment in self.state.active_shipments:
            # Only count shipments heading to an RDC
            if shipment.target_id in self._rdc_ids:
                for line in shipment.lines:
                    in_transit_qty[line.product_id] = in_transit_qty.get(line.product_id, 0.0) + line.quantity

        # 3. Process each finished product
        for product_id in self._finished_product_ids:
            p_idx = self.state.product_id_to_idx.get(product_id)
            if p_idx is None:
                continue

            # Sum On-Hand inventory across RDCs
            on_hand_inventory = 0.0
            for rdc_id in self._rdc_ids:
                n_idx = self.state.node_id_to_idx.get(rdc_id)
                if n_idx is not None:
                    on_hand_inventory += float(self.state.inventory[n_idx, p_idx])

            # Calculate total Inventory Position
            inventory_position = (
                on_hand_inventory + 
                in_transit_qty.get(product_id, 0.0) + 
                in_production_qty.get(product_id, 0.0)
            )

            # Estimate average daily demand for this product
            avg_daily_demand = self._estimate_demand(p_idx, daily_demand)

            # Calculate days of supply based on Inventory Position
            if avg_daily_demand > 0:
                dos_position = inventory_position / avg_daily_demand
            else:
                dos_position = float("inf")

            # Check if we need to order
            if dos_position < self.reorder_point_days:
                # Calculate quantity needed to reach target days supply
                target_inventory = avg_daily_demand * self.target_days_supply
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
    ) -> list[Order]:
        """
        Generate Purchase Orders for ingredients at plants.
        
        Uses Inventory Position for ingredients:
        IP = On Hand (Plant) + On Order (In-Transit from Supplier)
        """
        purchase_orders: list[Order] = []
        
        # 1. Map ingredients to plants that need them (from recipes)
        ingredient_ids: set[str] = set()
        for recipe in self.world.recipes.values():
            ingredient_ids.update(recipe.ingredients.keys())
            
        # 2. Calculate In-Transit qty for ingredients (to Plants)
        in_transit_qty: dict[tuple[str, str], float] = {} # (plant_id, ing_id) -> qty
        for shipment in self.state.active_shipments:
            # Check if target is a plant
            if shipment.target_id in self._plant_ids:
                for line in shipment.lines:
                    key = (shipment.target_id, line.product_id)
                    in_transit_qty[key] = in_transit_qty.get(key, 0.0) + line.quantity
                    
        # 3. Process each plant and each ingredient
        for plant_id in self._plant_ids:
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            if plant_idx is None:
                continue
                
            for ing_id in ingredient_ids:
                ing_idx = self.state.product_id_to_idx.get(ing_id)
                if ing_idx is None:
                    continue
                    
                # On-Hand at Plant
                on_hand = float(self.state.inventory[plant_idx, ing_idx])
                
                # Inventory Position
                ip = on_hand + in_transit_qty.get((plant_id, ing_id), 0.0)
                
                # For ingredients, we use manufacturing reorder points
                # Reorder if IP < ROP
                # ROP = Demand * reorder_point_days
                # But what is ingredient demand? 
                # Proxy: 50% of FG daily demand * BOM qty (very rough)
                # Better: Use target_days_supply as a fixed buffer for now since it's "Warm start"
                # Actually, let's use the manufacturing reorder point from config
                
                # Check reorder point
                # If IP < reorder_point_days * (typical demand)
                # Let's assume typical ingredient demand is 100k units/day for whole system
                # (matching our initial priming level of 10M / 100 days)
                
                # We'll use a fixed reorder point for now to avoid complexity of explosion
                # ROP = 1,000,000 units
                # Target = 5,000,000 units
                rop = 1_000_000.0
                target = 5_000_000.0
                
                if ip < rop:
                    qty_to_order = target - ip
                    
                    # Find a supplier for this ingredient for this plant
                    supplier_id = self._find_supplier_for_ingredient(plant_id, ing_id)
                    
                    if supplier_id:
                        # Create Purchase Order (as an Order object)
                        order_id = f"PO-ING-{current_day:03d}-{len(purchase_orders):06d}"
                        po = Order(
                            id=order_id,
                            source_id=supplier_id,
                            target_id=plant_id,
                            creation_day=current_day,
                            lines=[OrderLine(ing_id, qty_to_order)],
                            status="OPEN"
                        )
                        purchase_orders.append(po)
                        
        return purchase_orders

    def _find_supplier_for_ingredient(self, plant_id: str, ing_id: str) -> str | None:
        """Find a supplier that provides the ingredient to the plant."""
        # Special case for SPOF specialty ingredient
        mfg_config = self.config.get("simulation_parameters", {}).get("manufacturing", {})
        spof_config = mfg_config.get("spof", {})
        if ing_id == spof_config.get("ingredient_id"):
            return spof_config.get("primary_supplier_id")
            
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
