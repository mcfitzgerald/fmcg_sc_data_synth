"""
Transform Engine: Processes Production Orders with Physics constraints.

[Task 5.2] [Intent: 4. Architecture - Production Physics (L6)]
[Task 5.3] [Intent: 2. Supply Chain Resilience - SPOF Simulation]
[Task 8.3] [Intent: World Builder Overhaul - Vectorized Execution]

This module enforces:
- Finite Capacity: run_rate_cases_per_hour from Recipe
- Changeover Penalty: Little's Law friction when switching products
- Raw Material Constraints: Checks ingredient availability (SPOF logic)
- Mass Balance: Ingredient consumption equals BOM * production quantity
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prism_sim.network.core import (
    Batch,
    BatchStatus,
    NodeType,
    ProductionOrder,
    ProductionOrderStatus,
)
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


@dataclass
class PlantState:
    """Tracks the production state of a plant for changeover logic."""

    plant_id: str
    last_product_id: str | None = None
    remaining_capacity_hours: float = 0.0
    max_capacity_hours: float = 0.0  # Effective daily capacity
    # Tracks in-progress production orders
    active_orders: list[ProductionOrder] = field(default_factory=list)


class TransformEngine:
    """
    Production physics engine that processes Production Orders.

    Enforces:
    - Finite Capacity: Production limited by run_rate_cases_per_hour
    - Changeover Penalty: Time lost when switching product types
    - Raw Material Constraints: Checks ingredient availability (SPOF logic)
    - Mass Balance: Ingredient consumption equals BOM * production quantity
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        # Extract manufacturing config
        mfg_config = config.get("simulation_parameters", {}).get("manufacturing", {})
        self.hours_per_day = mfg_config.get("production_hours_per_day", 16.0)
        self.backup_supplier_cost_premium = mfg_config.get(
            "backup_supplier_cost_premium", 0.25
        )
        self.default_yield_percent = mfg_config.get("default_yield_percent", 98.5)
        self.recall_batch_trigger_day = mfg_config.get("recall_batch_trigger_day", 30)

        # Recall Scenario Config
        self.recall_scenario = mfg_config.get("recall_scenario", {})
        self.recall_product_id = self.recall_scenario.get("product_id", "SKU-DET-001")
        self.recall_batch_id = self.recall_scenario.get("batch_id", "B-2024-RECALL-001")
        self.recall_status_str = self.recall_scenario.get("status", "hold")
        self.recall_notes = self.recall_scenario.get(
            "notes", "RECALL: Contaminated sorbitol detected"
        )

        # SPOF Configuration
        spof_config = mfg_config.get("spof", {})
        self.spof_ingredient_id = spof_config.get("ingredient_id", "ING-SURF-SPEC")
        self.primary_supplier_id = spof_config.get(
            "primary_supplier_id", "SUP-SURF-SPEC"
        )
        self.backup_supplier_id = spof_config.get(
            "backup_supplier_id", "SUP-SURF-BACKUP"
        )
        self.spof_warning_threshold = spof_config.get("warning_threshold", 10.0)

        # Initialize plant states
        self._plant_states: dict[str, PlantState] = {}
        self._initialize_plant_states()

        # Batch tracking
        self._batch_counter = 0
        self.batches: list[Batch] = []

        # Track the recall batch for deterministic scheduling
        self._recall_batch_created = False

    def _initialize_plant_states(self) -> None:
        """Initialize production state for each plant."""
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        plant_params = mfg_config.get("plant_parameters", {})

        # Global defaults
        global_efficiency = mfg_config.get("efficiency_factor", 0.85)
        global_downtime = mfg_config.get("unplanned_downtime_pct", 0.05)

        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.PLANT:
                # Get plant-specific overrides
                p_config = plant_params.get(node_id, {})
                efficiency = p_config.get("efficiency_factor", global_efficiency)
                downtime = p_config.get("unplanned_downtime_pct", global_downtime)

                # Calculate effective capacity hours
                # Effective = Total Hours * (1 - Downtime) * Efficiency
                # This treats efficiency as a speed multiplier, but since we can't
                # easily change run_rate per plant without complicating the recipe
                # lookup, we effectively reduce the available hours to simulate
                # slower/interrupted production.
                effective_hours = self.hours_per_day * (1.0 - downtime) * efficiency

                self._plant_states[node_id] = PlantState(
                    plant_id=node_id,
                    remaining_capacity_hours=effective_hours,
                    max_capacity_hours=effective_hours,
                )

    def process_production_orders(
        self, orders: list[ProductionOrder], current_day: int
    ) -> tuple[list[ProductionOrder], list[Batch], dict[str, float]]:
        """
        Process Production Orders for the current day.

        Args:
            orders: List of Production Orders to process
            current_day: Current simulation day

        Returns:
            Tuple of (updated_orders, new_batches, plant_oee)
        """
        new_batches: list[Batch] = []
        plant_oee: dict[str, float] = {}

        # Reset daily capacity for all plants
        for plant_state in self._plant_states.values():
            plant_state.remaining_capacity_hours = plant_state.max_capacity_hours

        # Sort orders by due date (priority)
        sorted_orders = sorted(orders, key=lambda o: o.due_day)

        for order in sorted_orders:
            if order.status == ProductionOrderStatus.COMPLETE:
                continue

            # Process order at its assigned plant
            batch = self._process_single_order(order, current_day)
            if batch is not None:
                new_batches.append(batch)
                self.batches.append(batch)

        # Calculate OEE for each plant (Utilization of Effective Capacity)
        for plant_id, plant_state in self._plant_states.items():
            used_cap = (
                plant_state.max_capacity_hours - plant_state.remaining_capacity_hours
            )
            if plant_state.max_capacity_hours > 0:
                oee = used_cap / plant_state.max_capacity_hours
            else:
                oee = 0.0
            plant_oee[plant_id] = oee

        return sorted_orders, new_batches, plant_oee

    def _process_single_order(
        self, order: ProductionOrder, current_day: int
    ) -> Batch | None:
        """
        Process a single Production Order.

        Returns:
            Batch if production completed, None otherwise
        """
        plant_state = self._plant_states.get(order.plant_id)
        recipe = self.world.recipes.get(order.product_id)

        if plant_state is None or recipe is None:
            return None

        # Calculate remaining quantity
        remaining_qty = order.quantity_cases - order.produced_quantity
        if remaining_qty <= 0:
            order.status = ProductionOrderStatus.COMPLETE
            return None

        # Calculate production time needed for remaining
        production_time_hours = remaining_qty / recipe.run_rate_cases_per_hour

        # Check for changeover penalty
        changeover_time = 0.0
        if (
            plant_state.last_product_id is not None
            and plant_state.last_product_id != order.product_id
        ):
            changeover_time = recipe.changeover_time_hours

        # How much can we actually produce today?
        available_time = plant_state.remaining_capacity_hours
        available_time_for_prod = max(0.0, available_time - changeover_time)

        if available_time_for_prod <= 0 and available_time < changeover_time:
            # Not even enough time for changeover
            return None

        max_qty_today = min(
            remaining_qty,
            available_time_for_prod * recipe.run_rate_cases_per_hour,
        )

        if max_qty_today <= 0:
            return None

        # Check raw material availability for TODAY's potential production
        material_available, _ = self._check_material_availability(
            order.plant_id, order.product_id, max_qty_today
        )

        if not material_available:
            # Cannot produce - material shortage (SPOF triggered)
            order.status = ProductionOrderStatus.PLANNED  # Keep waiting
            return None

        # If we have materials for today, proceed with production calculation
        total_time_needed = production_time_hours + changeover_time
        if plant_state.remaining_capacity_hours < total_time_needed:
            actual_qty = max_qty_today
        else:
            actual_qty = remaining_qty

        # Start production
        if order.actual_start_day is None:
            order.actual_start_day = current_day
        order.status = ProductionOrderStatus.IN_PROGRESS

        # Consume raw materials
        self._consume_materials(order.plant_id, order.product_id, actual_qty)

        # Update order progress
        order.produced_quantity += actual_qty

        # Update plant state
        time_used = changeover_time + (actual_qty / recipe.run_rate_cases_per_hour)
        plant_state.remaining_capacity_hours -= time_used
        plant_state.last_product_id = order.product_id

        # Create batch for TODAY's production
        batch = self._create_batch(order, current_day, actual_qty)

        # Add produced goods to plant inventory
        self._add_to_inventory(order.plant_id, order.product_id, actual_qty)

        # Check if order is complete
        if order.produced_quantity >= order.quantity_cases:
            order.status = ProductionOrderStatus.COMPLETE
            order.actual_end_day = current_day

        return batch

    def _check_material_availability(
        self, plant_id: str, product_id: str, quantity: float
    ) -> tuple[bool, dict[str, float]]:
        """
        Check if raw materials are available using Matrix ops.
        
        Returns: Tuple of (is_available, shortage_dict)
        """
        p_idx = self.state.product_id_to_idx.get(product_id)
        plant_idx = self.state.node_id_to_idx.get(plant_id)

        if p_idx is None or plant_idx is None:
            return True, {}

        # 1. Calculate Required: Req = Qty * BOM[p]
        bom_vector = self.state.recipe_matrix[p_idx]
        required_vector = bom_vector * quantity

        # 2. Get Available: Use ACTUAL inventory (not perceived)
        # This prevents phantom inventory from causing over-production
        available_vector = np.maximum(0, self.state.actual_inventory[plant_idx])

        # 3. Check for Shortage
        # We only care where requirement > 0
        # shortage = required - available (if required > available)
        diff = required_vector - available_vector
        shortage_mask = (diff > 0) & (required_vector > 0)

        if np.any(shortage_mask):
            # Convert shortage to dictionary for logging
            shortage_dict = {}
            shortage_indices = np.where(shortage_mask)[0]
            for idx in shortage_indices:
                ing_id = self.state.product_idx_to_id[idx]
                shortage_dict[ing_id] = float(diff[idx])
            return False, shortage_dict

        return True, {}

    def _consume_materials(
        self, plant_id: str, product_id: str, quantity: float
    ) -> dict[str, float]:
        """
        Consume raw materials using Matrix ops (Mass Balance).
        
        Returns: Dictionary of consumed materials
        """
        p_idx = self.state.product_id_to_idx.get(product_id)
        plant_idx = self.state.node_id_to_idx.get(plant_id)

        if p_idx is None or plant_idx is None:
            return {}

        # 1. Calculate Consumed: Qty * BOM[p]
        bom_vector = self.state.recipe_matrix[p_idx]
        consumed_vector = bom_vector * quantity

        # 2. Update Inventory (Direct Array Access)
        # Constrain consumption to available actual inventory to prevent negatives
        actual_available = np.maximum(0, self.state.actual_inventory[plant_idx])
        actual_consumed = np.minimum(consumed_vector, actual_available)

        # Update both inventories by the actually consumed amount
        self.state.perceived_inventory[plant_idx] -= actual_consumed
        self.state.actual_inventory[plant_idx] -= actual_consumed
        # Floor to zero - prevent floating point noise
        np.maximum(self.state.perceived_inventory[plant_idx], 0,
                   out=self.state.perceived_inventory[plant_idx])
        np.maximum(self.state.actual_inventory[plant_idx], 0,
                   out=self.state.actual_inventory[plant_idx])

        # 3. Return consumed dict (only non-zero)
        consumed_dict = {}
        # Optimization: only iterate non-zero elements
        non_zero_indices = np.where(actual_consumed > 0)[0]
        for idx in non_zero_indices:
            ing_id = self.state.product_idx_to_id[idx]
            consumed_dict[ing_id] = float(actual_consumed[idx])

        return consumed_dict

    def _add_to_inventory(
        self, plant_id: str, product_id: str, quantity: float
    ) -> None:
        """Add produced goods to plant inventory."""
        self.state.update_inventory(plant_id, product_id, quantity)

    def _create_batch(
        self, order: ProductionOrder, current_day: int, quantity: float
    ) -> Batch:
        """
        Create a Batch record for completed production.

        Handles deterministic B-2024-RECALL-001 scheduling.
        """
        # Check for deterministic recall batch (Task 5.2 constraint)
        if (
            not self._recall_batch_created
            and order.product_id == self.recall_product_id
            and current_day >= self.recall_batch_trigger_day
        ):
            self._recall_batch_created = True
            batch_id = self.recall_batch_id
            notes = self.recall_notes

            # Map config string to Enum
            try:
                status = BatchStatus(self.recall_status_str.lower())
            except ValueError:
                status = BatchStatus.HOLD
        else:
            self._batch_counter += 1
            batch_id = f"B-{current_day:03d}-{self._batch_counter:06d}"
            notes = ""
            status = BatchStatus.COMPLETE

        # Get ingredients consumed (for genealogy)
        # Re-using matrix logic might be cleaner but for single batch creation
        # using the recipe dict is fine as we need IDs anyway.
        # But we should consistent. Let's use recipe dict for now as it's O(1) lookup
        # and we need to return a dict.
        recipe = self.world.recipes.get(order.product_id)
        ingredients_consumed: dict[str, float] = {}
        if recipe:
            for ing_id, qty_per_case in recipe.ingredients.items():
                ingredients_consumed[ing_id] = qty_per_case * quantity

        return Batch(
            id=batch_id,
            production_order_id=order.id,
            product_id=order.product_id,
            plant_id=order.plant_id,
            production_day=current_day,
            quantity_cases=quantity,
            status=status,
            ingredients_consumed=ingredients_consumed,
            yield_percent=self.default_yield_percent,
            notes=notes,
        )

    def get_spof_status(self) -> dict[str, Any]:
        """
        Get the current SPOF (Single Point of Failure) status.

        Returns status of specialty ingredient availability across plants.
        """
        spof_status: dict[str, Any] = {
            "ingredient_id": self.spof_ingredient_id,
            "plants": {},
        }

        ing_idx = self.state.product_id_to_idx.get(self.spof_ingredient_id)
        if ing_idx is None:
            return spof_status

        for plant_id in self._plant_states:
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            if plant_idx is not None:
                available = float(self.state.inventory[plant_idx, ing_idx])
                spof_status["plants"][plant_id] = {
                    "available_qty": available,
                    "is_constrained": available < self.spof_warning_threshold,
                }

        return spof_status
