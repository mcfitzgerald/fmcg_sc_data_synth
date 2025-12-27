"""
Transform Engine: Processes Production Orders with Physics constraints.

[Task 5.2] [Intent: 4. Architecture - Production Physics (L6)]
[Task 5.3] [Intent: 2. Supply Chain Resilience - SPOF Simulation]

This module enforces:
- Finite Capacity: run_rate_cases_per_hour from Recipe
- Changeover Penalty: Little's Law friction when switching products
- Raw Material Constraints: Production fails without ingredients (SPOF)
- Deterministic Batch Tracking: B-2024-RECALL-001 scheduling
"""

from dataclasses import dataclass, field
from typing import Any

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
        self.recall_batch_id = self.recall_scenario.get(
            "batch_id", "B-2024-RECALL-001"
        )
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
        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.PLANT:
                self._plant_states[node_id] = PlantState(
                    plant_id=node_id,
                    remaining_capacity_hours=self.hours_per_day,
                )

    def process_production_orders(
        self, orders: list[ProductionOrder], current_day: int
    ) -> tuple[list[ProductionOrder], list[Batch]]:
        """
        Process Production Orders for the current day.

        Args:
            orders: List of Production Orders to process
            current_day: Current simulation day

        Returns:
            Tuple of (updated_orders, new_batches)
        """
        new_batches: list[Batch] = []

        # Reset daily capacity for all plants
        for plant_state in self._plant_states.values():
            plant_state.remaining_capacity_hours = self.hours_per_day

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

        return sorted_orders, new_batches

    def _process_single_order(
        self, order: ProductionOrder, current_day: int
    ) -> Batch | None:
        """
        Process a single Production Order.

        Returns:
            Batch if production completed, None otherwise
        """
        plant_state = self._plant_states.get(order.plant_id)
        if plant_state is None:
            return None

        recipe = self.world.recipes.get(order.product_id)
        if recipe is None:
            # No recipe for this product - skip
            return None

        # Check raw material availability (SPOF logic)
        material_available, _material_shortage = self._check_material_availability(
            order.plant_id, order.product_id, order.quantity_cases
        )

        if not material_available:
            # Cannot produce - material shortage (SPOF triggered)
            order.status = ProductionOrderStatus.PLANNED  # Keep waiting
            return None

        # Calculate production time needed
        production_time_hours = order.quantity_cases / recipe.run_rate_cases_per_hour

        # Check for changeover penalty
        changeover_time = 0.0
        if (
            plant_state.last_product_id is not None
            and plant_state.last_product_id != order.product_id
        ):
            changeover_time = recipe.changeover_time_hours

        total_time_needed = production_time_hours + changeover_time

        # Check capacity
        if plant_state.remaining_capacity_hours < total_time_needed:
            # Partial production or defer to next day
            available_production_time = max(
                0.0, plant_state.remaining_capacity_hours - changeover_time
            )

            if available_production_time <= 0:
                # No capacity today - wait for tomorrow
                return None

            # Partial production
            producible_qty = available_production_time * recipe.run_rate_cases_per_hour
            actual_qty = min(
                producible_qty, order.quantity_cases - order.produced_quantity
            )
        else:
            actual_qty = order.quantity_cases - order.produced_quantity

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

        # Check if order is complete
        if order.produced_quantity >= order.quantity_cases:
            order.status = ProductionOrderStatus.COMPLETE
            order.actual_end_day = current_day

            # Create batch
            batch = self._create_batch(order, current_day, actual_qty)

            # Add produced goods to plant inventory
            self._add_to_inventory(
                order.plant_id, order.product_id, order.quantity_cases
            )

            return batch

        return None

    def _check_material_availability(
        self, plant_id: str, product_id: str, quantity: float
    ) -> tuple[bool, dict[str, float]]:
        """
        Check if raw materials are available for production.

        Implements SPOF logic for specialty ingredients.

        Returns:
            Tuple of (is_available, shortage_dict)
        """
        recipe = self.world.recipes.get(product_id)
        if recipe is None:
            return True, {}

        shortage: dict[str, float] = {}

        for ingredient_id, qty_per_case in recipe.ingredients.items():
            required_qty = qty_per_case * quantity

            # Get current ingredient inventory at plant
            plant_idx = self.state.node_id_to_idx.get(plant_id)
            ing_idx = self.state.product_id_to_idx.get(ingredient_id)

            if plant_idx is None or ing_idx is None:
                # Can't find indices - assume unlimited for non-tracked ingredients
                continue

            available_qty = float(self.state.inventory[plant_idx, ing_idx])

            if available_qty < required_qty:
                shortage[ingredient_id] = required_qty - available_qty

        return len(shortage) == 0, shortage

    def _consume_materials(
        self, plant_id: str, product_id: str, quantity: float
    ) -> dict[str, float]:
        """
        Consume raw materials for production (Mass Balance).

        Returns:
            Dictionary of consumed materials {ingredient_id: quantity}
        """
        recipe = self.world.recipes.get(product_id)
        if recipe is None:
            return {}

        consumed: dict[str, float] = {}

        for ingredient_id, qty_per_case in recipe.ingredients.items():
            consume_qty = qty_per_case * quantity

            plant_idx = self.state.node_id_to_idx.get(plant_id)
            ing_idx = self.state.product_id_to_idx.get(ingredient_id)

            if plant_idx is not None and ing_idx is not None:
                self.state.inventory[plant_idx, ing_idx] -= consume_qty
                consumed[ingredient_id] = consume_qty

        return consumed

    def _add_to_inventory(
        self, plant_id: str, product_id: str, quantity: float
    ) -> None:
        """Add produced goods to plant inventory."""
        plant_idx = self.state.node_id_to_idx.get(plant_id)
        prod_idx = self.state.product_id_to_idx.get(product_id)

        if plant_idx is not None and prod_idx is not None:
            self.state.inventory[plant_idx, prod_idx] += quantity

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
