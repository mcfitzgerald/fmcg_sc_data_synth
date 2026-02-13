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
    ABCClass,
    Batch,
    BatchStatus,
    NodeType,
    ProductionOrder,
    ProductionOrderStatus,
)
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World


@dataclass
class LineState:
    """Tracks state of a single production line."""

    line_id: str
    plant_id: str

    # Capacity
    remaining_capacity_hours: float = 0.0
    max_capacity_hours: float = 0.0

    # Changeover tracking
    last_product_id: str | None = None

    # OEE component tracking (reset daily)
    run_hours_today: float = 0.0
    changeover_hours_today: float = 0.0
    output_cases_today: float = 0.0


@dataclass
class PlantState:
    """Tracks the production state of a plant."""

    plant_id: str
    lines: list[LineState] = field(default_factory=list)

    # Keep for convenience/aggregation
    max_capacity_hours: float = 0.0  # Sum of all lines
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
        self.hours_per_day = mfg_config.get("production_hours_per_day", 24.0)
        self.backup_supplier_cost_premium = mfg_config.get(
            "backup_supplier_cost_premium", 0.25
        )
        self.default_yield_percent = mfg_config.get("default_yield_percent", 98.5)
        self.recall_batch_trigger_day = mfg_config.get("recall_batch_trigger_day", 30)

        # Production rate multiplier - scales run_rate to match capacity/demand ratio
        # Default 1.0 means use recipe run_rate as-is
        # Higher values simulate multiple production lines or faster equipment
        self.rate_multiplier = mfg_config.get("production_rate_multiplier", 1.0)

        # Changeover time multiplier - scales recipe changeover times
        # Default 0.1 (6 min for 1h changeover) represents modern high-speed lines
        # with automated changeover. Recipe default of 0.5h becomes 3 minutes.
        self.changeover_multiplier = mfg_config.get("changeover_time_multiplier", 0.1)

        # v0.29.0: Seasonal capacity flex configuration
        # Mirrors demand seasonality so production can flex with demand:
        # - Peak demand → higher capacity (overtime, extra shifts)
        # - Trough demand → lower capacity (reduced shifts, maintenance)
        demand_config = config.get("simulation_parameters", {}).get("demand", {})
        season_cfg = demand_config.get("seasonality", {})
        self._seasonal_capacity_amplitude = season_cfg.get("capacity_amplitude", 0.0)
        self._seasonal_phase_shift = season_cfg.get("phase_shift_days", 150)
        self._seasonal_cycle_days = season_cfg.get("cycle_days", 365)

        # Recall Scenario Config
        self.recall_scenario = mfg_config.get("recall_scenario", {})
        self.recall_product_id = self.recall_scenario.get("product_id", "SKU-DET-001")
        self.recall_batch_id = self.recall_scenario.get("batch_id", "B-2024-RECALL-001")
        self.recall_status_str = self.recall_scenario.get("status", "hold")
        self.recall_notes = self.recall_scenario.get(
            "notes", "RECALL: Contaminated sorbitol detected"
        )

        # v0.19.2: Base demand matrix for production prioritization
        # Set by orchestrator after POS engine is initialized
        self._base_demand: np.ndarray | None = None

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
        self._plant_efficiency: dict[str, float] = {}  # Store efficiency for OEE calc
        self._initialize_plant_states()

        # Batch tracking
        self._batch_counter = 0
        self.batches: list[Batch] = []

        # Track the recall batch for deterministic scheduling
        self._recall_batch_created = False

    def _initialize_plant_states(self) -> None:
        """Initialize production state for each plant with lines."""
        mfg_config = self.config.get("simulation_parameters", {}).get(
            "manufacturing", {}
        )
        plant_params = mfg_config.get("plant_parameters", {})
        default_num_lines = mfg_config.get("default_num_lines", 4)

        # Global defaults
        global_efficiency = mfg_config.get("efficiency_factor", 0.85)
        global_downtime = mfg_config.get("unplanned_downtime_pct", 0.05)

        for node_id, node in self.world.nodes.items():
            if node.type == NodeType.PLANT:
                # Get plant-specific overrides
                p_config = plant_params.get(node_id, {})
                num_lines = p_config.get("num_lines", default_num_lines)
                efficiency = p_config.get("efficiency_factor", global_efficiency)
                downtime = p_config.get("unplanned_downtime_pct", global_downtime)

                # Calculate effective capacity hours PER LINE
                # Effective = Total Hours * (1 - Downtime)
                # Note: efficiency_factor is NOT applied here - it is captured
                # solely via the OEE Performance component to avoid double-counting
                hours_per_line = self.hours_per_day * (1.0 - downtime)

                # Store efficiency factor for OEE Performance calculation
                self._plant_efficiency[node_id] = efficiency

                # Create lines
                lines = []
                for i in range(num_lines):
                    line = LineState(
                        line_id=f"{node_id}-L{i + 1}",
                        plant_id=node_id,
                        remaining_capacity_hours=hours_per_line,
                        max_capacity_hours=hours_per_line,
                    )
                    lines.append(line)

                self._plant_states[node_id] = PlantState(
                    plant_id=node_id,
                    lines=lines,
                    max_capacity_hours=hours_per_line * num_lines,
                )

    def set_base_demand(self, base_demand: np.ndarray) -> None:
        """
        Set the base demand matrix for production prioritization.

        v0.19.2: Used to prioritize high-demand products over low-demand ones
        during production scheduling, breaking the SLOB accumulation pattern.
        """
        self._base_demand = base_demand
        self._classify_products_abc()

    def _get_seasonal_capacity_factor(self, day: int) -> float:
        """
        Calculate seasonal capacity multiplier for a given day.

        v0.29.0: Mirrors demand seasonality so production can flex with demand:
        - Peak demand → higher capacity (overtime, extra shifts)
        - Trough demand → lower capacity (reduced shifts, maintenance)

        Args:
            day: Current simulation day

        Returns:
            Capacity multiplier (e.g., 1.12 for +12% capacity during peak)
        """
        if self._seasonal_capacity_amplitude == 0:
            return 1.0

        return float(
            1.0
            + self._seasonal_capacity_amplitude
            * np.sin(
                2
                * np.pi
                * (day - self._seasonal_phase_shift)
                / self._seasonal_cycle_days
            )
        )

    def _classify_products_abc(self) -> None:
        """
        Classify products into A/B/C based on base demand velocity.
        Used for production capacity reservation (Phase 4).
        """
        if self._base_demand is None:
            return

        # Get config
        abc_config = (
            self.config.get("simulation_parameters", {})
            .get("agents", {})
            .get("abc_prioritization", {})
        )
        if not abc_config.get("enabled", True):
            self._product_abc_class: dict[str, str] = {}
            return

        thresh_a = abc_config.get("a_threshold_pct", 0.80)
        thresh_b = abc_config.get("b_threshold_pct", 0.95)

        # Calculate network-wide velocity per product
        velocity = np.sum(self._base_demand, axis=0)
        total_volume = np.sum(velocity)

        if total_volume == 0:
            return

        # Sort descending
        sorted_indices = np.argsort(velocity)[::-1]
        cumulative = np.cumsum(velocity[sorted_indices])

        # Determine cutoffs
        idx_a = np.searchsorted(cumulative, total_volume * thresh_a, side='right')
        idx_b = np.searchsorted(cumulative, total_volume * thresh_b, side='right')

        self._product_abc_class = {}
        for i, p_idx in enumerate(sorted_indices):
            p_id = self.state.product_idx_to_id[p_idx]
            if i < idx_a:
                self._product_abc_class[p_id] = ABCClass.A
            elif i < idx_b:
                self._product_abc_class[p_id] = ABCClass.B
            else:
                self._product_abc_class[p_id] = ABCClass.C

    def _get_abc_priority(self, product_id: str) -> int:
        """Return sort priority for product (A=1, B=2, C=3)."""
        if not hasattr(self, "_product_abc_class"):
            return 2  # Default to B (Standard)

        cls = self._product_abc_class.get(product_id, ABCClass.B)
        if cls == ABCClass.A:
            return 1
        elif cls == ABCClass.B:
            return 2
        else:
            return 3

    def _get_bom_level(self, product_id: str) -> int:
        """Return BOM level for production ordering.

        Bulk intermediates (level 1) must be produced before
        finished SKUs (level 0) within each daily cycle.
        """
        product = self.world.products.get(product_id)
        if product is None:
            return 0
        return product.bom_level

    def _select_line(self, plant_id: str, product_id: str) -> LineState | None:
        """
        Select best line for production order.

        Strategy:
        1. Prefer line already making this product (no changeover)
        2. Else, pick line with most remaining capacity
        """
        plant_state = self._plant_states.get(plant_id)
        if plant_state is None:
            return None

        best_same_product: LineState | None = None
        best_available: LineState | None = None
        best_available_hours = -1.0

        for line in plant_state.lines:
            if line.remaining_capacity_hours <= 0:
                continue

            # Prefer line already making this product
            if line.last_product_id == product_id:
                if (
                    best_same_product is None
                    or line.remaining_capacity_hours
                    > best_same_product.remaining_capacity_hours
                ):
                    best_same_product = line

            # Track best available line
            if line.remaining_capacity_hours > best_available_hours:
                best_available_hours = line.remaining_capacity_hours
                best_available = line

        return best_same_product or best_available

    def process_production_orders(
        self, orders: list[ProductionOrder], current_day: int
    ) -> tuple[
        list[ProductionOrder], list[Batch], dict[str, float], dict[str, float]
    ]:
        """
        Process Production Orders for the current day using Line logic.

        Args:
            orders: List of Production Orders to process
            current_day: Current simulation day

        Returns:
            Tuple of (updated_orders, new_batches, plant_oee, plant_teep)
        """
        new_batches: list[Batch] = []
        plant_oee: dict[str, float] = {}
        plant_teep: dict[str, float] = {}

        # v0.29.0: Apply seasonal capacity factor for flexible capacity
        seasonal_factor = self._get_seasonal_capacity_factor(current_day)

        # Reset daily capacity for all lines (with seasonal adjustment)
        for plant_state in self._plant_states.values():
            for line in plant_state.lines:
                effective_capacity = line.max_capacity_hours * seasonal_factor
                line.remaining_capacity_hours = effective_capacity
                # Reset OEE counters
                line.run_hours_today = 0.0
                line.changeover_hours_today = 0.0
                line.output_cases_today = 0.0

        # Sort orders by:
        # 1. Plant ID -> Process each plant's orders together
        # 2. BOM level (descending) -> Intermediates before SKUs
        # 3. ABC Priority (A=1 first) -> Reserve capacity for runners
        # 4. Product ID -> Group same products (minimize changeovers)
        # 5. Due Date (Earliest first)
        sorted_orders = sorted(
            orders,
            key=lambda o: (
                o.plant_id,
                -self._get_bom_level(o.product_id),
                self._get_abc_priority(o.product_id),
                o.product_id,
                o.due_day,
            ),
        )

        for order in sorted_orders:
            if order.status == ProductionOrderStatus.COMPLETE:
                continue

            # Process order at its assigned plant (logic selects best line)
            batch = self._process_single_order(order, current_day)
            if batch is not None:
                new_batches.append(batch)
                self.batches.append(batch)

        # Calculate OEE and TEEP for each plant.
        #
        # OEE = Availability x Performance x Quality
        #   Uses Planned Production Time as denominator (only lines
        #   that received work). Per SMRP/Vorne standards, idle lines
        #   with no demand are a Schedule Loss excluded from OEE.
        #
        # TEEP = OEE x Utilization
        #   Uses raw calendar time (24h x all lines) as denominator.
        #   Reveals total hidden capacity for strategic planning.
        #
        # Components:
        # - Availability = (run + changeover) / planned_production_time
        # - Performance = efficiency_factor (0.78-0.88 per config)
        # - Quality = yield % (98.5%)
        # - Utilization = planned_time / (24h x all lines)
        for plant_id, plant_state in self._plant_states.items():
            num_lines = len(plant_state.lines)
            total_run = sum(line.run_hours_today for line in plant_state.lines)
            total_changeover = sum(
                line.changeover_hours_today for line in plant_state.lines
            )
            actual_operating = total_run + total_changeover

            # Performance and Quality are the same for OEE and TEEP
            performance = self._plant_efficiency.get(plant_id, 0.85)
            quality = self.default_yield_percent / 100.0

            # Count active lines (lines that received at least one order)
            active_lines = sum(
                1
                for line in plant_state.lines
                if line.run_hours_today > 0 or line.changeover_hours_today > 0
            )

            # OEE: denominator = planned production time (active lines)
            # Each active line's planned time = hours_per_day x (1-dt)
            # which equals max_capacity_hours (set in _initialize)
            if active_lines > 0:
                planned_production_time = sum(
                    line.max_capacity_hours
                    for line in plant_state.lines
                    if line.run_hours_today > 0
                    or line.changeover_hours_today > 0
                )
                availability = actual_operating / planned_production_time
                plant_oee[plant_id] = availability * performance * quality
            else:
                plant_oee[plant_id] = 0.0

            # TEEP: denominator = raw calendar time (all lines)
            total_calendar_time = self.hours_per_day * num_lines
            if total_calendar_time > 0:
                utilization = actual_operating / total_calendar_time
                plant_teep[plant_id] = utilization * performance * quality
            else:
                plant_teep[plant_id] = 0.0

        return sorted_orders, new_batches, plant_oee, plant_teep

    def _process_single_order(
        self, order: ProductionOrder, current_day: int
    ) -> Batch | None:
        """
        Process a single Production Order on a specific line.

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

        # Bulk intermediates use separate mixing vessels — no filling line
        # capacity consumed. Check materials and produce immediately.
        if self._get_bom_level(order.product_id) > 0:
            return self._process_bulk_order(order, current_day, remaining_qty)

        # --- SKU production: uses filling line capacity ---

        # Select a line for this order
        line = self._select_line(order.plant_id, order.product_id)
        if line is None:
            # No capacity on any line
            return None

        # Use base run rate (rate_multiplier is now 1.0 ideally, but we keep the calc)
        effective_run_rate = recipe.run_rate_cases_per_hour * self.rate_multiplier

        # Calculate production time needed for remaining
        production_time_hours = remaining_qty / effective_run_rate

        # Check for changeover penalty on THIS LINE
        changeover_time = 0.0
        if (
            line.last_product_id is not None
            and line.last_product_id != order.product_id
        ):
            changeover_time = recipe.changeover_time_hours * self.changeover_multiplier

        # How much can we actually produce today on THIS LINE?
        available_time = line.remaining_capacity_hours
        available_time_for_prod = max(0.0, available_time - changeover_time)

        if available_time_for_prod <= 0 and available_time < changeover_time:
            # Not even enough time for changeover
            return None

        max_qty_today = min(
            remaining_qty,
            available_time_for_prod * effective_run_rate,
        )

        if max_qty_today <= 0:
            return None

        # Check raw material availability for TODAY's potential production
        # Note: Material check is still plant-wide (shared inventory)
        material_available, _ = self._check_material_availability(
            order.plant_id, order.product_id, max_qty_today
        )

        if not material_available:
            # Cannot produce - material shortage (SPOF triggered)
            # order.status stays PLANNED or IN_PROGRESS
            return None

        # If we have materials for today, proceed with production calculation
        total_time_needed = production_time_hours + changeover_time
        if line.remaining_capacity_hours < total_time_needed:
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

        # Update LINE state
        run_time = actual_qty / effective_run_rate
        time_used = changeover_time + run_time

        line.remaining_capacity_hours -= time_used
        line.last_product_id = order.product_id
        line.run_hours_today += run_time
        line.changeover_hours_today += changeover_time
        line.output_cases_today += actual_qty

        # Create batch for TODAY's production
        batch = self._create_batch(order, current_day, actual_qty)

        # Add produced goods to plant inventory
        self._add_to_inventory(order.plant_id, order.product_id, actual_qty)

        # Check if order is complete
        if order.produced_quantity >= order.quantity_cases:
            order.status = ProductionOrderStatus.COMPLETE
            order.actual_end_day = current_day

        return batch

    def _process_bulk_order(
        self, order: ProductionOrder, current_day: int, remaining_qty: float
    ) -> Batch | None:
        """Process a bulk intermediate production order.

        Bulk intermediates (compounding) use separate mixing vessels that
        don't compete for filling line capacity. They complete in a single
        day if raw materials are available — no capacity gating.
        """
        material_available, _ = self._check_material_availability(
            order.plant_id, order.product_id, remaining_qty
        )
        if not material_available:
            return None

        if order.actual_start_day is None:
            order.actual_start_day = current_day
        order.status = ProductionOrderStatus.IN_PROGRESS

        self._consume_materials(order.plant_id, order.product_id, remaining_qty)
        order.produced_quantity += remaining_qty

        batch = self._create_batch(order, current_day, remaining_qty)
        self._add_to_inventory(order.plant_id, order.product_id, remaining_qty)

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

        # FIFO age reduction on ingredient consumption
        old_qty = np.maximum(0.0, self.state.actual_inventory[plant_idx])
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_remaining = np.where(
                old_qty > 0,
                np.clip((old_qty - actual_consumed) / old_qty, 0.0, 1.0),
                0.0,
            )
        consume_mask = actual_consumed > 0
        self.state.inventory_age[plant_idx] = np.where(
            consume_mask,
            self.state.inventory_age[plant_idx] * frac_remaining,
            self.state.inventory_age[plant_idx],
        )

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
        """Add produced goods to plant inventory with age blending.

        Uses receive_inventory() so fresh production (age 0) blends with
        existing FG, keeping plant age realistic instead of climbing
        monotonically.
        """
        node_idx = self.state.get_node_idx(plant_id)
        product_idx = self.state.get_product_idx(product_id)
        self.state.receive_inventory(node_idx, product_idx, quantity)

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
