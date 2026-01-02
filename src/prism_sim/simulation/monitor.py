from dataclasses import dataclass
from typing import Any

import numpy as np

from prism_sim.network.core import Batch, Shipment, ShipmentStatus
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

# Default minimum samples for variance calculation
# Can be overridden via config: validation.min_samples_for_variance
DEFAULT_MIN_SAMPLES_FOR_VARIANCE = 2

# Module-level variable that gets updated from config
_min_samples_for_variance = DEFAULT_MIN_SAMPLES_FOR_VARIANCE


@dataclass
class WelfordAccumulator:
    """
    Implements Welford's online algorithm for calculating mean and variance
    in a single pass (O(1) update).
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the current mean

    def update(self, new_value: float) -> None:
        val = float(new_value)
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < _min_samples_for_variance:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        return float(np.sqrt(self.variance))


class RealismMonitor:
    """
    Online streaming validator using Welford accumulators to track
    key supply chain metrics and ensure they stay within realistic bounds.
    """

    def __init__(self, config: dict[str, Any]):
        global _min_samples_for_variance
        self.config = config.get("validation", {})

        # Update module-level variance threshold from config
        _min_samples_for_variance = self.config.get(
            "min_samples_for_variance", DEFAULT_MIN_SAMPLES_FOR_VARIANCE
        )

        # Accumulators for metrics
        self.oee_tracker = WelfordAccumulator()
        self.truck_fill_tracker = WelfordAccumulator()
        self.slob_tracker = WelfordAccumulator()  # SLOB: Slow Moving & Obsolete
        self.cost_tracker = WelfordAccumulator()
        self.inventory_turns_tracker = WelfordAccumulator()
        self.service_level_tracker = WelfordAccumulator()  # Aggregate Fill Rate
        self.store_service_level_tracker = WelfordAccumulator()  # Consumer Service Level

        # New KPIs (Fix 8)
        self.perfect_order_tracker = WelfordAccumulator()
        self.cash_to_cash_tracker = WelfordAccumulator()
        self.scope_3_tracker = WelfordAccumulator()
        self.mape_tracker = WelfordAccumulator()
        self.shrinkage_tracker = WelfordAccumulator()

        # Ranges from config
        self.oee_range = self.config.get("oee_range", [0.65, 0.85])
        self.truck_fill_target = self.config.get("truck_fill_target", 0.85)
        self.slob_max_pct = self.config.get("slob_max_pct", 0.30)
        self.cost_range = self.config.get("cost_per_case_range", [1.00, 3.00])
        self.turns_range = self.config.get("inventory_turns_range", [6.0, 14.0])
        self.perfect_order_threshold = self.config.get("perfect_order_threshold", 0.90)

    def record_oee(self, oee_value: float) -> None:
        self.oee_tracker.update(oee_value)

    def record_service_level(self, fill_rate: float) -> None:
        self.service_level_tracker.update(fill_rate)

    def record_store_service_level(self, service_level: float) -> None:
        self.store_service_level_tracker.update(service_level)

    def record_truck_fill(self, fill_rate: float) -> None:
        self.truck_fill_tracker.update(fill_rate)

    def record_slob(self, slob_pct: float) -> None:
        self.slob_tracker.update(slob_pct)

    def record_cost_per_case(self, cost: float) -> None:
        self.cost_tracker.update(cost)

    def record_inventory_turns(self, turns: float) -> None:
        self.inventory_turns_tracker.update(turns)

    def record_perfect_order(self, rate: float) -> None:
        self.perfect_order_tracker.update(rate)

    def record_cash_to_cash(self, days: float) -> None:
        self.cash_to_cash_tracker.update(days)

    def record_scope_3(self, emissions_per_case: float) -> None:
        self.scope_3_tracker.update(emissions_per_case)

    def record_mape(self, mape: float) -> None:
        self.mape_tracker.update(mape)

    def record_shrinkage_rate(self, rate: float) -> None:
        self.shrinkage_tracker.update(rate)

    def get_report(self) -> dict[str, Any]:
        return {
            "service_level": {
                "mean": self.service_level_tracker.mean,
                "std": self.service_level_tracker.std_dev,
                "status": "OK",
            },
            "store_service_level": {
                "mean": self.store_service_level_tracker.mean,
                "std": self.store_service_level_tracker.std_dev,
                "status": "OK",
            },
            "oee": {
                "mean": self.oee_tracker.mean,
                "std": self.oee_tracker.std_dev,
                "target": self.oee_range,
                "status": (
                    "OK"
                    if self.oee_range[0] <= self.oee_tracker.mean <= self.oee_range[1]
                    else "DRIFT"
                ),
            },
            "truck_fill": {
                "mean": self.truck_fill_tracker.mean,
                "target": self.truck_fill_target,
                "status": (
                    "OK"
                    if self.truck_fill_tracker.mean >= self.truck_fill_target
                    else "LOW"
                ),
            },
            "slob": {
                "mean": self.slob_tracker.mean,
                "max": self.slob_max_pct,
                "status": (
                    "OK" if self.slob_tracker.mean <= self.slob_max_pct else "HIGH"
                ),
            },
            "cost_per_case": {
                "mean": self.cost_tracker.mean,
                "target": self.cost_range,
                "status": (
                    "OK"
                    if self.cost_range[0]
                    <= self.cost_tracker.mean
                    <= self.cost_range[1]
                    else "DRIFT"
                ),
            },
            "inventory_turns": {
                "mean": self.inventory_turns_tracker.mean,
                "target": self.turns_range,
                "status": (
                    "OK"
                    if self.turns_range[0]
                    <= self.inventory_turns_tracker.mean
                    <= self.turns_range[1]
                    else "DRIFT"
                ),
            },
            "perfect_order_rate": {
                "mean": self.perfect_order_tracker.mean,
                "status": "OK" if self.perfect_order_tracker.mean >= self.perfect_order_threshold else "LOW"
            },
            "cash_to_cash_days": {
                "mean": self.cash_to_cash_tracker.mean,
                "status": "OK"
            },
            "scope_3_emissions": {
                "mean": self.scope_3_tracker.mean,
                "status": "OK"
            },
            "mape": {
                "mean": self.mape_tracker.mean,
                "status": "OK"
            },
            "shrinkage_rate": {
                "mean": self.shrinkage_tracker.mean,
                "status": "OK"
            }
        }


@dataclass
class DailyFlows:
    """Tracks all inventory flows for a single day for mass balance auditing."""

    day: int
    opening_inventory: np.ndarray  # Shape: [n_nodes, n_products]

    # Flow accumulators (same shape)
    sales: np.ndarray  # POS demand consumed
    receipts: np.ndarray  # Shipments arrived at destination
    allocation_out: np.ndarray  # Inventory decremented during allocation
    production_in: np.ndarray  # Finished goods added at plants
    consumption_out: np.ndarray  # Raw materials consumed in production
    shrinkage: np.ndarray  # Phantom inventory losses

    closing_inventory: np.ndarray | None = None


class PhysicsAuditor:
    """
    Automated checks for Supply Chain Physics integrity.
    [Task 6.6] [Intent: 5. Validation - Physics Audit]
    """

    def __init__(self, state: StateManager, world: World, config: dict[str, Any]) -> None:
        self.state = state
        self.world = world
        self.config = config

        # Drift threshold (e.g. 0.02 = 2% mismatch allowed)
        self.mass_balance_drift_max = config.get("mass_balance_drift_max", 0.02)
        self.mass_balance_min_threshold = config.get("mass_balance_min_threshold", 1.0)
        self.all_violations: list[str] = []

        self.current_flows: DailyFlows | None = None

    def start_day(self, day: int) -> None:
        """Snapshot opening inventory and reset flow accumulators."""
        # We use actual_inventory (Ground Truth) for the audit
        self.current_flows = DailyFlows(
            day=day,
            opening_inventory=self.state.actual_inventory.copy(),
            sales=np.zeros_like(self.state.actual_inventory),
            receipts=np.zeros_like(self.state.actual_inventory),
            allocation_out=np.zeros_like(self.state.actual_inventory),
            production_in=np.zeros_like(self.state.actual_inventory),
            consumption_out=np.zeros_like(self.state.actual_inventory),
            shrinkage=np.zeros_like(self.state.actual_inventory),
        )

    def record_sales(self, demand: np.ndarray) -> None:
        """Record sales (POS demand) consumption."""
        if self.current_flows:
            self.current_flows.sales += demand

    def record_receipts(self, shipments: list[Shipment]) -> None:
        """Record inventory additions from arriving shipments."""
        if not self.current_flows:
            return
        for s in shipments:
            node_idx = self.state.node_id_to_idx.get(s.target_id)
            if node_idx is None:
                continue
            for line in s.lines:
                prod_idx = self.state.product_id_to_idx.get(line.product_id)
                if prod_idx is not None:
                    self.current_flows.receipts[node_idx, prod_idx] += line.quantity

    def record_allocation_out(self, allocation_matrix: np.ndarray) -> None:
        """
        Record inventory decremented during allocation.

        This replaces shipments_out tracking to fix the FTL consolidation timing
        mismatch. Inventory is decremented at allocation time, not when shipments
        are created, so we track allocation directly for accurate mass balance.

        Args:
            allocation_matrix: Shape [n_nodes, n_products] with quantities
                              decremented from each node during allocation.
        """
        if not self.current_flows:
            return
        self.current_flows.allocation_out += allocation_matrix

    def record_plant_shipments_out(self, shipments: list[Shipment]) -> None:
        """
        Record inventory decremented for plant-to-RDC shipments.

        Plant shipments are created immediately (no FTL hold) so we track them
        when shipments are created, unlike customer orders which are tracked
        at allocation time.
        """
        if not self.current_flows:
            return
        for s in shipments:
            source_idx = self.state.node_id_to_idx.get(s.source_id)
            if source_idx is None:
                continue
            for line in s.lines:
                prod_idx = self.state.product_id_to_idx.get(line.product_id)
                if prod_idx is not None:
                    self.current_flows.allocation_out[source_idx, prod_idx] += line.quantity

    def record_production(self, batches: list[Batch]) -> None:
        """Record finished goods added and raw materials consumed in production."""
        if not self.current_flows:
            return
        for b in batches:
            plant_idx = self.state.node_id_to_idx.get(b.plant_id)
            if plant_idx is None:
                continue

            # Record FG output
            prod_idx = self.state.product_id_to_idx.get(b.product_id)
            if prod_idx is not None:
                self.current_flows.production_in[plant_idx, prod_idx] += b.quantity_cases

            # Record Ingredient consumption
            for ing_id, qty in b.ingredients_consumed.items():
                ing_idx = self.state.product_id_to_idx.get(ing_id)
                if ing_idx is not None:
                    self.current_flows.consumption_out[plant_idx, ing_idx] += qty

    def record_shrinkage(self, events: list[Any]) -> None:
        """Record inventory losses from shrinkage (quirks)."""
        if not self.current_flows:
            return
        for e in events:
            # Assuming event has node_id, product_id, and quantity_lost (per plan)
            node_idx = self.state.node_id_to_idx.get(e.node_id)
            prod_idx = self.state.product_id_to_idx.get(e.product_id)
            if node_idx is not None and prod_idx is not None:
                self.current_flows.shrinkage[node_idx, prod_idx] += e.quantity_lost

    def end_day(self) -> None:
        """Snapshot closing inventory."""
        if self.current_flows:
            self.current_flows.closing_inventory = self.state.actual_inventory.copy()

    def check_mass_balance(self) -> list[str]:
        """
        Validate I_t = I_{t-1} + Inflows - Outflows.

        Conservation Law:
        Opening + Receipts + ProdIn - Sales - AllocationOut - Consumed - Shrinkage == Closing

        Note: We track allocation_out (inventory decremented at allocation time) instead
        of shipments_out to fix the FTL consolidation timing mismatch where inventory
        is decremented before shipments are created.
        """
        f = self.current_flows
        if f is None or f.closing_inventory is None:
            return []

        violations: list[str] = []

        # Expected closing = opening + inflows - outflows
        expected = (
            f.opening_inventory
            + f.receipts
            + f.production_in
            - f.sales
            - f.allocation_out
            - f.consumption_out
            - f.shrinkage
        )

        # Calculate drift as percentage of opening (avoid div by zero with max(1.0))
        # We use absolute difference for the numerator
        abs_diff = np.abs(expected - f.closing_inventory)
        denominator = np.maximum(np.abs(f.opening_inventory), 1.0)
        drift = abs_diff / denominator

        # Find violations exceeding threshold
        # Also require minimum absolute difference (1.0 case) to filter
        # floating-point noise and floor guard artifacts
        violation_mask = (drift > self.mass_balance_drift_max) & (
            abs_diff > self.mass_balance_min_threshold
        )

        if np.any(violation_mask):
            # Report top violations (limit to avoid flood)
            indices = np.where(violation_mask)
            # Take up to 10 indices
            for i in range(min(10, len(indices[0]))):
                node_idx = indices[0][i]
                prod_idx = indices[1][i]

                node_id = self.state.node_idx_to_id[node_idx]
                prod_id = self.state.product_idx_to_id[prod_idx]

                msg = (
                    f"Day {f.day}: Node={node_id} Product={prod_id} "
                    f"Expected={expected[node_idx, prod_idx]:.1f} "
                    f"Actual={f.closing_inventory[node_idx, prod_idx]:.1f} "
                    f"Drift={drift[node_idx, prod_idx]*100:.2f}%"
                )
                violations.append(msg)
                self.all_violations.append(msg)

        return violations

    def check_kinematic_consistency(
        self, shipments: list[Shipment], current_day: int
    ) -> list[str]:
        violations: list[str] = []
        # Travel time = Distance / Speed
        # In our case, check if arrival_day - creation_day >= min_lead_time
        for s in shipments:
            if s.status == ShipmentStatus.DELIVERED:
                transit_time = s.arrival_day - s.creation_day
                # Simple check: no teleportation
                if transit_time < 0:
                    violations.append(
                        f"Teleportation detected for {s.id}: "
                        f"transit_time={transit_time}"
                    )
        return violations


class ResilienceTracker:
    """Tracks TTS and TTR per Simchi-Levi framework."""

    def __init__(self, state: StateManager, world: World):
        self.state = state
        self.world = world
        self.disruption_start_day: dict[str, int] = {}  # node_id -> day
        self.recovery_start_day: dict[str, int] = {}  # node_id -> day

        # Track recovery status
        self.node_status: dict[str, str] = {
            n: "NORMAL" for n in world.nodes
        }  # NORMAL, DISRUPTED, RECOVERING

    def start_disruption(self, node_id: str, day: int) -> None:
        self.disruption_start_day[node_id] = day
        self.node_status[node_id] = "DISRUPTED"

    def check_survival(self, day: int) -> dict[str, int]:
        """
        Returns TTS (Time-to-Survive) for disrupted nodes.
        TTS = Days until stockout when node goes dark.
        """
        tts_results: dict[str, int] = {}
        for node_id, status in self.node_status.items():
            if status == "DISRUPTED":
                # Check if stockout occurred
                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is not None:
                    # If any product inventory <= 0
                    min_inv = np.min(self.state.perceived_inventory[node_idx, :])
                    if min_inv <= 0:
                        start = self.disruption_start_day.get(node_id, day)
                        tts_results[node_id] = day - start
        return tts_results

    def check_recovery(self, day: int) -> dict[str, int]:
        """
        Returns TTR (Time-to-Recover) for recovering nodes.
        TTR = Days to restore full service after disruption.
        """
        ttr_results: dict[str, int] = {}
        # Implementation would track when node returns to stable inventory levels
        return ttr_results


class LegacyValidator:
    """Ported checks from reference validation.py."""

    def __init__(self, state: StateManager, world: World):
        self.state = state
        self.world = world

    def check_hub_concentration(self) -> bool:
        # Check if Chicago (or main hub) handles ~20-30% volume
        # This requires tracking accumulated flow through nodes, which we might not
        # have in StateManager yet.
        # Placeholder
        return True

    def check_named_entities(self) -> bool:
        # Check if required entities exist
        required = ["DC-CHI", "DC-NYC", "DC-LAX"]  # Examples
        missing = [n for n in required if n not in self.world.nodes]
        return len(missing) == 0
