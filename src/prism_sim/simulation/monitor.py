"""
Validation framework for Milestone 6.

Implements:
- RealismMonitor: Online streaming validator for OEE, Truck Fill, SLOB, etc.
- ResilienceTracker: TTS and TTR measurement during disruptions
- LegacyValidator: Ported checks from reference validation.py
- PhysicsAuditor: Mass balance and conservation law enforcement

[Task 6.1, 6.2, 6.5, 6.6] Validation, Quirks & Realism
[Intent: Section 4 - The Realism Monitor]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from prism_sim.network.core import Batch, Shipment
    from prism_sim.simulation.state import StateManager
    from prism_sim.simulation.world import World


# =============================================================================
# Welford Accumulator for O(1) Streaming Statistics
# =============================================================================


@dataclass
class WelfordAccumulator:
    """
    Online streaming mean/variance calculation using Welford's algorithm.

    Provides O(1) updates and O(1) access to running statistics.
    Numerically stable for large sample sizes.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the mean

    def update(self, value: float) -> None:
        """Add a new sample to the accumulator."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    # Minimum count for variance calculation
    MIN_VARIANCE_COUNT: int = 2

    def variance(self) -> float:
        """Return the sample variance."""
        if self.count < self.MIN_VARIANCE_COUNT:
            return 0.0
        return float(self.m2 / (self.count - 1))

    def std(self) -> float:
        """Return the sample standard deviation."""
        return float(np.sqrt(self.variance()))

    def reset(self) -> None:
        """Reset the accumulator."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


# =============================================================================
# Mass Balance Checker
# =============================================================================


@dataclass
class MassBalanceResult:
    """Result of a mass balance check."""

    check_name: str
    input_qty: float
    output_qty: float
    scrap_qty: float
    drift_pct: float
    is_valid: bool
    message: str


@dataclass
class ProductionMetrics:
    """Metrics for a production event."""

    capacity_used_hours: float
    capacity_available_hours: float
    changeover_hours: float
    ingredients_consumed: float
    batch_quantity: float
    scrap: float = 0.0


@dataclass
class ShipmentMetrics:
    """Metrics for a shipment event."""

    shipment_weight: float
    shipment_volume: float
    truck_max_weight: float
    truck_max_volume: float
    cases: float
    freight_cost: float = 0.0


class MassBalanceChecker:
    """
    Checks conservation of mass across simulation phases.

    Tracks cumulative inputs and outputs to detect drift.
    """

    def __init__(self, max_drift_pct: float = 0.02) -> None:
        self.max_drift_pct = max_drift_pct

        # Cumulative tracking
        self.total_ingredients_consumed: float = 0.0
        self.total_batches_produced: float = 0.0
        self.total_scrap: float = 0.0

        self.total_orders_created: float = 0.0
        self.total_orders_fulfilled: float = 0.0

        self.total_shipped: float = 0.0
        self.total_received: float = 0.0

    def record_production(
        self, ingredients_consumed: float, batch_quantity: float, scrap: float = 0.0
    ) -> None:
        """Record a production event."""
        self.total_ingredients_consumed += ingredients_consumed
        self.total_batches_produced += batch_quantity
        self.total_scrap += scrap

    def record_order(self, order_quantity: float) -> None:
        """Record an order created."""
        self.total_orders_created += order_quantity

    def record_fulfillment(self, fulfilled_quantity: float) -> None:
        """Record an order fulfilled."""
        self.total_orders_fulfilled += fulfilled_quantity

    def record_shipment(self, shipped_quantity: float) -> None:
        """Record goods shipped."""
        self.total_shipped += shipped_quantity

    def record_receipt(self, received_quantity: float) -> None:
        """Record goods received."""
        self.total_received += received_quantity

    def check_production_balance(self) -> MassBalanceResult:
        """Check: Ingredient -> Batch + Scrap (drift <= +2%)."""
        total_out = self.total_batches_produced + self.total_scrap
        drift = 0.0
        if self.total_ingredients_consumed > 0:
            drift = (total_out - self.total_ingredients_consumed) / (
                self.total_ingredients_consumed
            )

        is_valid = drift <= self.max_drift_pct

        return MassBalanceResult(
            check_name="Production Balance",
            input_qty=self.total_ingredients_consumed,
            output_qty=total_out,
            scrap_qty=self.total_scrap,
            drift_pct=drift,
            is_valid=is_valid,
            message=(
                f"Production balance: {drift:.2%} drift "
                f"({'OK' if is_valid else 'FAIL'})"
            ),
        )

    def check_fulfillment_balance(self) -> MassBalanceResult:
        """Check: Order -> Fulfill (drift <= +2%, can't ship more than ordered)."""
        drift = 0.0
        if self.total_orders_created > 0:
            drift = (self.total_orders_fulfilled - self.total_orders_created) / (
                self.total_orders_created
            )

        # Can't ship more than ordered
        is_valid = drift <= self.max_drift_pct

        return MassBalanceResult(
            check_name="Fulfillment Balance",
            input_qty=self.total_orders_created,
            output_qty=self.total_orders_fulfilled,
            scrap_qty=0.0,
            drift_pct=drift,
            is_valid=is_valid,
            message=(
                f"Fulfillment balance: {drift:.2%} drift "
                f"({'OK' if is_valid else 'FAIL'})"
            ),
        )

    def check_logistics_balance(self) -> MassBalanceResult:
        """Check: Shipped -> Received (allow ±10% for in-transit)."""
        drift = 0.0
        if self.total_shipped > 0:
            drift = abs(self.total_received - self.total_shipped) / self.total_shipped

        # Allow larger drift for in-transit goods (10%)
        logistics_drift_tolerance = 0.10
        is_valid = drift <= logistics_drift_tolerance

        return MassBalanceResult(
            check_name="Logistics Balance",
            input_qty=self.total_shipped,
            output_qty=self.total_received,
            scrap_qty=0.0,
            drift_pct=drift,
            is_valid=is_valid,
            message=(
                f"Logistics balance: {drift:.2%} drift "
                f"({'OK' if is_valid else 'FAIL'})"
            ),
        )


# =============================================================================
# Metrics Report
# =============================================================================


@dataclass
class MetricsSnapshot:
    """A point-in-time snapshot of key metrics."""

    day: int
    oee: float
    truck_fill_weight: float
    truck_fill_volume: float
    slob_pct: float
    inventory_turns: float
    cost_per_case: float
    schedule_adherence_days: float
    mass_balance_drift: float
    otif_pct: float
    osa_pct: float


@dataclass
class MetricsReport:
    """Aggregated metrics report across the simulation."""

    snapshots: list[MetricsSnapshot] = field(default_factory=list)

    # Welford accumulators for streaming stats
    oee_acc: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    truck_fill_weight_acc: WelfordAccumulator = field(
        default_factory=WelfordAccumulator
    )
    truck_fill_volume_acc: WelfordAccumulator = field(
        default_factory=WelfordAccumulator
    )
    slob_acc: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    inventory_turns_acc: WelfordAccumulator = field(
        default_factory=WelfordAccumulator
    )
    cost_per_case_acc: WelfordAccumulator = field(
        default_factory=WelfordAccumulator
    )

    def add_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Add a daily snapshot and update accumulators."""
        self.snapshots.append(snapshot)
        self.oee_acc.update(snapshot.oee)
        self.truck_fill_weight_acc.update(snapshot.truck_fill_weight)
        self.truck_fill_volume_acc.update(snapshot.truck_fill_volume)
        self.slob_acc.update(snapshot.slob_pct)
        self.inventory_turns_acc.update(snapshot.inventory_turns)
        self.cost_per_case_acc.update(snapshot.cost_per_case)

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "days_simulated": len(self.snapshots),
            "oee_mean": self.oee_acc.mean,
            "oee_std": self.oee_acc.std(),
            "truck_fill_weight_mean": self.truck_fill_weight_acc.mean,
            "truck_fill_volume_mean": self.truck_fill_volume_acc.mean,
            "slob_mean": self.slob_acc.mean,
            "inventory_turns_mean": self.inventory_turns_acc.mean,
            "cost_per_case_mean": self.cost_per_case_acc.mean,
        }


# =============================================================================
# Realism Monitor (Task 6.1)
# =============================================================================


class RealismMonitor:
    """
    Online streaming validator for supply chain metrics.

    Tracks key performance indicators against industry benchmarks:
    - OEE (Overall Equipment Effectiveness): 65-85% target
    - Truck Fill Rate: >85% weight/cube
    - SLOB (Slow/Obsolete) Inventory: <30% working capital
    - Inventory Turns: 6-14x
    - Cost-to-Serve: $1-3/case
    - Schedule Adherence: <1.1 days
    - Mass Balance: drift <2%

    [Task 6.1] Realism Monitor
    [Intent: Section 4 - The Realism Monitor]
    """

    def __init__(self, config: dict[str, Any]) -> None:
        validation_config = config.get("simulation_parameters", {}).get(
            "validation", {}
        )

        # Target ranges from config
        self.oee_range = tuple(validation_config.get("oee_range", [0.65, 0.85]))
        self.truck_fill_target = validation_config.get("truck_fill_target", 0.85)
        self.slob_max_pct = validation_config.get("slob_max_pct", 0.30)
        self.inventory_turns_range = tuple(
            validation_config.get("inventory_turns_range", [6.0, 14.0])
        )
        self.cost_per_case_range = tuple(
            validation_config.get("cost_per_case_range", [1.00, 3.00])
        )
        self.mass_balance_drift_max = validation_config.get(
            "mass_balance_drift_max", 0.02
        )

        # Components
        self.mass_checker = MassBalanceChecker(self.mass_balance_drift_max)
        self.report = MetricsReport()

        # Daily tracking
        self.daily_capacity_used: float = 0.0
        self.daily_capacity_available: float = 0.0
        self.daily_changeover_time: float = 0.0
        self.daily_shipment_weight: float = 0.0
        self.daily_shipment_volume: float = 0.0
        self.daily_truck_capacity_weight: float = 0.0
        self.daily_truck_capacity_volume: float = 0.0
        self.daily_cases_shipped: float = 0.0
        self.daily_freight_cost: float = 0.0
        self.daily_orders_created: float = 0.0
        self.daily_orders_fulfilled: float = 0.0

        # Violation tracking
        self.violations: list[str] = []

    def record_production(self, metrics: ProductionMetrics) -> None:
        """Record production metrics for OEE calculation."""
        self.daily_capacity_used += metrics.capacity_used_hours
        self.daily_capacity_available += metrics.capacity_available_hours
        self.daily_changeover_time += metrics.changeover_hours
        self.mass_checker.record_production(
            metrics.ingredients_consumed, metrics.batch_quantity, metrics.scrap
        )

    def record_shipment(self, metrics: ShipmentMetrics) -> None:
        """Record shipment metrics for truck fill calculation."""
        self.daily_shipment_weight += metrics.shipment_weight
        self.daily_shipment_volume += metrics.shipment_volume
        self.daily_truck_capacity_weight += metrics.truck_max_weight
        self.daily_truck_capacity_volume += metrics.truck_max_volume
        self.daily_cases_shipped += metrics.cases
        self.daily_freight_cost += metrics.freight_cost
        self.mass_checker.record_shipment(metrics.cases)

    def record_receipt(self, cases: float) -> None:
        """Record goods received."""
        self.mass_checker.record_receipt(cases)

    def record_order(self, order_quantity: float, fulfilled_quantity: float) -> None:
        """Record order creation and fulfillment."""
        self.daily_orders_created += order_quantity
        self.daily_orders_fulfilled += fulfilled_quantity
        self.mass_checker.record_order(order_quantity)
        self.mass_checker.record_fulfillment(fulfilled_quantity)

    def calculate_oee(self) -> float:
        """
        Calculate Overall Equipment Effectiveness.

        OEE = (Production Time - Changeover) / Available Time
        """
        if self.daily_capacity_available <= 0:
            return 1.0  # No capacity means no loss

        effective_production = self.daily_capacity_used - self.daily_changeover_time
        return max(0.0, effective_production / self.daily_capacity_available)

    def calculate_truck_fill(self) -> tuple[float, float]:
        """
        Calculate truck fill rates.

        Returns (weight_fill_rate, volume_fill_rate)
        """
        weight_fill = 1.0
        volume_fill = 1.0

        if self.daily_truck_capacity_weight > 0:
            weight_fill = self.daily_shipment_weight / self.daily_truck_capacity_weight

        if self.daily_truck_capacity_volume > 0:
            volume_fill = self.daily_shipment_volume / self.daily_truck_capacity_volume

        return (weight_fill, volume_fill)

    def calculate_cost_per_case(self) -> float:
        """Calculate cost-to-serve per case."""
        if self.daily_cases_shipped <= 0:
            return 0.0
        return self.daily_freight_cost / self.daily_cases_shipped

    def calculate_otif(self) -> float:
        """Calculate On-Time In-Full percentage."""
        if self.daily_orders_created <= 0:
            return 1.0
        return min(1.0, self.daily_orders_fulfilled / self.daily_orders_created)

    def step(
        self,
        day: int,
        state: StateManager,
        total_demand: float,
        initial_inventory: float,
    ) -> MetricsSnapshot:
        """
        End-of-day validation step.

        Calculates metrics, checks thresholds, records violations.
        """
        oee = self.calculate_oee()
        truck_fill_weight, truck_fill_volume = self.calculate_truck_fill()
        cost_per_case = self.calculate_cost_per_case()
        otif = self.calculate_otif()

        # SLOB calculation (simplified: inventory > 30 days supply = slow)
        avg_daily_demand = total_demand / max(1, day)
        current_inventory = float(np.sum(state.inventory))
        if avg_daily_demand > 0:
            days_supply = current_inventory / avg_daily_demand
        else:
            days_supply = 0.0
        slob_pct = min(1.0, max(0.0, (days_supply - 30) / 60))

        # Inventory turns (annualized)
        if current_inventory > 0:
            inventory_turns = (initial_inventory * 365 / day) / current_inventory
        else:
            inventory_turns = 0.0

        # Mass balance check
        production_balance = self.mass_checker.check_production_balance()

        # OSA calculation (simplified: what % of inventory is positive)
        positive_inventory = np.sum(state.inventory > 0)
        total_skus = state.inventory.size
        osa = positive_inventory / total_skus if total_skus > 0 else 1.0

        snapshot = MetricsSnapshot(
            day=day,
            oee=oee,
            truck_fill_weight=truck_fill_weight,
            truck_fill_volume=truck_fill_volume,
            slob_pct=slob_pct,
            inventory_turns=inventory_turns,
            cost_per_case=cost_per_case,
            schedule_adherence_days=0.0,  # Calculated elsewhere
            mass_balance_drift=production_balance.drift_pct,
            otif_pct=otif,
            osa_pct=float(osa),
        )

        self.report.add_snapshot(snapshot)

        # Check violations
        self._check_violations(snapshot)

        # Reset daily counters
        self._reset_daily()

        return snapshot

    def _check_violations(self, snapshot: MetricsSnapshot) -> None:
        """Check for metric violations against targets."""
        if not (self.oee_range[0] <= snapshot.oee <= self.oee_range[1]):
            self.violations.append(
                f"Day {snapshot.day}: OEE {snapshot.oee:.1%} "
                f"outside range {self.oee_range}"
            )

        truck_fill = max(snapshot.truck_fill_weight, snapshot.truck_fill_volume)
        if truck_fill < self.truck_fill_target:
            self.violations.append(
                f"Day {snapshot.day}: Truck fill {truck_fill:.1%} "
                f"below target {self.truck_fill_target:.1%}"
            )

        if snapshot.slob_pct > self.slob_max_pct:
            self.violations.append(
                f"Day {snapshot.day}: SLOB {snapshot.slob_pct:.1%} "
                f"exceeds max {self.slob_max_pct:.1%}"
            )

    def _reset_daily(self) -> None:
        """Reset daily tracking counters."""
        self.daily_capacity_used = 0.0
        self.daily_capacity_available = 0.0
        self.daily_changeover_time = 0.0
        self.daily_shipment_weight = 0.0
        self.daily_shipment_volume = 0.0
        self.daily_truck_capacity_weight = 0.0
        self.daily_truck_capacity_volume = 0.0
        self.daily_cases_shipped = 0.0
        self.daily_freight_cost = 0.0
        self.daily_orders_created = 0.0
        self.daily_orders_fulfilled = 0.0


# =============================================================================
# Resilience Tracker (Task 6.2)
# =============================================================================


@dataclass
class DisruptionEvent:
    """Tracks a disruption event for resilience measurement."""

    event_code: str
    start_day: int
    affected_nodes: list[str]
    baseline_inventory: dict[str, float]  # node_id -> inventory at start
    tts_days: dict[str, int] = field(default_factory=dict)  # node_id -> TTS
    ttr_days: dict[str, int] = field(default_factory=dict)  # node_id -> TTR
    is_recovered: bool = False


class ResilienceTracker:
    """
    Tracks Time-to-Survive (TTS) and Time-to-Recover (TTR) per Simchi-Levi.

    TTS: Days until stockout when node goes dark (driven by safety stock)
    TTR: Days to restore full service after disruption (driven by lead times + capacity)

    [Task 6.2] Resilience Metrics
    [Intent: Section 2 - Supply Chain Resilience]
    """

    def __init__(self, state: StateManager, config: dict[str, Any]) -> None:
        self.state = state
        self.config = config
        self.active_disruptions: dict[str, DisruptionEvent] = {}
        self.completed_disruptions: list[DisruptionEvent] = []

        # Threshold for "stockout" and "recovery"
        validation_config = config.get("simulation_parameters", {}).get(
            "validation", {}
        )
        self.stockout_threshold = validation_config.get("stockout_threshold", 0.0)
        self.recovery_threshold = validation_config.get("recovery_threshold", 50.0)

    def start_disruption(
        self, event_code: str, day: int, affected_nodes: list[str]
    ) -> None:
        """
        Start tracking a disruption event.

        Captures baseline inventory at affected nodes.
        """
        baseline: dict[str, float] = {}

        for node_id in affected_nodes:
            node_idx = self.state.node_id_to_idx.get(node_id)
            if node_idx is not None:
                # Sum inventory across all products at this node
                baseline[node_id] = float(np.sum(self.state.inventory[node_idx, :]))

        event = DisruptionEvent(
            event_code=event_code,
            start_day=day,
            affected_nodes=affected_nodes,
            baseline_inventory=baseline,
        )

        self.active_disruptions[event_code] = event

    def check_survival(self, day: int) -> dict[str, dict[str, int]]:
        """
        Check survival status for all active disruptions.

        Returns: {event_code: {node_id: tts_days}}
        """
        results: dict[str, dict[str, int]] = {}

        for event_code, event in self.active_disruptions.items():
            results[event_code] = {}

            for node_id in event.affected_nodes:
                if node_id in event.tts_days:
                    # Already recorded TTS for this node
                    continue

                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is None:
                    continue

                # Check if any product hit stockout
                node_inventory = self.state.inventory[node_idx, :]
                if np.any(node_inventory <= self.stockout_threshold):
                    tts = day - event.start_day
                    event.tts_days[node_id] = tts
                    results[event_code][node_id] = tts

        return results

    def check_recovery(self, day: int) -> dict[str, dict[str, int]]:
        """
        Check recovery status for all active disruptions.

        Returns: {event_code: {node_id: ttr_days}}
        """
        results: dict[str, dict[str, int]] = {}

        for event_code, event in list(self.active_disruptions.items()):
            results[event_code] = {}
            all_recovered = True

            for node_id in event.affected_nodes:
                if node_id in event.ttr_days:
                    # Already recorded TTR for this node
                    continue

                node_idx = self.state.node_id_to_idx.get(node_id)
                if node_idx is None:
                    continue

                # Check if all products back above threshold
                node_inventory = self.state.inventory[node_idx, :]
                if np.all(node_inventory >= self.recovery_threshold):
                    ttr = day - event.start_day
                    event.ttr_days[node_id] = ttr
                    results[event_code][node_id] = ttr
                else:
                    all_recovered = False

            # Mark event as recovered if all nodes recovered
            if all_recovered and len(event.ttr_days) == len(event.affected_nodes):
                event.is_recovered = True
                self.completed_disruptions.append(event)
                del self.active_disruptions[event_code]

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all disruption events and their resilience metrics."""
        all_events = list(self.active_disruptions.values()) + self.completed_disruptions

        return {
            "active_disruptions": len(self.active_disruptions),
            "completed_disruptions": len(self.completed_disruptions),
            "events": [
                {
                    "event_code": e.event_code,
                    "start_day": e.start_day,
                    "affected_nodes": e.affected_nodes,
                    "tts_days": e.tts_days,
                    "ttr_days": e.ttr_days,
                    "is_recovered": e.is_recovered,
                }
                for e in all_events
            ],
        }


# =============================================================================
# Legacy Validator (Task 6.5)
# =============================================================================


class LegacyValidator:
    """
    Ported checks from reference validation.py.

    Validates emergent properties against industry benchmarks:
    - Pareto distribution: Top 20% SKUs = 75-85% volume
    - Hub concentration: Chicago ~40%
    - Named entities exist
    - Bullwhip ratio: Order CV / POS CV = 1.5-3.0x
    - Referential integrity

    [Task 6.5] Legacy Validation
    [Intent: Section 4 - The Realism Monitor]
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        validation_config = config.get("simulation_parameters", {}).get(
            "validation", {}
        )
        self.pareto_range = tuple(validation_config.get("pareto_range", [0.75, 0.85]))
        self.hub_concentration_range = tuple(
            validation_config.get("hub_concentration_range", [0.20, 0.30])
        )
        self.bullwhip_range = tuple(
            validation_config.get("bullwhip_range", [1.5, 3.0])
        )
        self.otif_range = tuple(validation_config.get("otif_range", [0.85, 0.98]))
        self.osa_range = tuple(validation_config.get("osa_range", [0.88, 0.96]))

        # Tracking for validation
        self.daily_pos_by_product: list[np.ndarray] = []
        self.daily_orders_by_product: list[np.ndarray] = []
        self.total_volume_by_node: dict[str, float] = {}

    def record_daily_demand(self, pos_demand: np.ndarray) -> None:
        """Record daily POS demand for Pareto and Bullwhip checks."""
        self.daily_pos_by_product.append(pos_demand.sum(axis=0).copy())

    def record_daily_orders(self, orders_by_product: np.ndarray) -> None:
        """Record daily order quantities for Bullwhip check."""
        self.daily_orders_by_product.append(orders_by_product.copy())

    def record_volume_flow(self, node_id: str, volume: float) -> None:
        """Record volume flow through a node for hub concentration check."""
        if node_id not in self.total_volume_by_node:
            self.total_volume_by_node[node_id] = 0.0
        self.total_volume_by_node[node_id] += volume

    def check_pareto_distribution(self) -> tuple[bool, str]:
        """
        Check: Top 20% of SKUs = 75-85% of volume.

        Returns (is_valid, message)
        """
        if not self.daily_pos_by_product:
            return True, "Pareto check: No data yet"

        # Sum across all days
        total_by_product = np.sum(self.daily_pos_by_product, axis=0)
        total_volume = np.sum(total_by_product)

        if total_volume <= 0:
            return True, "Pareto check: No volume"

        # Sort descending
        sorted_volumes = np.sort(total_by_product)[::-1]
        n_products = len(sorted_volumes)
        top_20_pct_idx = max(1, int(n_products * 0.20))

        top_20_volume = np.sum(sorted_volumes[:top_20_pct_idx])
        top_20_share = top_20_volume / total_volume

        is_valid = self.pareto_range[0] <= top_20_share <= self.pareto_range[1]
        message = (
            f"Pareto check: Top 20% SKUs = {top_20_share:.1%} of volume "
            f"(target: {self.pareto_range[0]:.0%}-{self.pareto_range[1]:.0%}) "
            f"{'OK' if is_valid else 'FAIL'}"
        )

        return is_valid, message

    def check_hub_concentration(self) -> tuple[bool, str]:
        """
        Check: Chicago hub handles 20-30% of volume.

        Returns (is_valid, message)
        """
        total_volume = sum(self.total_volume_by_node.values())

        if total_volume <= 0:
            return True, "Hub concentration check: No volume"

        # Find Chicago hub
        chicago_volume = 0.0
        for node_id, volume in self.total_volume_by_node.items():
            if "CHI" in node_id.upper():
                chicago_volume += volume

        chicago_share = chicago_volume / total_volume
        is_valid = (
            self.hub_concentration_range[0]
            <= chicago_share
            <= self.hub_concentration_range[1]
        )

        message = (
            f"Hub concentration check: Chicago = {chicago_share:.1%} "
            f"(target: {self.hub_concentration_range[0]:.0%}-"
            f"{self.hub_concentration_range[1]:.0%}) "
            f"{'OK' if is_valid else 'FAIL'}"
        )

        return is_valid, message

    def check_named_entities(self) -> tuple[bool, str]:
        """
        Check: Required named entities exist.

        Returns (is_valid, message)
        """
        required_nodes = ["RDC-NAM-CHI", "PLANT-OH", "PLANT-TX"]
        required_products = ["FG-PASTE-001", "FG-SOAP-001", "FG-DET-001"]

        missing_nodes = [
            n for n in required_nodes if n not in self.world.nodes
        ]
        missing_products = [
            p for p in required_products if p not in self.world.products
        ]

        is_valid = not missing_nodes and not missing_products
        message = f"Named entities check: {'OK' if is_valid else 'FAIL'}"

        if not is_valid:
            if missing_nodes:
                message += f" Missing nodes: {missing_nodes}"
            if missing_products:
                message += f" Missing products: {missing_products}"

        return is_valid, message

    # Minimum days of data needed for bullwhip calculation
    MIN_BULLWHIP_DAYS: int = 7

    def check_bullwhip_ratio(self) -> tuple[bool, str]:
        """
        Check: Order CV / POS CV = 1.5-3.0x.

        Returns (is_valid, message)
        """
        if len(self.daily_pos_by_product) < self.MIN_BULLWHIP_DAYS:
            return True, "Bullwhip check: Need at least 7 days of data"

        if len(self.daily_orders_by_product) < self.MIN_BULLWHIP_DAYS:
            return True, "Bullwhip check: Need order data"

        # Calculate CV for POS and Orders
        pos_array = np.array(self.daily_pos_by_product)
        orders_array = np.array(self.daily_orders_by_product)

        # Sum across products to get daily totals
        pos_daily = pos_array.sum(axis=1)
        orders_daily = orders_array.sum(axis=1)

        pos_mean = np.mean(pos_daily)
        pos_std = np.std(pos_daily)
        pos_cv = pos_std / pos_mean if pos_mean > 0 else 0

        orders_mean = np.mean(orders_daily)
        orders_std = np.std(orders_daily)
        orders_cv = orders_std / orders_mean if orders_mean > 0 else 0

        bullwhip_ratio = orders_cv / pos_cv if pos_cv > 0 else 0

        is_valid = self.bullwhip_range[0] <= bullwhip_ratio <= self.bullwhip_range[1]
        message = (
            f"Bullwhip ratio: {bullwhip_ratio:.2f}x "
            f"(target: {self.bullwhip_range[0]:.1f}x-{self.bullwhip_range[1]:.1f}x) "
            f"{'OK' if is_valid else 'FAIL'}"
        )

        return is_valid, message

    def check_referential_integrity(self) -> tuple[bool, str]:
        """
        Check: All foreign key references are valid.

        Returns (is_valid, message)
        """
        errors: list[str] = []

        # Check links reference valid nodes
        for link_id, link in self.world.links.items():
            if link.source_id not in self.world.nodes:
                errors.append(f"Link {link_id}: invalid source {link.source_id}")
            if link.target_id not in self.world.nodes:
                errors.append(f"Link {link_id}: invalid target {link.target_id}")

        # Check recipes reference valid products
        for recipe_id, recipe in self.world.recipes.items():
            if recipe.product_id not in self.world.products:
                errors.append(
                    f"Recipe {recipe_id}: invalid output {recipe.product_id}"
                )
            for ing_id in recipe.ingredients:
                if ing_id not in self.world.products:
                    errors.append(f"Recipe {recipe_id}: invalid ingredient {ing_id}")

        is_valid = len(errors) == 0
        message = f"Referential integrity check: {'OK' if is_valid else 'FAIL'}"
        if errors:
            message += f" Errors: {errors[:5]}"  # Show first 5

        return is_valid, message

    def run_all_checks(self) -> dict[str, tuple[bool, str]]:
        """Run all validation checks."""
        return {
            "pareto": self.check_pareto_distribution(),
            "hub_concentration": self.check_hub_concentration(),
            "named_entities": self.check_named_entities(),
            "bullwhip": self.check_bullwhip_ratio(),
            "referential_integrity": self.check_referential_integrity(),
        }


# =============================================================================
# Physics Auditor (Task 6.6)
# =============================================================================


class PhysicsAuditor:
    """
    Enforces conservation laws and physics consistency.

    Checks:
    1. Mass Balance: Ingredient -> Batch (drift <= +2%)
    2. Inventory Positivity: No negative inventory
    3. Kinematic Consistency: Travel time = Distance / Speed

    [Task 6.6] Physics Audit
    [Intent: Section 8 - The Supply Chain Physics Rubric]
    """

    def __init__(
        self, world: World, state: StateManager, config: dict[str, Any]
    ) -> None:
        self.world = world
        self.state = state
        self.config = config

        validation_config = config.get("simulation_parameters", {}).get(
            "validation", {}
        )
        self.mass_balance_drift_max = validation_config.get(
            "mass_balance_drift_max", 0.02
        )
        self.allow_negative_inventory = validation_config.get(
            "allow_negative_inventory", False
        )

        self.mass_checker = MassBalanceChecker(self.mass_balance_drift_max)

        # Cumulative tracking
        self.total_ingredients_used: float = 0.0
        self.total_batches_produced: float = 0.0
        self.total_scrap: float = 0.0

    def record_batch(self, batch: Batch) -> None:
        """Record a completed batch for mass balance tracking."""
        # Sum ingredient consumption
        ingredients_used = sum(batch.ingredients_consumed.values())
        self.total_ingredients_used += ingredients_used
        self.total_batches_produced += batch.quantity_cases

        # Scrap = ingredients - output (accounting for yield)
        expected_output = ingredients_used * (batch.yield_percent / 100.0)
        scrap = max(0.0, expected_output - batch.quantity_cases)
        self.total_scrap += scrap

        self.mass_checker.record_production(
            ingredients_used, batch.quantity_cases, scrap
        )

    def check_mass_balance(self) -> list[str]:
        """
        Check mass balance across production.

        Returns list of violations.
        """
        violations: list[str] = []

        result = self.mass_checker.check_production_balance()
        if not result.is_valid:
            violations.append(result.message)

        return violations

    # Max violations to report before truncating
    MAX_VIOLATION_DETAILS: int = 10

    def check_inventory_positivity(self) -> list[str]:
        """
        Check for negative inventory (physics violation).

        Returns list of violations.
        """
        if self.allow_negative_inventory:
            return []

        violations: list[str] = []

        # Find negative inventory positions
        negative_mask = self.state.inventory < 0
        if np.any(negative_mask):
            negative_positions = np.argwhere(negative_mask)
            for pos in negative_positions[: self.MAX_VIOLATION_DETAILS]:
                node_idx, prod_idx = pos
                node_id = self.state.node_idx_to_id.get(int(node_idx), "?")
                prod_id = self.state.product_idx_to_id.get(int(prod_idx), "?")
                qty = self.state.inventory[node_idx, prod_idx]
                violations.append(
                    f"Negative inventory: {node_id}/{prod_id} = {qty:.1f}"
                )

            remaining = len(negative_positions) - self.MAX_VIOLATION_DETAILS
            if remaining > 0:
                violations.append(f"... and {remaining} more negative positions")

        return violations

    def check_kinematic_consistency(
        self, shipments: list[Shipment]
    ) -> list[str]:
        """
        Check that travel times are consistent with distance/speed.

        Returns list of violations.
        """
        violations: list[str] = []

        for shipment in shipments:
            # Find the link for this route
            link = None
            for route_link in self.world.links.values():
                if (
                    route_link.source_id == shipment.source_id
                    and route_link.target_id == shipment.target_id
                ):
                    link = route_link
                    break

            if link is None:
                continue

            # Calculate expected travel time
            travel_days = shipment.arrival_day - shipment.creation_day

            # Check if travel time matches lead time (with tolerance)
            expected_days = link.lead_time_days
            if abs(travel_days - expected_days) > 1:
                violations.append(
                    f"Shipment {shipment.id}: travel={travel_days}d, "
                    f"expected={expected_days:.1f}d"
                )

        return violations

    def run_all_checks(
        self, shipments: list[Shipment] | None = None
    ) -> dict[str, list[str]]:
        """Run all physics audits."""
        return {
            "mass_balance": self.check_mass_balance(),
            "inventory_positivity": self.check_inventory_positivity(),
            "kinematic_consistency": (
                self.check_kinematic_consistency(shipments)
                if shipments
                else []
            ),
        }

    def summary(self) -> dict[str, Any]:
        """Get summary of physics audit."""
        result = self.mass_checker.check_production_balance()
        return {
            "total_ingredients_used": self.total_ingredients_used,
            "total_batches_produced": self.total_batches_produced,
            "total_scrap": self.total_scrap,
            "mass_balance_drift_pct": result.drift_pct,
            "mass_balance_valid": result.is_valid,
        }
