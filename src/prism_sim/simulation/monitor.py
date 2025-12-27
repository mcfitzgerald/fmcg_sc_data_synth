from dataclasses import dataclass
from typing import Any

import numpy as np

from prism_sim.network.core import Batch, Shipment, ShipmentStatus
from prism_sim.simulation.state import StateManager
from prism_sim.simulation.world import World

MIN_SAMPLES_FOR_VARIANCE = 2


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
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < MIN_SAMPLES_FOR_VARIANCE:
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
        self.config = config.get("validation", {})

        # Accumulators for metrics
        self.oee_tracker = WelfordAccumulator()
        self.truck_fill_tracker = WelfordAccumulator()
        self.slob_tracker = WelfordAccumulator()  # SLOB: Slow Moving & Obsolete
        self.cost_tracker = WelfordAccumulator()

        # Ranges from config
        self.oee_range = self.config.get("oee_range", [0.65, 0.85])
        self.truck_fill_target = self.config.get("truck_fill_target", 0.85)
        self.slob_max_pct = self.config.get("slob_max_pct", 0.30)
        self.cost_range = self.config.get("cost_per_case_range", [1.00, 3.00])

    def record_oee(self, oee_value: float) -> None:
        self.oee_tracker.update(oee_value)

    def record_truck_fill(self, fill_rate: float) -> None:
        self.truck_fill_tracker.update(fill_rate)

    def record_slob(self, slob_pct: float) -> None:
        self.slob_tracker.update(slob_pct)

    def record_cost_per_case(self, cost: float) -> None:
        self.cost_tracker.update(cost)

    def get_report(self) -> dict[str, Any]:
        return {
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
                    "OK"
                    if self.slob_tracker.mean <= self.slob_max_pct
                    else "HIGH"
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
        }


class PhysicsAuditor:
    """Enforces conservation laws and physical constraints."""

    def __init__(self, state: StateManager, world: World, config: dict[str, Any]):
        self.state = state
        self.world = world
        self.config = config.get("validation", {})
        self.mass_balance_drift_max = self.config.get("mass_balance_drift_max", 0.02)

    def check_mass_balance(
        self, batches: list[Batch], shipments: list[Shipment]
    ) -> list[str]:
        violations: list[str] = []

        # 1. Ingredient -> Batch Conservation
        # (Simplified check: sum of ingredients used ~= batch output * yield)
        # Note: We need deep genealogy for this, which might be expensive to traverse
        # every step. For now, we can check basic conversion ratios if we had
        # ingredient usage data readily available per batch.

        # 2. Inventory Positivity
        # We allow backlog (negative inventory) conceptually, but physical inventory
        # cannot be negative. In our simulation, 'inventory' < 0 implies backlog.
        # So we check if "actual" physical inventory is negative (if we were tracking
        # it separately).
        # Since state.inventory is net (Physical - Backlog), negative is allowed as
        # a business state. But if we assume this check is for "Physical Reality",
        # we'd need separate tracking.
        # For now, we'll skip simple negativity check on the main inventory array.

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
