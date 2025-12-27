"""
Behavioral quirks for realistic supply chain pathologies.

Implements:
- PortCongestionQuirk: AR(1) auto-regressive delays creating clustered late arrivals
- OptimismBiasQuirk: Over-forecast new products by 15%
- PhantomInventoryQuirk: Shrinkage with detection lag (dual inventory tracking)

[Task 6.3] Quirk Injection
[Intent: Section 7 - Preserving Reference Fidelity]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from prism_sim.network.core import Shipment
    from prism_sim.simulation.state import StateManager


# =============================================================================
# Shrinkage Event Tracking
# =============================================================================


@dataclass
class ShrinkageEvent:
    """Records a shrinkage event for later discovery."""

    day_occurred: int
    day_discovered: int
    node_id: str
    product_id: str
    quantity_lost: float
    is_discovered: bool = False


# =============================================================================
# AR(1) Port Congestion Quirk (Task 6.3.1)
# =============================================================================


class PortCongestionQuirk:
    """
    Auto-regressive delays creating clustered late arrivals.

    Formula: current_delay = ar_coefficient * prev_delay + N(0, noise_std)
    Clustering: if prev_delay > threshold, add Uniform(2,6) for consecutive shipments

    [Task 6.3.1] AR(1) Port Congestion
    [Intent: Section 7.A.1 - AR(1) Port Congestion]
    """

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        quirk_config = (
            config.get("simulation_parameters", {})
            .get("quirks", {})
            .get("port_congestion", {})
        )

        self.enabled = quirk_config.get("enabled", True)
        self.ar_coefficient = quirk_config.get("ar_coefficient", 0.70)
        self.noise_std_hours = quirk_config.get("noise_std_hours", 2.0)
        self.cluster_threshold_hours = quirk_config.get("cluster_threshold_hours", 4.0)
        self.cluster_size = quirk_config.get("cluster_size", 3)
        self.cluster_delay_min_hours = quirk_config.get("cluster_delay_min_hours", 2.0)
        self.cluster_delay_max_hours = quirk_config.get("cluster_delay_max_hours", 6.0)
        self.affected_ports = quirk_config.get("affected_ports", ["USLAX", "CNSHA"])

        self._rng = np.random.default_rng(seed)

        # Track previous delay per port for AR(1) correlation
        self._prev_delay_by_port: dict[str, float] = {}
        self._delay_streak_by_port: dict[str, int] = {}

    def apply_to_shipment(
        self, shipment: Shipment, prev_delay: float | None = None
    ) -> float:
        """
        Apply AR(1) congestion delay to a shipment.

        Args:
            shipment: The shipment to potentially delay
            prev_delay: Override for previous delay (uses internal state if None)

        Returns:
            The delay in hours applied to this shipment
        """
        if not self.enabled:
            return 0.0

        # Check if shipment passes through affected port
        affected_port = None
        for port in self.affected_ports:
            if port in shipment.source_id or port in shipment.target_id:
                affected_port = port
                break

        if affected_port is None:
            return 0.0

        # Get previous delay for this port
        if prev_delay is None:
            prev_delay = self._prev_delay_by_port.get(affected_port, 0.0)

        # AR(1): current = coef * prev + noise
        noise = self._rng.normal(0, self.noise_std_hours)
        current_delay = self.ar_coefficient * prev_delay + noise

        # Clustering: if previous delay was large, amplify
        streak = self._delay_streak_by_port.get(affected_port, 0)
        if prev_delay > self.cluster_threshold_hours:
            streak += 1
            if streak <= self.cluster_size:
                current_delay += self._rng.uniform(
                    self.cluster_delay_min_hours, self.cluster_delay_max_hours
                )
        else:
            streak = 0

        # Update state
        self._prev_delay_by_port[affected_port] = max(0.0, current_delay)
        self._delay_streak_by_port[affected_port] = streak

        # Return non-negative delay
        return float(max(0.0, current_delay))

    def apply_delay_to_arrival(self, shipment: Shipment) -> int:
        """
        Apply delay and return new arrival day.

        Modifies shipment.arrival_day in place.
        """
        delay_hours = self.apply_to_shipment(shipment)

        if delay_hours > 0:
            # Convert hours to days (round up)
            delay_days = int(np.ceil(delay_hours / 24.0))
            shipment.arrival_day += delay_days
            return delay_days

        return 0


# =============================================================================
# Optimism Bias Quirk (Task 6.3.2)
# =============================================================================


class OptimismBiasQuirk:
    """
    Over-forecast new products by a configurable percentage.

    Affects SKUs launched within the last N months.

    [Task 6.3.2] Optimism Bias
    [Intent: Section 7.A.3 - Human Optimism Bias]
    """

    def __init__(self, config: dict[str, Any]) -> None:
        quirk_config = (
            config.get("simulation_parameters", {})
            .get("quirks", {})
            .get("optimism_bias", {})
        )

        self.enabled = quirk_config.get("enabled", True)
        self.bias_pct = quirk_config.get("bias_pct", 0.15)
        self.affected_age_months = quirk_config.get("affected_age_months", 6)
        self.affected_age_days = self.affected_age_months * 30

        # Track product launch days (product_id -> launch_day)
        self.product_launch_days: dict[str, int] = {}

    def register_product_launch(self, product_id: str, launch_day: int) -> None:
        """Register when a product was launched."""
        self.product_launch_days[product_id] = launch_day

    def is_new_product(self, product_id: str, current_day: int) -> bool:
        """Check if a product is considered 'new' (within affected age)."""
        launch_day = self.product_launch_days.get(product_id, 0)
        age_days = current_day - launch_day
        return bool(age_days <= self.affected_age_days)

    def apply_to_forecast(
        self, forecast: float, product_id: str, current_day: int
    ) -> float:
        """
        Apply optimism bias to a demand forecast.

        Args:
            forecast: The base forecast quantity
            product_id: The product being forecasted
            current_day: Current simulation day

        Returns:
            Inflated forecast if product is new, otherwise unchanged
        """
        if not self.enabled:
            return float(forecast)

        if self.is_new_product(product_id, current_day):
            return float(forecast * (1 + self.bias_pct))

        return float(forecast)

    def apply_to_demand_tensor(
        self, demand: np.ndarray, product_ids: list[str], current_day: int
    ) -> np.ndarray:
        """
        Apply optimism bias to a full demand tensor.

        Args:
            demand: Shape [Nodes, Products] demand array
            product_ids: List of product IDs matching the product axis
            current_day: Current simulation day

        Returns:
            Modified demand array with bias applied to new products
        """
        if not self.enabled:
            return demand

        result = demand.copy()

        for p_idx, product_id in enumerate(product_ids):
            if self.is_new_product(product_id, current_day):
                result[:, p_idx] *= 1 + self.bias_pct

        return result


# =============================================================================
# Phantom Inventory Quirk (Task 6.3.3)
# =============================================================================


class PhantomInventoryQuirk:
    """
    Shrinkage with detection lag using dual inventory tracking.

    Dual Inventory Model:
    - actual_inventory: Physical reality (shrinkage applied immediately)
    - perceived_inventory: System view (shrinkage discovered after lag)

    [Task 6.3.3] Phantom Inventory
    [Intent: Section 7.A.4 - Phantom Inventory]
    """

    def __init__(self, config: dict[str, Any], seed: int = 42) -> None:
        quirk_config = (
            config.get("simulation_parameters", {})
            .get("quirks", {})
            .get("phantom_inventory", {})
        )

        self.enabled = quirk_config.get("enabled", True)
        self.shrinkage_pct = quirk_config.get("shrinkage_pct", 0.02)
        self.detection_lag_days = quirk_config.get("detection_lag_days", 14)
        self.shrinkage_factor_min = quirk_config.get("shrinkage_factor_min", 0.5)
        self.shrinkage_factor_max = quirk_config.get("shrinkage_factor_max", 1.0)

        self._rng = np.random.default_rng(seed)

        # Pending shrinkage events (day_discovered -> [events])
        self._pending_discoveries: dict[int, list[ShrinkageEvent]] = {}

        # Cumulative tracking
        self.total_shrinkage: float = 0.0
        self.total_discovered: float = 0.0

    def apply_shrinkage(self, state: StateManager, day: int) -> list[ShrinkageEvent]:
        """
        Apply daily shrinkage to actual inventory.

        Shrinkage occurs immediately on actual_inventory.
        Discovery is scheduled for detection_lag_days later.

        Returns list of shrinkage events created.
        """
        if not self.enabled:
            return []

        events: list[ShrinkageEvent] = []

        # Calculate number of cells to affect based on shrinkage_pct
        total_cells = state.inventory.size
        n_affected = max(1, int(total_cells * self.shrinkage_pct))

        # Random selection of (node, product) positions
        flat_indices = self._rng.choice(total_cells, size=n_affected, replace=False)

        for flat_idx in flat_indices:
            node_idx = int(flat_idx // state.n_products)
            prod_idx = int(flat_idx % state.n_products)

            # Get current inventory
            current_qty = state.inventory[node_idx, prod_idx]
            if current_qty <= 0:
                continue

            # Apply shrinkage (random between min and max factor of shrinkage_pct)
            shrink_factor = (
                self._rng.uniform(self.shrinkage_factor_min, self.shrinkage_factor_max)
                * self.shrinkage_pct
            )
            shrink_qty = current_qty * shrink_factor

            # Apply to actual inventory
            state.actual_inventory[node_idx, prod_idx] -= shrink_qty

            self.total_shrinkage += shrink_qty

            # Schedule discovery
            discovery_day = day + self.detection_lag_days
            node_id = state.node_idx_to_id.get(node_idx, f"node_{node_idx}")
            prod_id = state.product_idx_to_id.get(prod_idx, f"prod_{prod_idx}")

            event = ShrinkageEvent(
                day_occurred=day,
                day_discovered=discovery_day,
                node_id=node_id,
                product_id=prod_id,
                quantity_lost=shrink_qty,
            )
            events.append(event)

            if discovery_day not in self._pending_discoveries:
                self._pending_discoveries[discovery_day] = []
            self._pending_discoveries[discovery_day].append(event)

        return events

    def process_discoveries(
        self, state: StateManager, day: int
    ) -> list[ShrinkageEvent]:
        """
        Process shrinkage discoveries for the current day.

        After detection_lag_days, sync perceived_inventory to actual_inventory.

        Returns list of shrinkage events discovered today.
        """
        if not self.enabled:
            return []

        if day not in self._pending_discoveries:
            return []

        events = self._pending_discoveries.pop(day)

        for event in events:
            event.is_discovered = True
            self.total_discovered += event.quantity_lost

            # Sync perceived to actual for this position
            node_idx = state.node_id_to_idx.get(event.node_id)
            prod_idx = state.product_id_to_idx.get(event.product_id)

            if node_idx is not None and prod_idx is not None:
                # Sync perceived to actual for this cell
                actual_val = state.actual_inventory[node_idx, prod_idx]
                state.perceived_inventory[node_idx, prod_idx] = actual_val

        return events

    def get_inventory_discrepancy(self, state: StateManager) -> np.ndarray:
        """
        Returns perceived - actual (positive = phantom stock).

        Phantom stock is inventory the system thinks exists but doesn't.
        """
        # Note: These attrs added dynamically when phantom_inventory enabled
        perceived = state.perceived_inventory
        actual = state.actual_inventory
        result: np.ndarray = perceived - actual
        return result

    def get_total_phantom_stock(self, state: StateManager) -> float:
        """Returns total phantom stock across all positions."""
        discrepancy = self.get_inventory_discrepancy(state)
        return float(np.sum(np.maximum(0, discrepancy)))


# =============================================================================
# Quirk Manager (Unified Interface)
# =============================================================================


@dataclass
class QuirkManager:
    """
    Unified manager for all behavioral quirks.

    Provides a single interface for enabling/disabling quirks and applying them
    at the appropriate points in the simulation loop.

    [Task 6.3] Quirk Injection
    """

    config: dict[str, Any]
    seed: int = 42

    # Individual quirk engines
    port_congestion: PortCongestionQuirk = field(init=False)
    optimism_bias: OptimismBiasQuirk = field(init=False)
    phantom_inventory: PhantomInventoryQuirk = field(init=False)

    def __post_init__(self) -> None:
        """Initialize individual quirk engines."""
        self.port_congestion = PortCongestionQuirk(self.config, self.seed)
        self.optimism_bias = OptimismBiasQuirk(self.config)
        self.phantom_inventory = PhantomInventoryQuirk(self.config, self.seed + 1)

    def is_enabled(self, quirk_name: str) -> bool:
        """Check if a specific quirk is enabled."""
        if quirk_name == "port_congestion":
            return bool(self.port_congestion.enabled)
        elif quirk_name == "optimism_bias":
            return bool(self.optimism_bias.enabled)
        elif quirk_name == "phantom_inventory":
            return bool(self.phantom_inventory.enabled)
        return False

    def get_enabled_quirks(self) -> list[str]:
        """Get list of all enabled quirk names."""
        enabled = []
        if self.port_congestion.enabled:
            enabled.append("port_congestion")
        if self.optimism_bias.enabled:
            enabled.append("optimism_bias")
        if self.phantom_inventory.enabled:
            enabled.append("phantom_inventory")
        return enabled

    def apply_port_congestion(self, shipments: list[Shipment]) -> dict[str, int]:
        """
        Apply port congestion delays to shipments.

        Returns dict of shipment_id -> delay_days applied.
        """
        delays: dict[str, int] = {}
        for shipment in shipments:
            delay = self.port_congestion.apply_delay_to_arrival(shipment)
            if delay > 0:
                delays[shipment.id] = delay
        return delays

    def apply_optimism_bias(
        self, demand: np.ndarray, product_ids: list[str], current_day: int
    ) -> np.ndarray:
        """Apply optimism bias to demand forecast."""
        return self.optimism_bias.apply_to_demand_tensor(
            demand, product_ids, current_day
        )

    def apply_shrinkage(self, state: StateManager, day: int) -> list[ShrinkageEvent]:
        """Apply daily shrinkage to inventory."""
        return self.phantom_inventory.apply_shrinkage(state, day)

    def process_discoveries(
        self, state: StateManager, day: int
    ) -> list[ShrinkageEvent]:
        """Process shrinkage discoveries for the day."""
        return self.phantom_inventory.process_discoveries(state, day)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all quirks and their status."""
        return {
            "enabled_quirks": self.get_enabled_quirks(),
            "port_congestion": {
                "enabled": self.port_congestion.enabled,
                "ar_coefficient": self.port_congestion.ar_coefficient,
                "affected_ports": self.port_congestion.affected_ports,
            },
            "optimism_bias": {
                "enabled": self.optimism_bias.enabled,
                "bias_pct": self.optimism_bias.bias_pct,
                "affected_age_months": self.optimism_bias.affected_age_months,
            },
            "phantom_inventory": {
                "enabled": self.phantom_inventory.enabled,
                "shrinkage_pct": self.phantom_inventory.shrinkage_pct,
                "detection_lag_days": self.phantom_inventory.detection_lag_days,
                "total_shrinkage": self.phantom_inventory.total_shrinkage,
                "total_discovered": self.phantom_inventory.total_discovered,
            },
        }
