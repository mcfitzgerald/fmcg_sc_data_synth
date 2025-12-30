from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskEvent:
    event_code: str
    event_type: str  # port_strike, cyber_outage, etc.
    trigger_day: int
    duration_days: int
    parameters: dict[str, Any] = field(default_factory=dict)

    # State
    active: bool = False
    start_day: int | None = None


class RiskEventManager:
    """Manages deterministic risk events that disrupt the supply chain."""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("risk_events", {})
        self.enabled = self.config.get("enabled", True)

        # Load events from config
        self.events: list[RiskEvent] = []
        for e in self.config.get("events", []):
            self.events.append(
                RiskEvent(
                    event_code=e["code"],
                    event_type=e.get("type", "generic"),
                    trigger_day=e["trigger_day"],
                    duration_days=e["duration_days"],
                    parameters=e.get("parameters", {}),
                )
            )

        self.active_events: list[RiskEvent] = []

    def check_triggers(self, day: int) -> list[RiskEvent]:
        """Check if any events should start on this day."""
        if not self.enabled:
            return []

        triggered = []
        for event in self.events:
            if event.trigger_day == day and not event.active:
                event.active = True
                event.start_day = day
                self.active_events.append(event)
                triggered.append(event)
        return triggered

    def check_recovery(self, day: int) -> list[str]:
        """Check if any active events should end."""
        recovered = []
        remaining = []
        for event in self.active_events:
            if event.start_day is not None and (
                day - event.start_day >= event.duration_days
            ):
                event.active = False
                recovered.append(event.event_code)
            else:
                remaining.append(event)

        self.active_events = remaining
        return recovered

    def is_event_active(self, event_type: str) -> bool:
        """Check if any event of a specific type is active."""
        return any(e.event_type == event_type for e in self.active_events)

    def get_logistics_delay_multiplier(self) -> float:
        """
        Calculate cumulative delay multiplier from active PORT STRIKE events.
        """
        multiplier = 1.0
        if not self.enabled:
            return multiplier

        for e in self.active_events:
            if e.event_type == "port_strike":
                m = e.parameters.get("delay_multiplier", 1.0)
                multiplier *= m
        return multiplier

    def get_batch_overrides(self) -> dict[str, str]:
        """
        Get batch status overrides from active CONTAMINATION events.

        Returns:
            Dict of {ingredient_id: new_status} (e.g. {"ING-SORB": "rejected"})
        """
        overrides = {}
        if not self.enabled:
            return overrides

        for e in self.active_events:
            if e.event_type == "contamination":
                target = e.parameters.get("target_ingredient")
                status = e.parameters.get("qc_status_override")
                if target and status:
                    overrides[target] = status
        return overrides

    def get_supplier_overrides(self) -> dict[str, float]:
        """
        Get supplier reliability overrides from active SUPPLIER OPACITY events.

        Returns:
            Dict of {supplier_id: new_reliability_score}
        """
        overrides = {}
        if not self.enabled:
            return overrides

        for e in self.active_events:
            if e.event_type == "supplier_opacity":
                target = e.parameters.get("target_supplier")
                score = e.parameters.get("degraded_otd_rate")
                if target and score is not None:
                    overrides[target] = float(score)
        return overrides

    def get_dc_overrides(self) -> list[str]:
        """
        Get list of DCs affected by active CYBER OUTAGE events.

        Returns:
            List of DC IDs that are effectively down.
        """
        down_dcs = []
        if not self.enabled:
            return down_dcs

        for e in self.active_events:
            if e.event_type == "cyber_outage":
                target = e.parameters.get("target_dc")
                if target:
                    down_dcs.append(target)
        return down_dcs

    def get_emission_overrides(self) -> float:
        """
        Get CO2 cost multiplier from active CARBON TAX events.
        """
        multiplier = 1.0
        if not self.enabled:
            return multiplier

        for e in self.active_events:
            if e.event_type == "carbon_tax_spike":
                m = e.parameters.get("co2_cost_multiplier", 1.0)
                multiplier *= m
        return multiplier
