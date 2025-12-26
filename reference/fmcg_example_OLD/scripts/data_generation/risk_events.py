"""
RiskEventManager - Chaos injection engine for supply chain disruptions.

Triggers deterministic risk events based on manifest definitions to create
realistic supply chain chaos for testing beast mode queries (recall trace,
SPOF detection, etc.).

Events:
- RSK-BIO-001: Contamination (Sorbitol recall)
- RSK-LOG-002: Port strike (LA port, 4x delays with Gamma distribution)
- RSK-SUP-003: Supplier opacity (Palm Oil SPOF, OTD drops to 40%)
- RSK-CYB-004: Cyber outage (Chicago DC, pick waves ON_HOLD)
- RSK-ENV-005: Carbon tax spike (3x CO2 multiplier)

Usage:
    from data_generation import RiskEventManager

    manager = RiskEventManager(manifest_path, seed=42)
    triggered = manager.trigger_all()  # Deterministically trigger all events

    # Check for specific events
    if manager.is_triggered("RSK-BIO-001"):
        overrides = manager.get_batch_overrides()
        # Apply qc_status='REJECTED' to affected batches

    # Get stochastic mode for port strike
    mode = manager.get_stochastic_mode()  # Returns DISRUPTED if RSK-LOG-002 active
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .realism_monitor import StochasticMode


@dataclass
class RiskEvent:
    """
    A triggered risk event with its configuration.

    Attributes:
        event_code: Unique identifier (e.g., "RSK-BIO-001")
        event_type: Category of event (contamination, port_strike, etc.)
        description: Human-readable description
        affected_entity: Primary entity affected (e.g., "ING-SORB-001")
        parameter_overrides: Dict of parameters to override during generation
        probability_per_run: Probability of triggering (1.0 = always)
        is_triggered: Whether the event has been triggered this run
    """

    event_code: str
    event_type: str
    description: str
    affected_entity: str | None
    parameter_overrides: dict[str, Any]
    probability_per_run: float = 1.0
    is_triggered: bool = False


@dataclass
class RiskEventManager:
    """
    Orchestrates chaos injection based on manifest definitions.

    Loads risk events from BenchmarkManifest.json and provides methods
    to trigger events and retrieve override configurations for each
    generation level.

    Attributes:
        manifest_path: Path to BenchmarkManifest.json
        seed: Random seed for reproducibility
        events: Dict of event_code -> RiskEvent
        triggered_events: List of event codes that have been triggered
    """

    manifest_path: Path
    seed: int = 42
    events: dict[str, RiskEvent] = field(default_factory=dict)
    triggered_events: list[str] = field(default_factory=list)
    _rng: np.random.Generator = field(init=False, repr=False)
    _manifest: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize RNG and load events from manifest."""
        self._rng = np.random.default_rng(self.seed)
        self._load_manifest()
        self._load_events()

    def _load_manifest(self) -> None:
        """Load the benchmark manifest JSON."""
        with open(self.manifest_path) as f:
            self._manifest = json.load(f)

    def _load_events(self) -> None:
        """Parse risk_events section from manifest into RiskEvent objects."""
        risk_events = self._manifest.get("risk_events", {})

        for event_code, config in risk_events.items():
            # Determine affected entity based on event type
            affected_entity = self._extract_affected_entity(config)

            # Build parameter overrides dict
            overrides = {k: v for k, v in config.items()
                         if k not in ("event_type", "description", "probability_per_run")}

            self.events[event_code] = RiskEvent(
                event_code=event_code,
                event_type=config.get("event_type", "unknown"),
                description=config.get("description", ""),
                affected_entity=affected_entity,
                parameter_overrides=overrides,
                probability_per_run=config.get("probability_per_run", 1.0),
                is_triggered=False,
            )

    def _extract_affected_entity(self, config: dict) -> str | None:
        """Extract the primary affected entity from event config."""
        # Check for various entity reference fields
        for field in ("target_ingredient", "target_supplier", "target_dc", "affected_ports"):
            if field in config:
                value = config[field]
                # Handle list (e.g., affected_ports)
                if isinstance(value, list):
                    return value[0] if value else None
                return value
        return None

    def trigger_all(self) -> list[str]:
        """
        Trigger all risk events deterministically for testing.

        All events fire regardless of probability_per_run. This ensures
        consistent chaos for validating beast mode queries.

        Returns:
            List of triggered event codes
        """
        self.triggered_events = []
        for event_code, event in self.events.items():
            event.is_triggered = True
            self.triggered_events.append(event_code)
        return self.triggered_events

    def trigger_probabilistic(self) -> list[str]:
        """
        Trigger events based on their probability_per_run.

        Alternative to trigger_all() for more realistic scenarios.

        Returns:
            List of triggered event codes
        """
        self.triggered_events = []
        for event_code, event in self.events.items():
            if self._rng.random() < event.probability_per_run:
                event.is_triggered = True
                self.triggered_events.append(event_code)
        return self.triggered_events

    def is_triggered(self, event_code: str) -> bool:
        """Check if a specific event has been triggered."""
        return event_code in self.triggered_events

    def get_event(self, event_code: str) -> RiskEvent | None:
        """Get a specific event by code."""
        return self.events.get(event_code)

    def get_triggered_events(self) -> list[RiskEvent]:
        """Get all triggered RiskEvent objects."""
        return [self.events[code] for code in self.triggered_events]

    # =========================================================================
    # Level-specific override getters
    # =========================================================================

    def get_stochastic_mode(self) -> StochasticMode:
        """
        Return stochastic mode based on triggered events.

        Returns DISRUPTED if port_strike (RSK-LOG-002) is triggered,
        enabling Gamma distribution for fat-tail delays.
        """
        if self.is_triggered("RSK-LOG-002"):
            return StochasticMode.DISRUPTED
        return StochasticMode.NORMAL

    def get_emission_overrides(self) -> dict[str, Any]:
        """
        Get emission factor overrides for carbon tax event (RSK-ENV-005).

        Returns:
            Dict with co2_multiplier if event triggered, empty dict otherwise
        """
        if not self.is_triggered("RSK-ENV-005"):
            return {}

        event = self.events["RSK-ENV-005"]
        return {
            "co2_multiplier": event.parameter_overrides.get("co2_cost_multiplier", 3.0),
            "triggers_modal_shift": event.parameter_overrides.get("triggers_modal_shift", True),
        }

    def get_supplier_overrides(self) -> dict[str, Any]:
        """
        Get supplier ingredient overrides for opacity event (RSK-SUP-003).

        Returns:
            Dict with target_supplier and degraded OTD rate
        """
        if not self.is_triggered("RSK-SUP-003"):
            return {}

        event = self.events["RSK-SUP-003"]
        return {
            "target_supplier": event.parameter_overrides.get("target_supplier", "SUP-PALM-MY-001"),
            "degraded_otd_rate": event.parameter_overrides.get("degraded_otd_rate", 0.40),
            "normal_otd_rate": event.parameter_overrides.get("normal_otd_rate", 0.92),
        }

    def get_batch_overrides(self) -> dict[str, Any]:
        """
        Get batch QC overrides for contamination event (RSK-BIO-001).

        Returns:
            Dict with target_ingredient and qc_status override
        """
        if not self.is_triggered("RSK-BIO-001"):
            return {}

        event = self.events["RSK-BIO-001"]
        return {
            "target_ingredient": event.parameter_overrides.get("target_ingredient", "ING-SORB-001"),
            "qc_status_override": event.parameter_overrides.get("qc_status_override", "REJECTED"),
            "return_lines_target": event.parameter_overrides.get("return_lines_target", 500),
        }

    def get_dc_overrides(self) -> dict[str, Any]:
        """
        Get DC/pick wave overrides for cyber outage event (RSK-CYB-004).

        Returns:
            Dict with target_dc and hold duration
        """
        if not self.is_triggered("RSK-CYB-004"):
            return {}

        event = self.events["RSK-CYB-004"]
        return {
            "target_dc": event.parameter_overrides.get("target_dc", "DC-NAM-CHI-001"),
            "hold_duration_hours": event.parameter_overrides.get("hold_duration_hours", 72),
            "inventory_visibility_pct": event.parameter_overrides.get("inventory_visibility_pct", 0.0),
        }

    def get_port_strike_overrides(self) -> dict[str, Any]:
        """
        Get port strike parameters for logistics disruption (RSK-LOG-002).

        Returns:
            Dict with affected ports, delay multiplier, and distribution params
        """
        if not self.is_triggered("RSK-LOG-002"):
            return {}

        event = self.events["RSK-LOG-002"]
        return {
            "affected_ports": event.parameter_overrides.get("affected_ports", ["USLAX"]),
            "delay_multiplier": event.parameter_overrides.get("delay_multiplier", 4.0),
            "delay_distribution": event.parameter_overrides.get("delay_distribution", "gamma"),
            "gamma_shape": event.parameter_overrides.get("gamma_shape", 2.0),
            "duration_days": event.parameter_overrides.get("duration_days", 14),
        }

    # =========================================================================
    # Summary and reporting
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all events and their trigger status."""
        return {
            "total_events": len(self.events),
            "triggered_count": len(self.triggered_events),
            "triggered_events": self.triggered_events,
            "events": {
                code: {
                    "type": event.event_type,
                    "description": event.description,
                    "affected_entity": event.affected_entity,
                    "is_triggered": event.is_triggered,
                }
                for code, event in self.events.items()
            },
        }

    def __repr__(self) -> str:
        triggered = len(self.triggered_events)
        total = len(self.events)
        return f"RiskEventManager({triggered}/{total} events triggered)"
