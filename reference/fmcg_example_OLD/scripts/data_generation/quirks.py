"""
QuirksManager - Pathology injection for realistic behavioral anomalies.

Injects realistic supply chain pathologies into generated data:
- Bullwhip "Whip Crack" - Order batching during promos
- Phantom Inventory - Shrinkage creating inventory discrepancy
- Port Congestion Flicker - Autoregressive clustered delays
- Single Source Fragility - SPOF cascade effects
- Human Optimism Bias - Over-forecasting new products
- Data Decay - Older batches have higher rejection rates

All quirks are enabled by default (per manifest) to maximize data realism.

Usage:
    from data_generation import QuirksManager

    manager = QuirksManager(manifest_path, seed=42)

    # Check if quirk is enabled
    if manager.is_enabled("phantom_inventory"):
        inventory = manager.apply_phantom_inventory(inventory)

    # Get all enabled quirks
    enabled = manager.get_enabled_quirks()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Quirk:
    """
    A behavioral pathology to inject into generation.

    Attributes:
        name: Unique identifier (e.g., "bullwhip_whip_crack")
        description: Human-readable description
        enabled: Whether this quirk is active
        parameters: Dict of quirk-specific parameters
        affected_tables: List of tables this quirk modifies
    """

    name: str
    description: str
    enabled: bool
    parameters: dict[str, Any]
    affected_tables: list[str] = field(default_factory=list)


@dataclass
class QuirksManager:
    """
    Injects realistic pathologies into generated data.

    Loads quirk definitions from BenchmarkManifest.json and provides
    methods to apply each quirk to the appropriate data tables.

    Attributes:
        manifest_path: Path to BenchmarkManifest.json
        seed: Random seed for reproducibility
        quirks: Dict of quirk_name -> Quirk
    """

    manifest_path: Path
    seed: int = 42
    quirks: dict[str, Quirk] = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False, repr=False)
    _manifest: dict = field(init=False, repr=False)

    # Mapping of quirk names to affected tables
    QUIRK_TABLE_MAP: dict[str, list[str]] = field(default_factory=lambda: {
        "bullwhip_whip_crack": ["orders", "order_lines"],
        "phantom_inventory": ["inventory"],
        "port_congestion_flicker": ["shipment_legs"],
        "single_source_fragility": ["batches", "work_orders"],
        "human_optimism_bias": ["demand_forecasts"],
        "data_decay": ["batches"],
    })

    def __post_init__(self) -> None:
        """Initialize RNG and load quirks from manifest."""
        self._rng = np.random.default_rng(self.seed)
        self._load_manifest()
        self._load_quirks()

    def _load_manifest(self) -> None:
        """Load the benchmark manifest JSON."""
        with open(self.manifest_path) as f:
            self._manifest = json.load(f)

    def _load_quirks(self) -> None:
        """Parse quirks section from manifest into Quirk objects."""
        quirks_config = self._manifest.get("quirks", {})

        for name, config in quirks_config.items():
            # Extract parameters (everything except description and enabled)
            params = {k: v for k, v in config.items()
                      if k not in ("description", "enabled")}

            self.quirks[name] = Quirk(
                name=name,
                description=config.get("description", ""),
                enabled=config.get("enabled", True),  # Default to enabled
                parameters=params,
                affected_tables=self.QUIRK_TABLE_MAP.get(name, []),
            )

    def is_enabled(self, quirk_name: str) -> bool:
        """Check if a specific quirk is enabled."""
        quirk = self.quirks.get(quirk_name)
        return quirk.enabled if quirk else False

    def get_quirk(self, quirk_name: str) -> Quirk | None:
        """Get a specific quirk by name."""
        return self.quirks.get(quirk_name)

    def get_enabled_quirks(self) -> list[str]:
        """Get list of all enabled quirk names."""
        return [name for name, quirk in self.quirks.items() if quirk.enabled]

    def get_params(self, quirk_name: str) -> dict[str, Any]:
        """Get parameters for a specific quirk."""
        quirk = self.quirks.get(quirk_name)
        return quirk.parameters if quirk else {}

    # =========================================================================
    # Quirk Implementations
    # =========================================================================

    def apply_bullwhip(
        self,
        orders: list[dict],
        order_lines: list[dict],
        promo_order_ids: set[int],
    ) -> tuple[list[dict], list[dict]]:
        """
        Apply bullwhip "whip crack" effect during promo weeks.

        Instead of 10 small orders, customers place 1 large order (batching).
        This increases order quantity variance relative to POS variance.

        Args:
            orders: List of order dicts
            order_lines: List of order line dicts
            promo_order_ids: Set of order IDs that are promo-related

        Returns:
            Tuple of (modified_orders, modified_order_lines)
        """
        if not self.is_enabled("bullwhip_whip_crack"):
            return orders, order_lines

        params = self.get_params("bullwhip_whip_crack")
        batching_factor = params.get("batching_factor", 3.0)

        # Find promo orders and increase their quantities
        order_id_to_order = {o["id"]: o for o in orders}

        # Mark orders as batched
        for order_id in promo_order_ids:
            if order_id in order_id_to_order:
                order = order_id_to_order[order_id]
                existing_notes = order.get("notes") or ""
                order["notes"] = f"{existing_notes} [Quirk: Bullwhip batched]".strip()

        for line in order_lines:
            if line.get("order_id") in promo_order_ids:
                # Increase quantity by batching factor
                original_qty = line.get("quantity_cases", 1)
                line["quantity_cases"] = int(original_qty * batching_factor)

                # Recalculate line amount if present
                if "unit_price" in line and "line_amount" in line:
                    line["line_amount"] = line["quantity_cases"] * line["unit_price"]

        return orders, order_lines

    def apply_phantom_inventory(
        self,
        inventory: list[dict],
        reference_date: datetime | None = None,
    ) -> list[dict]:
        """
        Inject phantom inventory shrinkage.

        Creates 2% discrepancy where actual inventory is less than recorded.
        Shrinkage is discovered after a detection lag.

        Args:
            inventory: List of inventory dicts
            reference_date: Current date for lag calculation

        Returns:
            Modified inventory list
        """
        if not self.is_enabled("phantom_inventory"):
            return inventory

        params = self.get_params("phantom_inventory")
        shrinkage_pct = params.get("shrinkage_pct", 0.02)
        detection_lag_days = params.get("detection_lag_days", 14)

        if reference_date is None:
            reference_date = datetime.now()

        # Apply shrinkage to random subset of inventory
        n_affected = int(len(inventory) * shrinkage_pct)
        affected_indices = self._rng.choice(
            len(inventory), size=n_affected, replace=False
        )

        for idx in affected_indices:
            inv = inventory[idx]

            # Record original quantity (for audit)
            original_qty = inv.get("quantity_cases", 0)

            # Apply random shrinkage (50-100% of the shrinkage_pct)
            shrink_amount = self._rng.uniform(0.5, 1.0) * shrinkage_pct
            new_qty = int(original_qty * (1 - shrink_amount))
            inv["quantity_cases"] = max(0, new_qty)

            # Update eaches if present
            if "quantity_eaches" in inv and "case_pack" in inv:
                case_pack = inv.get("case_pack", 12)
                inv["quantity_eaches"] = inv["quantity_cases"] * case_pack

            # Mark as having shrinkage (for validation)
            inv["has_shrinkage"] = True
            inv["shrinkage_amount"] = original_qty - new_qty

            # Set discovery date (for temporal queries)
            if "created_at" in inv:
                created = inv["created_at"]
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                discovery_date = created + timedelta(days=detection_lag_days)
                inv["shrinkage_discovered_date"] = discovery_date.isoformat()

        return inventory

    def apply_port_congestion(
        self,
        shipment_legs: list[dict],
        affected_ports: list[str] | None = None,
        segment_lookup: dict[int, dict] | None = None,
        port_ids: dict[str, int] | None = None,
    ) -> list[dict]:
        """
        Apply AR(1) autoregressive delays for port congestion.

        When one shipment is late, subsequent shipments through the same
        port are also likely to be late (clustering effect).

        Args:
            shipment_legs: List of shipment leg dicts
            affected_ports: List of port codes to affect (default from manifest)
            segment_lookup: Dict mapping segment_id -> segment dict (for looking up port info)
            port_ids: Dict mapping port_code -> port_id

        Returns:
            Modified shipment legs
        """
        if not self.is_enabled("port_congestion_flicker"):
            return shipment_legs

        params = self.get_params("port_congestion_flicker")
        ar_coef = params.get("autoregressive_coefficient", 0.70)
        cluster_size = params.get("cluster_size", 3)

        if affected_ports is None:
            affected_ports = params.get("affected_ports", ["USLAX", "CNSHA"])

        # Build affected port IDs set if we have port_ids mapping
        affected_port_ids = set()
        if port_ids:
            affected_port_ids = {port_ids.get(code) for code in affected_ports if code in port_ids}

        # Group legs by port
        port_legs: dict[str, list[dict]] = {}
        for leg in shipment_legs:
            matched_port = None

            # Try segment-based lookup first (preferred)
            if segment_lookup and port_ids:
                seg_id = leg.get("route_segment_id")
                if seg_id and seg_id in segment_lookup:
                    seg = segment_lookup[seg_id]
                    # Check origin
                    if seg.get("origin_type") == "port" and seg.get("origin_id") in affected_port_ids:
                        # Find port code from ID
                        for code, pid in port_ids.items():
                            if pid == seg.get("origin_id"):
                                matched_port = code
                                break
                    # Check destination
                    if not matched_port and seg.get("destination_type") == "port" and seg.get("destination_id") in affected_port_ids:
                        for code, pid in port_ids.items():
                            if pid == seg.get("destination_id"):
                                matched_port = code
                                break
            else:
                # Fallback to direct origin_code/destination_code on leg
                origin = leg.get("origin_code", "")
                dest = leg.get("destination_code", "")
                for port in affected_ports:
                    if port in origin or port in dest:
                        matched_port = port
                        break

            if matched_port:
                if matched_port not in port_legs:
                    port_legs[matched_port] = []
                port_legs[matched_port].append(leg)

        # Apply AR(1) correlation within each port's legs
        for port, legs in port_legs.items():
            # Sort by planned departure for temporal ordering
            legs.sort(key=lambda x: x.get("departure_datetime", x.get("planned_departure", "")))

            prev_delay = 0.0
            delay_streak = 0

            for leg in legs:
                # AR(1): current delay = ar_coef * prev_delay + noise
                noise = self._rng.normal(0, 2)  # 2-hour std dev
                current_delay = ar_coef * prev_delay + noise

                # If in a delay streak, amplify
                if prev_delay > 4:  # Previous leg was >4 hours late
                    delay_streak += 1
                    if delay_streak <= cluster_size:
                        current_delay += self._rng.uniform(2, 6)
                else:
                    delay_streak = 0

                # Apply delay
                if current_delay > 0:
                    leg["delay_hours"] = leg.get("delay_hours", 0) + max(0, current_delay)
                    leg["congestion_affected"] = True
                    
                    existing_notes = leg.get("notes") or ""
                    if "congestion" not in existing_notes:
                        leg["notes"] = f"{existing_notes} [Quirk: Port congestion correlated delay]".strip()

                prev_delay = current_delay

        return shipment_legs

    def apply_single_source_fragility(
        self,
        batches: list[dict],
        ingredient_to_supplier: dict[str, list[str]],
        spof_ingredients: list[str] | None = None,
    ) -> list[dict]:
        """
        Apply lead time cascade for single-source ingredients.

        When a SPOF ingredient is used, batches experience 2.5x lead time
        delays due to supply chain fragility.

        Args:
            batches: List of batch dicts
            ingredient_to_supplier: Mapping of ingredient_id -> [supplier_ids]
            spof_ingredients: List of single-source ingredient codes

        Returns:
            Modified batches
        """
        if not self.is_enabled("single_source_fragility"):
            return batches

        params = self.get_params("single_source_fragility")
        delay_multiplier = params.get("cascade_delay_multiplier", 2.5)

        # Identify SPOF ingredients (single supplier)
        if spof_ingredients is None:
            spof_ingredients = [
                ing_id for ing_id, suppliers in ingredient_to_supplier.items()
                if len(suppliers) == 1
            ]

        # Apply delay to batches using SPOF ingredients
        for batch in batches:
            batch_ingredients = batch.get("ingredient_ids", [])

            # Check if any SPOF ingredient is used
            uses_spof = any(ing in spof_ingredients for ing in batch_ingredients)

            if uses_spof:
                # Increase manufacturing lead time
                if "lead_time_days" in batch:
                    batch["lead_time_days"] = int(batch["lead_time_days"] * delay_multiplier)

                # Delay completion date if present
                if "planned_completion" in batch and "actual_completion" in batch:
                    planned = batch["planned_completion"]
                    if isinstance(planned, str):
                        planned = datetime.fromisoformat(planned.replace("Z", "+00:00"))

                    # Add delay
                    delay_days = int(3 * (delay_multiplier - 1))  # 3-day base * multiplier
                    delayed = planned + timedelta(days=delay_days)
                    batch["actual_completion"] = delayed.isoformat()

                batch["spof_affected"] = True

        return batches

    def apply_optimism_bias(
        self,
        forecasts: list[dict],
        sku_launch_dates: dict[int, datetime],
        reference_date: datetime | None = None,
    ) -> list[dict]:
        """
        Apply human optimism bias to new product forecasts.

        Planners over-forecast new product launches by 15%.

        Args:
            forecasts: List of demand forecast dicts
            sku_launch_dates: Mapping of sku_id -> launch date
            reference_date: Current date for age calculation

        Returns:
            Modified forecasts
        """
        if not self.is_enabled("human_optimism_bias"):
            return forecasts

        params = self.get_params("human_optimism_bias")
        bias_pct = params.get("bias_pct", 0.15)
        age_threshold_months = params.get("affected_sku_age_months", 6)

        if reference_date is None:
            reference_date = datetime.now()

        # Find SKUs that are "new" (launched within threshold)
        new_sku_ids = set()
        for sku_id, launch_date in sku_launch_dates.items():
            if isinstance(launch_date, str):
                launch_date = datetime.fromisoformat(launch_date.replace("Z", "+00:00"))

            age_months = (reference_date - launch_date).days / 30
            if age_months <= age_threshold_months:
                new_sku_ids.add(sku_id)

        # Apply bias to forecasts for new SKUs
        for forecast in forecasts:
            sku_id = forecast.get("sku_id")

            if sku_id in new_sku_ids:
                # Inflate forecast quantity (check multiple field names)
                for field in ("final_forecast", "forecast_quantity", "consensus_forecast"):
                    if field in forecast:
                        original = forecast[field]
                        forecast[field] = int(original * (1 + bias_pct))
                        forecast["optimism_bias_applied"] = True
                        forecast["original_forecast"] = original
                        break  # Only inflate one field

        return forecasts

    def apply_data_decay(
        self,
        batches: list[dict],
        reference_date: datetime | None = None,
    ) -> list[dict]:
        """
        Apply data decay - older batches have higher QC rejection probability.

        Batches approaching expiry have 8% rejection rate vs 2% base.

        Args:
            batches: List of batch dicts
            reference_date: Current date for age calculation

        Returns:
            Modified batches
        """
        if not self.is_enabled("data_decay"):
            return batches

        params = self.get_params("data_decay")
        base_rate = params.get("base_rejection_rate", 0.02)
        elevated_rate = params.get("elevated_rejection_rate", 0.08)
        threshold_days = params.get("days_to_expiry_threshold", 30)

        if reference_date is None:
            reference_date = datetime.now()

        for batch in batches:
            # Get expiry date
            expiry = batch.get("expiry_date")
            if not expiry:
                continue

            if isinstance(expiry, str):
                expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00")).date()
            elif isinstance(expiry, datetime):
                expiry = expiry.date()
            # expiry is now a date object

            # Calculate days to expiry (compare dates, not datetimes)
            ref_date = reference_date.date() if isinstance(reference_date, datetime) else reference_date
            days_to_expiry = (expiry - ref_date).days

            # Determine rejection rate based on age
            if days_to_expiry <= threshold_days:
                rejection_rate = elevated_rate
            else:
                rejection_rate = base_rate

            # Apply rejection probabilistically
            if self._rng.random() < rejection_rate:
                batch["qc_status"] = "rejected"
                batch["rejection_reason"] = "quality_decay" if days_to_expiry <= threshold_days else "random_sample"
                batch["data_decay_affected"] = True

        return batches

    # =========================================================================
    # Summary and reporting
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all quirks and their status."""
        return {
            "total_quirks": len(self.quirks),
            "enabled_count": len(self.get_enabled_quirks()),
            "enabled_quirks": self.get_enabled_quirks(),
            "quirks": {
                name: {
                    "description": quirk.description,
                    "enabled": quirk.enabled,
                    "affected_tables": quirk.affected_tables,
                    "parameters": quirk.parameters,
                }
                for name, quirk in self.quirks.items()
            },
        }

    def __repr__(self) -> str:
        enabled = len(self.get_enabled_quirks())
        total = len(self.quirks)
        return f"QuirksManager({enabled}/{total} quirks enabled)"
