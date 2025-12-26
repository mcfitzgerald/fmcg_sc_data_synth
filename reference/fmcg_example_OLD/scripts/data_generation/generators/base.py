"""
Base classes for modular level generators.

Phase A of the generate_data.py refactor (see effervescent-shimmying-blum.md).

This module provides:
- GeneratorContext: Shared state dataclass passed to all level generators
- BaseLevelGenerator: Abstract base class for level-specific generators

Design Principles:
- Context owns all mutable state (ID tracking dicts, data storage)
- Generators are stateless functions that read/write to context
- LookupCache lifecycle managed by context.build_cache(level)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from faker import Faker

# Import performance modules from parent package
from ..bottleneck_fixes import LookupCache, PooledFaker
from ..promo_calendar import PromoCalendar
from ..quirks import QuirksManager
from ..realism_monitor import RealismMonitor, StochasticMode
from ..risk_events import RiskEventManager
from ..static_pool import StaticDataPool


@dataclass
class GeneratorContext:
    """
    Shared state for all level generators.

    This dataclass holds all mutable state that was previously scattered across
    the FMCGDataGenerator class. Generators receive this context and can:
    - Read/write to the shared data dict
    - Update ID tracking dicts for referential integrity
    - Access performance modules (pool, realism_monitor, etc.)
    - Request lookup cache builds for their level

    Attributes:
        seed: Random seed for reproducibility
        rng: NumPy random generator
        fake: Faker instance for realistic data
        pooled_faker: Batch Faker sampling wrapper
        base_year: Current generation year (e.g., 2024)
        named_entities: Deterministic testing entities (recalls, accounts, etc.)
        generated_levels: Set of levels already generated
        data: Shared data storage - table name -> list of row dicts
        pool: StaticDataPool for O(1) vectorized sampling
        realism_monitor: Online streaming validation
        risk_manager: Risk event triggering and chaos injection
        quirks_manager: Behavioral pathology injection
        stochastic_mode: Normal vs disrupted distribution mode
        triggered_events: List of triggered risk events

    ID Tracking Dicts (for referential integrity):
        Level 0: division_ids, channel_ids, product_ids, packaging_type_ids,
                 port_ids, carrier_ids, emission_factor_ids, kpi_threshold_ids,
                 business_rule_ids, ingredient_ids
        Level 1: supplier_ids, plant_ids, production_line_ids,
                 carrier_contract_ids, route_segment_ids
        Level 2: supplier_ingredient_ids, certification_ids, formula_ids,
                 carrier_rate_ids, route_ids
        Level 3: retail_account_ids, retail_location_ids, dc_ids
        Level 4: sku_ids, promotion_ids
        Level 5: purchase_order_ids, goods_receipt_ids, work_order_ids
        Level 6-7: batch_ids, inventory_ids
        Level 8-9: order_ids, demand_forecast_ids, pick_wave_ids
        Level 10-11: shipment_ids
        Level 12-13: rma_ids, return_ids
    """

    # ==========================================================================
    # Core Random State
    # ==========================================================================
    seed: int
    rng: np.random.Generator
    fake: Faker
    pooled_faker: PooledFaker
    base_year: int

    # ==========================================================================
    # Named Entities (Deterministic Testing - Section 4.8)
    # ==========================================================================
    named_entities: dict[str, Any]

    # ==========================================================================
    # Generation Tracking
    # ==========================================================================
    generated_levels: set[int] = field(default_factory=set)

    # ==========================================================================
    # Shared Data Storage
    # ==========================================================================
    # Each key is a table name, value is list of row dicts
    data: dict[str, list[dict]] = field(default_factory=dict)

    # ==========================================================================
    # Performance Modules
    # ==========================================================================
    pool: StaticDataPool | None = None
    realism_monitor: RealismMonitor | None = None
    risk_manager: RiskEventManager | None = None
    quirks_manager: QuirksManager | None = None
    stochastic_mode: StochasticMode = StochasticMode.NORMAL
    triggered_events: list[Any] = field(default_factory=list)

    # Promo calendar (built lazily for Level 8)
    promo_calendar: PromoCalendar | None = None

    # ==========================================================================
    # ID Tracking Dicts - Level 0: Reference Data
    # ==========================================================================
    division_ids: dict[str, int] = field(default_factory=dict)
    channel_ids: dict[str, int] = field(default_factory=dict)
    product_ids: dict[str, int] = field(default_factory=dict)
    packaging_type_ids: dict[str, int] = field(default_factory=dict)
    port_ids: dict[str, int] = field(default_factory=dict)
    carrier_ids: dict[str, int] = field(default_factory=dict)
    emission_factor_ids: dict[str, int] = field(default_factory=dict)
    kpi_threshold_ids: dict[str, int] = field(default_factory=dict)
    business_rule_ids: dict[str, int] = field(default_factory=dict)
    ingredient_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 1: Master Data
    # ==========================================================================
    supplier_ids: dict[str, int] = field(default_factory=dict)
    plant_ids: dict[str, int] = field(default_factory=dict)
    production_line_ids: dict[str, int] = field(default_factory=dict)
    carrier_contract_ids: dict[str, int] = field(default_factory=dict)
    route_segment_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 2: Relationships and Formulas
    # ==========================================================================
    supplier_ingredient_ids: dict[tuple, int] = field(default_factory=dict)
    certification_ids: dict[str, int] = field(default_factory=dict)
    formula_ids: dict[str, int] = field(default_factory=dict)
    carrier_rate_ids: dict[str, int] = field(default_factory=dict)
    route_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 3: Locations
    # ==========================================================================
    retail_account_ids: dict[str, int] = field(default_factory=dict)
    retail_location_ids: dict[str, int] = field(default_factory=dict)
    dc_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 4: SKUs and Promotions
    # ==========================================================================
    sku_ids: dict[str, int] = field(default_factory=dict)
    promotion_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 5: Orders/POs
    # ==========================================================================
    purchase_order_ids: dict[str, int] = field(default_factory=dict)
    goods_receipt_ids: dict[str, int] = field(default_factory=dict)
    work_order_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 6-7: Manufacturing
    # ==========================================================================
    batch_ids: dict[str, int] = field(default_factory=dict)
    inventory_ids: dict[tuple, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 8-9: Demand and Orders
    # ==========================================================================
    order_ids: dict[str, int] = field(default_factory=dict)
    demand_forecast_ids: dict[tuple, int] = field(default_factory=dict)
    pick_wave_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 10-11: Shipments
    # ==========================================================================
    shipment_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # ID Tracking Dicts - Level 12-13: Returns
    # ==========================================================================
    rma_ids: dict[str, int] = field(default_factory=dict)
    return_ids: dict[str, int] = field(default_factory=dict)

    # ==========================================================================
    # LookupCache - Managed by context, built lazily per level
    # ==========================================================================
    _lookup_cache: LookupCache | None = field(default=None, repr=False)

    # ==========================================================================
    # Performance Tracking
    # ==========================================================================
    _level_times: dict[int, float] = field(default_factory=dict, repr=False)
    _level_rows: dict[int, int] = field(default_factory=dict, repr=False)

    def build_cache(self, level: int) -> None:
        """
        Build lookup indices needed for the given level.

        Cache is built lazily when a level needs FK lookups that would
        otherwise be O(N) list comprehensions.

        Level 6 needs: PO lines by PO ID, formula ingredients by formula ID
        Level 8 needs: retail locations by account ID, promotions by date
        Level 9 needs: order lines by order ID
        Level 10 needs: batches by SKU ID

        Args:
            level: Generation level (0-14)
        """
        if level in (6, 7, 8, 9, 10, 11):
            # Build full cache from current data state
            if self._lookup_cache is None:
                self._lookup_cache = LookupCache()
            self._lookup_cache.build_from_data(self.data)

    @property
    def lookup_cache(self) -> LookupCache | None:
        """Get the current lookup cache, or None if not built."""
        return self._lookup_cache

    def init_data_tables(self) -> None:
        """Initialize empty lists for all 67 tables."""
        tables = [
            # Level 0
            "divisions",
            "channels",
            "products",
            "packaging_types",
            "ports",
            "carriers",
            "emission_factors",
            "kpi_thresholds",
            "business_rules",
            "ingredients",
            # Level 1
            "suppliers",
            "plants",
            "production_lines",
            "carrier_contracts",
            "route_segments",
            # Level 2
            "supplier_ingredients",
            "certifications",
            "formulas",
            "formula_ingredients",
            "carrier_rates",
            "routes",
            "route_segment_assignments",
            # Level 3
            "retail_accounts",
            "retail_locations",
            "distribution_centers",
            # Level 4
            "skus",
            "sku_costs",
            "sku_substitutes",
            "promotions",
            "promotion_skus",
            "promotion_accounts",
            # Level 5
            "purchase_orders",
            "purchase_order_lines",
            "goods_receipts",
            "goods_receipt_lines",
            "work_orders",
            "work_order_materials",
            # Level 6
            "batches",
            "batch_ingredients",
            "batch_cost_ledger",
            # Level 7
            "inventory",
            # Level 8
            "pos_sales",
            "demand_forecasts",
            "forecast_accuracy",
            "consensus_adjustments",
            # Level 9
            "orders",
            "order_lines",
            "order_allocations",
            "replenishment_params",
            "demand_allocations",
            "capacity_plans",
            "supply_plans",
            "plan_exceptions",
            # Level 10
            "shipments",
            "shipment_lines",
            "shipment_legs",
            "shipment_emissions",
            "pick_waves",
            "pick_wave_orders",
            # Level 11
            "sustainability_targets",
            "modal_shift_opportunities",
            "supplier_esg_scores",
            # Level 12
            "rma_authorizations",
            "returns",
            # Level 13
            "return_lines",
            "disposition_logs",
            # Level 14
            "kpi_actuals",
            "osa_metrics",
            "risk_events",
            "audit_logs",
        ]
        for table in tables:
            self.data[table] = []


class BaseLevelGenerator(ABC):
    """
    Abstract base class for level-specific generators.

    Each level generator implements the generate() method which reads from
    and writes to the shared GeneratorContext. Generators are stateless -
    all state lives in the context.

    Subclasses should:
    1. Call build_cache() if they need lookup indices
    2. Read required data from ctx.data
    3. Generate new data and append to ctx.data tables
    4. Update ID tracking dicts (e.g., ctx.supplier_ids[code] = id)

    Example:
        class Level3Generator(BaseLevelGenerator):
            def generate(self) -> None:
                # Build cache if needed
                self.ctx.build_cache(3)

                # Generate DCs
                for i, dc in enumerate(DCS):
                    dc_id = i + 1
                    self.ctx.data["distribution_centers"].append({
                        "id": dc_id,
                        "code": dc["code"],
                        ...
                    })
                    self.ctx.dc_ids[dc["code"]] = dc_id
    """

    def __init__(self, ctx: GeneratorContext) -> None:
        """
        Initialize generator with shared context.

        Args:
            ctx: Shared GeneratorContext instance
        """
        self.ctx = ctx

    @abstractmethod
    def generate(self) -> None:
        """
        Generate data for this level.

        Implementations should:
        - Read dependencies from self.ctx.data
        - Generate new rows and append to appropriate tables
        - Update ID tracking dicts for referential integrity
        - Mark level as complete: self.ctx.generated_levels.add(LEVEL)
        """
        pass

    @property
    def rng(self) -> np.random.Generator:
        """Convenience accessor for NumPy random generator."""
        return self.ctx.rng

    @property
    def fake(self) -> Faker:
        """Convenience accessor for Faker instance."""
        return self.ctx.fake

    @property
    def data(self) -> dict[str, list[dict]]:
        """Convenience accessor for shared data storage."""
        return self.ctx.data
