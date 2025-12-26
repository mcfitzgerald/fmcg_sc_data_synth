"""
Data Generation Module - Performance-optimized utilities for FMCG data generation.

This module provides foundational classes for the O(N) refactor of generate_data.py.
See robust-frolicking-coral.md for the full performance architecture plan.

Phase 1 - Foundation Classes:
- StaticDataPool: Pre-generated Faker data for O(1) vectorized sampling
- LookupIndex: Generic O(1) lookup index for grouped data
- LookupBuilder: Factory for building lookup indices from row dicts

Phase 2 - Streaming Validation Architecture:
- StreamingWriter: Memory-efficient streaming output with FK-aware management
- DependencyTracker: FK dependency graph for safe table memory purging
- RealismMonitor: Online streaming validation with O(1) space algorithms
- StochasticMode: Enum for normal vs disrupted distribution modes
- WelfordAccumulator: Online mean/variance computation
- FrequencySketch: Approximate frequency counting
- CardinalityEstimator: Unique element counting
- DegreeMonitor: Network hub concentration tracking

Phase 3 - Bottleneck Elimination:
- LookupCache: Pre-built indices for all FK patterns in generate_data.py
- PooledFaker: Batch Faker sampling wrapper for efficient string generation

Phase 4 - Vectorized Generation:
- POSSalesGenerator: Vectorized POS sales generation (~500K rows)
- OrderLinesGenerator: Vectorized order lines generation (~600K rows)
- ShipmentLegsGenerator: Vectorized shipment legs with delay distributions
- Structured array dtypes and conversion utilities

Phase 5 - Chaos Injection:
- RiskEventManager: Risk event triggering and override configuration
- QuirksManager: Behavioral pathology injection (bullwhip, phantom inventory, etc.)

Phase 6 - Multi-Promo Calendar:
- PromoCalendar: Pre-computed (week, account, sku) -> PromoEffect index
- PromoEffect: Promo effect data for lift/hangover application

Phase 7 - Modular Generators (see effervescent-shimmying-blum.md):
- GeneratorContext: Shared state dataclass passed to all generators
- BaseLevelGenerator: Abstract base class for level-specific generators

Phase D - Level Generators 1-13 (see effervescent-shimmying-blum.md):
- Level1Generator: Suppliers, plants, production_lines
- Level2Generator: Supplier_ingredients, formulas
- Level3Generator: Retail_accounts, retail_locations, DCs
- Level4Generator: SKUs, promotions
- Level5Generator: POs, work_orders, goods_receipts
- Level6Generator: PO_lines, batches
- Level7Generator: Batch_ingredients, inventory
- Level8Generator: POS_sales, orders, forecasts
- Level9Generator: Order_lines, allocations, pick_waves
- Level10Generator: Shipments, shipment_legs
- Level11Generator: Shipment_lines
- Level12Generator: Returns, RMAs
- Level13Generator: Disposition_logs

Phase B - Constants Package (see effervescent-shimmying-blum.md):
- constants/reference: DIVISIONS, CHANNELS, PRODUCTS, PLANTS, PACKAGING_TYPES
- constants/network: DCS, PORTS, CARRIERS
- constants/materials: INGREDIENTS
- constants/rules: KPI_THRESHOLDS, BUSINESS_RULES
- helpers: create_named_entities, TARGET_ROW_COUNTS
"""

from pathlib import Path

from .bottleneck_fixes import LookupCache, PooledFaker
from .promo_calendar import PromoCalendar, PromoEffect, date_to_week
from .quirks import Quirk, QuirksManager
from .risk_events import RiskEvent, RiskEventManager
from .lookup_builder import LookupBuilder, LookupIndex
from .realism_monitor import (
    CardinalityEstimator,
    DegreeMonitor,
    FrequencySketch,
    RealismMonitor,
    RealismViolationError,
    StochasticMode,
    WelfordAccumulator,
)
from .static_pool import StaticDataPool
from .streaming_writer import (
    DependencyTracker,
    StreamingWriter,
    copy_bool,
    copy_date,
    copy_num,
    copy_str,
    copy_timestamp,
    format_copy_value,
)
from .vectorized import (
    ORDER_LINES_COLUMNS,
    ORDER_LINES_DTYPE,
    POS_SALES_COLUMNS,
    POS_SALES_DTYPE,
    SHIPMENT_LEGS_COLUMNS,
    SHIPMENT_LEGS_DTYPE,
    OrderLinesGenerator,
    POSSalesGenerator,
    ShipmentLegsGenerator,
    VectorizedGenerator,
    apply_promo_effects,
    lumpy_demand,
    structured_to_copy_lines,
    structured_to_dicts,
    zipf_weights,
)
from .generators import (
    BaseLevelGenerator,
    GeneratorContext,
    Level0Generator,
    Level1Generator,
    Level2Generator,
    Level3Generator,
    Level4Generator,
    Level5Generator,
    Level6Generator,
    Level7Generator,
    Level8Generator,
    Level9Generator,
    Level10Generator,
    Level11Generator,
    Level12Generator,
    Level13Generator,
    Level14Generator,
)
from .validation import DataValidator
from .helpers import barabasi_albert_attachment
from .constants import (
    BUSINESS_RULES,
    CARRIERS,
    CHANNELS,
    DCS,
    DIVISIONS,
    INGREDIENTS,
    KPI_THRESHOLDS,
    PACKAGING_TYPES,
    PLANTS,
    PORTS,
    PRODUCTS,
)
from .helpers import TARGET_ROW_COUNTS, create_named_entities

# Path to default benchmark manifest
BENCHMARK_MANIFEST_PATH = Path(__file__).parent / "benchmark_manifest.json"

__all__ = [
    # Phase 1: Foundation
    "StaticDataPool",
    "LookupIndex",
    "LookupBuilder",
    # Phase 2: Streaming Validation
    "StreamingWriter",
    "DependencyTracker",
    "RealismMonitor",
    "RealismViolationError",
    "StochasticMode",
    # Phase 2: Online algorithms
    "WelfordAccumulator",
    "FrequencySketch",
    "CardinalityEstimator",
    "DegreeMonitor",
    # Phase 2: COPY format helpers
    "copy_str",
    "copy_num",
    "copy_bool",
    "copy_date",
    "copy_timestamp",
    "format_copy_value",
    # Phase 3: Bottleneck Elimination
    "LookupCache",
    "PooledFaker",
    # Phase 4: Vectorized Generation
    "VectorizedGenerator",
    "POSSalesGenerator",
    "OrderLinesGenerator",
    "ShipmentLegsGenerator",
    # Phase 4: Dtypes and columns
    "POS_SALES_DTYPE",
    "POS_SALES_COLUMNS",
    "ORDER_LINES_DTYPE",
    "ORDER_LINES_COLUMNS",
    "SHIPMENT_LEGS_DTYPE",
    "SHIPMENT_LEGS_COLUMNS",
    # Phase 4: Utilities
    "zipf_weights",
    "lumpy_demand",
    "apply_promo_effects",
    "structured_to_dicts",
    "structured_to_copy_lines",
    # Phase 5: Chaos Injection
    "RiskEventManager",
    "RiskEvent",
    "QuirksManager",
    "Quirk",
    # Phase 6: Multi-Promo Calendar
    "PromoCalendar",
    "PromoEffect",
    "date_to_week",
    # Phase 7: Modular Generators
    "GeneratorContext",
    "BaseLevelGenerator",
    "Level0Generator",
    "Level14Generator",
    # Phase D: Level Generators 1-13
    "Level1Generator",
    "Level2Generator",
    "Level3Generator",
    "Level4Generator",
    "Level5Generator",
    "Level6Generator",
    "Level7Generator",
    "Level8Generator",
    "Level9Generator",
    "Level10Generator",
    "Level11Generator",
    "Level12Generator",
    "Level13Generator",
    # Phase C: Validation
    "DataValidator",
    # Phase B: Constants
    "DIVISIONS",
    "CHANNELS",
    "PRODUCTS",
    "PLANTS",
    "PACKAGING_TYPES",
    "DCS",
    "PORTS",
    "CARRIERS",
    "INGREDIENTS",
    "KPI_THRESHOLDS",
    "BUSINESS_RULES",
    # Phase B: Helpers
    "barabasi_albert_attachment",
    "create_named_entities",
    "TARGET_ROW_COUNTS",
    # Paths
    "BENCHMARK_MANIFEST_PATH",
]
