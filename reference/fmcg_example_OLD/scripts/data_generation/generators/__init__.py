"""
Generators Package - Modular level generators for FMCG data generation.

This package contains the base classes and level-specific generators
for the refactored generate_data.py (see effervescent-shimmying-blum.md).

Base Classes:
- GeneratorContext: Shared state dataclass passed to all generators
- BaseLevelGenerator: Abstract base class for level-specific generators

Level Generators:
- Level0Generator: Reference data (divisions, channels, products, etc.)
- Level1Generator: Master data (suppliers, plants, production_lines)
- Level2Generator: Relationships (supplier_ingredients, formulas)
- Level3Generator: Network (retail_accounts, retail_locations, DCs)
- Level4Generator: Product (SKUs, promotions)
- Level5Generator: Procurement (POs, work_orders, goods_receipts)
- Level6Generator: Line items (PO lines, batches)
- Level7Generator: Batch consumption (batch_ingredients, inventory)
- Level8Generator: Demand (pos_sales, orders, forecasts)
- Level9Generator: Order lines and planning
- Level10Generator: Shipments and legs
- Level11Generator: Shipment lines
- Level12Generator: Returns (RMAs, returns, return_lines)
- Level13Generator: Disposition (disposition_logs)
- Level14Generator: Monitoring (KPIs)
"""

from .base import BaseLevelGenerator, GeneratorContext
from .level_0_reference import Level0Generator
from .level_1_2_source import Level1Generator, Level2Generator
from .level_3_network import Level3Generator
from .level_4_product import Level4Generator
from .level_5_7_manufacturing import Level5Generator, Level6Generator, Level7Generator
from .level_8_9_demand import Level8Generator, Level9Generator
from .level_10_11_fulfillment import Level10Generator, Level11Generator
from .level_12_13_returns import Level12Generator, Level13Generator
from .level_14_monitoring import Level14Generator

__all__ = [
    # Base classes
    "GeneratorContext",
    "BaseLevelGenerator",
    # Level 0: Reference data
    "Level0Generator",
    # Level 1-2: Source data
    "Level1Generator",
    "Level2Generator",
    # Level 3: Network
    "Level3Generator",
    # Level 4: Product
    "Level4Generator",
    # Level 5-7: Manufacturing
    "Level5Generator",
    "Level6Generator",
    "Level7Generator",
    # Level 8-9: Demand
    "Level8Generator",
    "Level9Generator",
    # Level 10-11: Fulfillment
    "Level10Generator",
    "Level11Generator",
    # Level 12-13: Returns
    "Level12Generator",
    "Level13Generator",
    # Level 14: Monitoring
    "Level14Generator",
]
