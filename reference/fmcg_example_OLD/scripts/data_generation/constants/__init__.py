"""
Constants Package - Master data and configuration for FMCG data generation.

This package centralizes all hardcoded constants used by generate_data.py,
removing ~1,400 lines of visual noise from the main file.

Modules:
- reference: Core business entities (divisions, channels, products, plants, packaging)
- network: Logistics infrastructure (DCs, ports, carriers)
- materials: Raw materials (50 ingredients with CAS numbers)
- rules: Operational parameters (KPI thresholds, business rules)

Usage:
    from data_generation.constants import (
        DIVISIONS, CHANNELS, PRODUCTS, PLANTS, PACKAGING_TYPES,
        DCS, PORTS, CARRIERS,
        INGREDIENTS,
        KPI_THRESHOLDS, BUSINESS_RULES,
    )
"""

from .materials import INGREDIENTS
from .network import CARRIERS, DCS, PORTS
from .reference import CHANNELS, DIVISIONS, PACKAGING_TYPES, PLANTS, PRODUCTS
from .rules import BUSINESS_RULES, KPI_THRESHOLDS

__all__ = [
    # Reference data
    "DIVISIONS",
    "CHANNELS",
    "PRODUCTS",
    "PLANTS",
    "PACKAGING_TYPES",
    # Network data
    "DCS",
    "PORTS",
    "CARRIERS",
    # Materials
    "INGREDIENTS",
    # Rules
    "KPI_THRESHOLDS",
    "BUSINESS_RULES",
]
