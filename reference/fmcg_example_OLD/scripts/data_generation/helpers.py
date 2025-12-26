"""
Helper functions and configuration for FMCG data generation.

Contains:
- barabasi_albert_attachment(): Preferential attachment for realistic network topology
- create_named_entities(): Deterministic testing entities (recalls, hot nodes, SPOFs)
- TARGET_ROW_COUNTS: Row count targets by generation level
"""

from typing import Any

import numpy as np


def barabasi_albert_attachment(
    existing_degrees: list[int], m: int = 1, rng: np.random.Generator | None = None
) -> list[int]:
    """
    Barabasi-Albert preferential attachment.

    Section 4.1: "Rich get richer" - big DCs get more stores.

    Args:
        existing_degrees: Current degree count for each node
        m: Number of connections to make
        rng: numpy random generator (uses global if None)

    Returns:
        Indices of nodes to connect to
    """
    if rng is None:
        rng = np.random.default_rng()

    # Add 1 to avoid division by zero for new nodes
    weights = np.array([d + 1 for d in existing_degrees], dtype=float)
    probs = weights / weights.sum()

    # Use numpy for efficient weighted selection
    m = min(m, len(weights))  # Can't select more than available
    selected = rng.choice(len(weights), size=m, replace=False, p=probs)
    return selected.tolist()


def create_named_entities() -> dict[str, Any]:
    """
    Create deterministic named entities for testing.

    These entities enable repeatable testing of specific scenarios:
    - Recall tracing (contaminated batch)
    - Hub stress testing (MegaMart hot node)
    - Single Point of Failure detection (Palm Oil supplier)
    - Centrality analysis (Chicago DC bottleneck)
    - Bullwhip effect (Black Friday promotion)
    - Seasonal routing (Shanghai-LA lane)
    - Ingredient risk scenarios

    Returns:
        Dict with entity codes mapped to their special properties.
    """
    return {
        # Contaminated batch for recall trace testing
        "B-2024-RECALL-001": {
            "type": "batch",
            "ingredient": "ING-SORB-001",
            "affected_stores": 500,
            "status": "QUALITY_HOLD",
        },
        # MegaMart hot node (hub stress test)
        "ACCT-MEGA-001": {
            "type": "retail_account",
            "store_count": 4500,
            "order_share": 0.25,  # 25% of all orders
            "channel": "B&M Large",
        },
        # Single-source Palm Oil (SPOF detection)
        "SUP-PALM-MY-001": {
            "type": "supplier",
            "ingredient": "ING-PALM-001",
            "is_sole_source": True,
            "country": "Malaysia",
        },
        # Bottleneck DC Chicago (centrality testing)
        "DC-NAM-CHI-001": {
            "type": "distribution_center",
            "stores_served": 2000,
            "volume_share": 0.40,  # 40% of NAM volume
            "division": "NAM",
        },
        # Black Friday 2024 (bullwhip effect)
        "PROMO-BF-2024": {
            "type": "promotion",
            "lift_multiplier": 3.0,
            "hangover_multiplier": 0.6,
            "week": 47,
        },
        # Seasonal ocean lane (temporal routing)
        "LANE-SH-LA-001": {
            "type": "route_segment",
            "origin": "Shanghai",
            "destination": "Los Angeles",
            "seasonal_capacity": {
                1: 0.50,
                2: 0.50,  # Jan-Feb: 50% capacity
                12: 0.75,  # Dec: 75% capacity
            },
        },
        # Problem ingredients
        "ING-PALM-001": {
            "type": "ingredient",
            "name": "Palm Oil",
            "lead_time_range": (60, 120),  # 2-4 months
            "on_time_rate": 0.50,
        },
        "ING-SORB-001": {
            "type": "ingredient",
            "name": "Sorbitol",
            "supplier_count": 1,  # Single qualified supplier
            "product_coverage": 1.0,  # All toothpaste
        },
        "ING-PEPP-001": {
            "type": "ingredient",
            "name": "Peppermint Oil",
            "available_quarters": [2, 3],  # Q2-Q3 only
            "off_season_price_multiplier": 3.0,
        },
    }


# Target row counts by level (from jolly-sauteeing-fern.md)
TARGET_ROW_COUNTS = {
    # Realistic B2B CPG model (Colgate->Retailer pattern)
    # Total: ~11.6M rows - validated against industry benchmarks
    0: 1_200,  # divisions, channels, products, packaging_types, ports, carriers, etc.
    1: 500,  # suppliers, plants, production_lines, carrier_contracts, route_segments
    2: 6_000,  # supplier_ingredients, certifications, formulas, formula_ingredients, etc.
    3: 10_000,  # retail_accounts, retail_locations, distribution_centers
    4: 20_000,  # skus, sku_costs, sku_substitutes, promotions, etc.
    5: 80_000,  # purchase_orders, goods_receipts, work_orders, etc.
    6: 550_000,  # purchase_order_lines, goods_receipt_lines, batches, etc.
    7: 700_000,  # batch_ingredients, inventory
    8: 2_500_000,  # pos_sales, demand_forecasts, orders, etc. (LARGEST)
    9: 7_000_000,  # order_lines (~3.2M) + allocations (~3.7M) - realistic B2B: avg 16 lines/order
    10: 1_200_000,  # pick_wave_orders, shipments, shipment_legs
    11: 1_000_000,  # shipment_lines
    12: 50_000,  # rma_authorizations, returns, return_lines
    13: 30_000,  # disposition_logs
    14: 570_000,  # kpi_actuals, osa_metrics, risk_events, audit_log
}
