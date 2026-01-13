#!/usr/bin/env python3
"""
Calibration script for Prism Sim configuration parameters.

Derives optimal simulation parameters from world definition and static world data
using supply chain physics and industry benchmarks (FMCG).

v0.31.0: Major rewrite to use target inventory turns as primary driver,
matching real FMCG company performance (Colgate 4.1x, P&G 5.5x, Unilever 6.2x).

Usage:
    poetry run python scripts/calibrate_config.py              # Analyze and recommend
    poetry run python scripts/calibrate_config.py --apply      # Apply recommendations
    poetry run python scripts/calibrate_config.py --target-turns 6.0  # Industry benchmark
    poetry run python scripts/calibrate_config.py --target-turns 5.0  # Conservative (Colgate-like)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save a JSON file with nice formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def count_nodes_by_type(locations_path: Path) -> dict[str, int]:
    """Count nodes by type from locations.csv."""
    counts: dict[str, int] = {}
    with open(locations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_type = row["type"].split(".")[-1]  # NodeType.STORE -> STORE
            counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def get_recipe_run_rates(recipes_path: Path) -> dict[str, float]:
    """Get run rates per product from recipes.csv."""
    rates: dict[str, float] = {}
    with open(recipes_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_id = row["product_id"]
            run_rate = float(row["run_rate_cases_per_hour"])
            rates[product_id] = run_rate
    return rates


def get_products_by_category(products_path: Path) -> dict[str, list[str]]:
    """Group products by category."""
    categories: dict[str, list[str]] = {}
    with open(products_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"].split(".")[-1]  # ProductCategory.ORAL_CARE -> ORAL_CARE
            product_id = row["id"]
            if cat not in categories:
                categories[cat] = []
            # Only include SKUs, not ingredients
            if product_id.startswith("SKU-"):
                categories[cat].append(product_id)
    return categories


def calculate_daily_demand(
    world_config: dict[str, Any],
    sim_config: dict[str, Any],
    node_counts: dict[str, int],
    products_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Calculate expected daily demand from configuration.

    Demand = sum over stores of (base_demand × format_scale × sku_count)
    """
    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {})
    category_profiles = demand_config.get("category_profiles", {})
    format_scales = demand_config.get("format_scale_factors", {})

    # Count stores (all STORE type nodes)
    n_stores = node_counts.get("STORE", 0)

    # Count SKUs per category
    sku_counts = {cat: len(skus) for cat, skus in products_by_category.items()}
    total_skus = sum(sku_counts.values())

    # Calculate demand per category
    category_demand = {}
    total_daily_demand = 0.0

    for cat, profile in category_profiles.items():
        if cat == "INGREDIENT":
            continue  # Ingredients don't have consumer demand

        base_demand = profile.get("base_daily_demand", 0.0)
        n_skus = sku_counts.get(cat, 0)

        # Demand = base_demand_per_store_per_sku × n_stores × n_skus
        # Using average format scale (most stores are small retailers)
        avg_format_scale = format_scales.get("SUPERMARKET", 1.0)

        cat_daily_demand = base_demand * avg_format_scale * n_stores * n_skus
        category_demand[cat] = cat_daily_demand
        total_daily_demand += cat_daily_demand

    return {
        "n_stores": n_stores,
        "total_skus": total_skus,
        "sku_counts": sku_counts,
        "category_demand": category_demand,
        "total_daily_demand": total_daily_demand,
    }


def calculate_plant_capacity(
    sim_config: dict[str, Any],
    recipe_rates: dict[str, float],
    products_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Calculate theoretical plant capacity from configuration.

    Each plant has ONE production line that can produce one SKU at a time.
    The production_rate_multiplier scales this to simulate multiple lines.

    Capacity per plant = effective_hours × avg_run_rate (for ONE line)
    Total capacity = sum of plant capacities × multiplier
    """
    sim_params = sim_config.get("simulation_parameters", {})
    mfg_config = sim_params.get("manufacturing", {})

    hours_per_day = mfg_config.get("production_hours_per_day", 24.0)
    efficiency = mfg_config.get("efficiency_factor", 0.95)
    downtime = mfg_config.get("unplanned_downtime_pct", 0.05)
    current_multiplier = mfg_config.get("production_rate_multiplier", 1.0)

    plant_params = mfg_config.get("plant_parameters", {})

    # Calculate capacity per plant
    plant_capacities = {}
    total_theoretical_capacity = 0.0

    for plant_id, params in plant_params.items():
        supported_cats = params.get("supported_categories", [])
        plant_efficiency = params.get("efficiency_factor", efficiency)
        plant_downtime = params.get("unplanned_downtime_pct", downtime)

        # Get AVERAGE run rate for supported SKUs (not sum!)
        # A plant can only produce ONE SKU at a time, so we use avg rate per line
        run_rates = []
        for cat in supported_cats:
            for product_id in products_by_category.get(cat, []):
                rate = recipe_rates.get(product_id, 0.0)
                if rate > 0:
                    run_rates.append(rate)

        avg_run_rate = sum(run_rates) / len(run_rates) if run_rates else 0.0

        # Theoretical capacity = hours × avg_run_rate × efficiency × (1 - downtime)
        # This is capacity for ONE production line at this plant
        effective_hours = hours_per_day * plant_efficiency * (1 - plant_downtime)
        plant_capacity = effective_hours * avg_run_rate

        plant_capacities[plant_id] = {
            "supported_categories": supported_cats,
            "avg_run_rate_per_hour": avg_run_rate,
            "n_supported_skus": len(run_rates),
            "effective_hours": effective_hours,
            "theoretical_capacity_per_line": plant_capacity,
        }
        total_theoretical_capacity += plant_capacity

    return {
        "hours_per_day": hours_per_day,
        "current_multiplier": current_multiplier,
        "plant_capacities": plant_capacities,
        "total_theoretical_capacity": total_theoretical_capacity,  # For 1 line per plant
        "total_with_multiplier": total_theoretical_capacity * current_multiplier,
    }


def derive_optimal_parameters(
    demand_analysis: dict[str, Any],
    capacity_analysis: dict[str, Any],
    sim_config: dict[str, Any],
    target_turns: float | None = None,
    target_oee: float = 0.65,
    target_service_level: float = 0.97,
) -> dict[str, Any]:
    """
    Derive optimal configuration parameters based on physics and industry benchmarks.

    v0.31.0: Complete rewrite to use target inventory turns as primary driver.

    Key insight: Real FMCG companies (Colgate 4.1x, P&G 5.5x, Unilever 6.2x) run
    with 52-89 days of network inventory to achieve 97%+ service. The previous
    OEE-driven approach resulted in too-lean inventory (17x turns) causing
    service level degradation.

    Physics relationships:
    - Network DOS = 365 / Target Turns
    - Network DOS = Store DOS + DC DOS + RDC DOS + Pipeline
    - Each echelon DOS = Lead Time + Safety Stock + Cycle Stock + Strategic Buffer
    - Trigger DOS = Replenishment Time + Safety Buffer + Review Period
    - SLOB Threshold = Expected DOS × ABC Velocity × Margin

    All physics parameters are read from config.calibration section.
    """
    sim_params = sim_config.get("simulation_parameters", {})

    # =================================================================
    # LOAD CALIBRATION PARAMETERS FROM CONFIG (v0.31.0 - No hardcodes)
    # =================================================================
    calibration = sim_params.get("calibration", {})
    industry = calibration.get("industry_benchmarks", {})
    z_scores = calibration.get("service_level_z_scores", {"A": 2.33, "B": 2.0, "C": 1.65})
    demand_cv = calibration.get("demand_variability_cv", {"store": 0.5, "customer_dc": 0.3, "rdc": 0.2})
    supply_cv = calibration.get("supply_variability_cv", {"store": 0.3, "customer_dc": 0.2, "rdc": 0.15})
    buffers = calibration.get("strategic_buffers", {})
    trigger_cfg = calibration.get("trigger_components", {})
    abc_prod = calibration.get("abc_production_factors", {})

    # Use config target turns if not specified via CLI
    if target_turns is None:
        target_turns = industry.get("target_turns", 6.0)

    # Extract other config parameters
    mfg_config = sim_params.get("manufacturing", {})
    mrp_thresholds = mfg_config.get("mrp_thresholds", {})
    campaign_config = mrp_thresholds.get("campaign_batching", {})
    replen_config = sim_params.get("agents", {}).get("replenishment", {})
    log_config = sim_params.get("logistics", {})
    validation_config = sim_params.get("validation", {})

    # Core timing parameters from config
    lead_time_days = log_config.get("default_lead_time_days", 3.0)
    order_cycle_days = replen_config.get("order_cycle_days", 5)
    current_production_horizon = campaign_config.get("production_horizon_days", 4)

    # =================================================================
    # STEP 1: DERIVE NETWORK DOS FROM TARGET TURNS (Primary Driver)
    # =================================================================
    # Network DOS = 365 / Target Turns
    # This is the TOTAL inventory across all echelons
    target_network_dos = 365.0 / target_turns

    # =================================================================
    # STEP 2: DECOMPOSE NETWORK DOS INTO ECHELON TARGETS
    # =================================================================
    # Network DOS = Store + Customer DC + RDC + Plant FG + Pipeline
    #
    # Industry-accurate allocation (based on FMCG practice):
    # - Stores: ~23% (retail shelf buffer)
    # - Customer DCs: ~23% (forward positioning)
    # - RDCs: ~35% (central buffer, flexibility)
    # - Plant FG: ~12% (production smoothing)
    # - Pipeline: ~7% (in-transit)
    #
    # Proportions are config-driven for calibration flexibility.
    # =================================================================

    # Load echelon proportions from config (v0.31.0 - no hardcodes)
    echelon_props = calibration.get("echelon_proportions", {})
    store_proportion = echelon_props.get("store", 0.23)
    dc_proportion = echelon_props.get("customer_dc", 0.23)
    rdc_proportion = echelon_props.get("rdc", 0.35)
    plant_proportion = echelon_props.get("plant", 0.12)
    pipeline_proportion = echelon_props.get("pipeline", 0.07)

    # Base DOS by echelon (before ABC differentiation)
    store_dos_base = target_network_dos * store_proportion
    dc_dos_base = target_network_dos * dc_proportion
    rdc_dos_base = target_network_dos * rdc_proportion
    plant_dos = target_network_dos * plant_proportion
    pipeline_dos = target_network_dos * pipeline_proportion

    # =================================================================
    # STEP 3: PHYSICS-BASED VALIDATION OF ECHELON DOS
    # =================================================================
    # Ensure each echelon has minimum DOS to cover:
    # 1. Lead time (replenishment runway)
    # 2. Safety stock (variability buffer)
    # 3. Cycle stock (order cycle average)
    # 4. Strategic buffer (presentation stock, forward positioning)
    # =================================================================

    # Strategic buffers from config
    presentation_stock = buffers.get("presentation_stock_days", 3.0)
    forward_positioning = buffers.get("forward_positioning_days", 5.0)
    production_smoothing = buffers.get("production_smoothing_days", 5.0)

    # Store minimum: Lead time + Safety + Cycle + Presentation
    # Safety = z × √(demand_var² × LT + supply_var² × demand²)
    # Simplified: z × combined_cv × √LT
    combined_cv_store = (demand_cv.get("store", 0.5)**2 + supply_cv.get("store", 0.3)**2)**0.5
    store_safety_b = z_scores.get("B", 2.0) * combined_cv_store * (lead_time_days ** 0.5)
    store_cycle = order_cycle_days / 2
    store_min = lead_time_days + store_safety_b + store_cycle + presentation_stock

    # DC minimum: Lead time + Safety + Cycle + Forward positioning
    combined_cv_dc = (demand_cv.get("customer_dc", 0.3)**2 + supply_cv.get("customer_dc", 0.2)**2)**0.5
    dc_safety_b = z_scores.get("B", 2.0) * combined_cv_dc * (lead_time_days ** 0.5)
    dc_cycle = order_cycle_days / 2
    dc_min = lead_time_days + dc_safety_b + dc_cycle + forward_positioning

    # RDC minimum: Lead time + Safety + Production cycle + Smoothing buffer
    combined_cv_rdc = (demand_cv.get("rdc", 0.2)**2 + supply_cv.get("rdc", 0.15)**2)**0.5
    rdc_safety_b = z_scores.get("B", 2.0) * combined_cv_rdc * (lead_time_days ** 0.5)
    # Production horizon determines RDC cycle stock
    production_horizon = max(current_production_horizon, 7)  # Recommend at least 7 days
    rdc_cycle = production_horizon / 2
    rdc_min = lead_time_days + rdc_safety_b + rdc_cycle + production_smoothing

    # Apply minimums (ensure physics requirements are met)
    store_dos = max(store_dos_base, store_min)
    dc_dos = max(dc_dos_base, dc_min)
    rdc_dos = max(rdc_dos_base, rdc_min)

    # Recalculate actual network DOS after applying minimums
    actual_network_dos = store_dos + dc_dos + rdc_dos + plant_dos + pipeline_dos
    actual_turns = 365.0 / actual_network_dos

    # =================================================================
    # STEP 4: ABC-DIFFERENTIATED DOS (Service level stratification)
    # =================================================================
    # A-items: Higher service target (99%) → MORE buffer (premium service)
    # B-items: Standard service (97%) → base buffer
    # C-items: Lower service (95%) → SLIGHTLY less buffer (SLOB control)
    #
    # CRITICAL: A-items must have MORE inventory than B-items, not less!
    # The z-score affects safety stock calculation, but for priming factors
    # we want A > B > C to ensure service level hierarchy.
    # =================================================================
    store_safety_a = z_scores.get("A", 2.33) * combined_cv_store * (lead_time_days ** 0.5)
    store_safety_c = z_scores.get("C", 1.65) * combined_cv_store * (lead_time_days ** 0.5)

    store_dos_a = round(lead_time_days + store_safety_a + store_cycle + presentation_stock, 1)
    store_dos_b = round(store_dos, 1)
    store_dos_c = round(lead_time_days + store_safety_c + store_cycle + presentation_stock * 0.5, 1)

    # ABC velocity factors for priming (relative to base)
    # CORRECTED v0.31.1: Ensure A > B > C hierarchy for service level priority
    # A-items: 20% more buffer (premium service target)
    # B-items: baseline (standard service)
    # C-items: 15% less buffer (SLOB control, but not too aggressive)
    abc_priming_factors = {
        "A": 1.2,  # Premium items get more buffer for 99% service
        "B": 1.0,  # Standard items at baseline
        "C": 0.85,  # Slightly less for SLOB control, but not starving
    }

    # =================================================================
    # STEP 5: TRIGGER THRESHOLD DERIVATION (Network DOS triggers)
    # =================================================================
    # Trigger = when to start production for a SKU
    # Must cover: Replenishment Time + Safety + Review Period
    #
    # Replenishment Time = Production Lead Time + Transit Time
    # Safety = ABC-differentiated buffer for variability
    # Review Period = How often MRP checks (order cycle)
    # =================================================================
    prod_lead_time = trigger_cfg.get("production_lead_time_days", 3)
    transit_time = trigger_cfg.get("transit_time_days", 3)
    replenishment_time = prod_lead_time + transit_time

    safety_a = trigger_cfg.get("safety_buffer_a", 10)
    safety_b = trigger_cfg.get("safety_buffer_b", 6)
    safety_c = trigger_cfg.get("safety_buffer_c", 3)

    review_a = trigger_cfg.get("review_period_a", 5)
    review_b = trigger_cfg.get("review_period_b", 5)
    review_c = trigger_cfg.get("review_period_c", 3)

    trigger_a = int(replenishment_time + safety_a + review_a)
    trigger_b = int(replenishment_time + safety_b + review_b)
    trigger_c = int(replenishment_time + safety_c + review_c)

    # =================================================================
    # STEP 6: SLOB THRESHOLD CALIBRATION
    # =================================================================
    # SLOB = Slow-moving/Obsolete inventory
    # Threshold = Expected DOS × ABC Velocity Factor × Margin
    #
    # A-items turn faster (lower expected DOS)
    # C-items turn slower (higher expected DOS)
    # Margin = 1.5x means "50% above expected = slow"
    # =================================================================
    abc_velocity_factors = validation_config.get(
        "slob_abc_velocity_factors", {"A": 0.7, "B": 1.0, "C": 1.5}
    )
    slob_margin = validation_config.get("slob_margin", 1.5)

    slob_thresholds = {
        abc_class: round(actual_network_dos * velocity_factor * slob_margin, 0)
        for abc_class, velocity_factor in abc_velocity_factors.items()
    }

    current_slob_thresholds = validation_config.get("slob_abc_thresholds", {})

    # =================================================================
    # STEP 7: CAPACITY ANALYSIS (OEE emerges from inventory policy)
    # =================================================================
    total_demand = demand_analysis["total_daily_demand"]
    theoretical_capacity = capacity_analysis["total_theoretical_capacity"]
    current_multiplier = capacity_analysis["current_multiplier"]
    current_capacity = theoretical_capacity * current_multiplier

    # With higher inventory, OEE will be lower but more stable
    # This is the physics trade-off: Service vs OEE
    expected_oee = total_demand / current_capacity if current_capacity > 0 else 0

    # RDC multiplier for aggregated demand
    rdc_store_multiplier = demand_analysis["n_stores"] / 40

    # ABC production factors from config
    a_buffer = abc_prod.get("a_buffer", 1.1)
    c_production_factor = abc_prod.get("c_production_factor", 0.4)
    c_demand_factor = abc_prod.get("c_demand_factor", 0.7)

    return {
        "analysis": {
            "total_daily_demand": total_demand,
            "theoretical_capacity_base": theoretical_capacity,
            "current_multiplier": current_multiplier,
            "current_capacity": current_capacity,
            "capacity_utilization": total_demand / current_capacity if current_capacity > 0 else 0,
            "target_turns": target_turns,
            "target_network_dos": round(target_network_dos, 1),
            "actual_network_dos": round(actual_network_dos, 1),
            "actual_turns": round(actual_turns, 1),
        },
        "inventory_analysis": {
            "store_dos_a": store_dos_a,
            "store_dos_b": store_dos_b,
            "store_dos_c": store_dos_c,
            "store_dos": round(store_dos, 1),
            "dc_dos": round(dc_dos, 1),
            "rdc_dos": round(rdc_dos, 1),
            "plant_dos": round(plant_dos, 1),
            "pipeline_dos": round(pipeline_dos, 1),
            "lead_time_days": lead_time_days,
            "order_cycle_days": order_cycle_days,
            "production_horizon": production_horizon,
            "base_network_dos": round(actual_network_dos, 1),
            "echelon_breakdown": {
                "store_pct": round(store_dos / actual_network_dos * 100, 1),
                "dc_pct": round(dc_dos / actual_network_dos * 100, 1),
                "rdc_pct": round(rdc_dos / actual_network_dos * 100, 1),
                "plant_pct": round(plant_dos / actual_network_dos * 100, 1),
                "pipeline_pct": round(pipeline_dos / actual_network_dos * 100, 1),
            },
        },
        "trigger_analysis": {
            "trigger_a": trigger_a,
            "trigger_b": trigger_b,
            "trigger_c": trigger_c,
            "current_trigger_a": campaign_config.get("trigger_dos_a", 5),
            "current_trigger_b": campaign_config.get("trigger_dos_b", 4),
            "current_trigger_c": campaign_config.get("trigger_dos_c", 3),
            "replenishment_time": replenishment_time,
        },
        "slob_analysis": {
            "current_thresholds": current_slob_thresholds,
            "derived_thresholds": slob_thresholds,
            "slob_margin": slob_margin,
            "abc_velocity_factors": abc_velocity_factors,
        },
        "recommendations": {
            "production_rate_multiplier": round(current_multiplier, 1),  # Keep current
            "store_days_supply": round(store_dos, 1),
            "rdc_days_supply": round(rdc_dos, 1),
            "customer_dc_days_supply": round(dc_dos, 1),
            "abc_velocity_factors": abc_priming_factors,
            "trigger_dos_a": trigger_a,
            "trigger_dos_b": trigger_b,
            "trigger_dos_c": trigger_c,
            "production_horizon_days": production_horizon,
            "rdc_store_multiplier": round(rdc_store_multiplier, 1),
            "slob_abc_thresholds": slob_thresholds,
            "a_production_buffer": a_buffer,
            "c_production_factor": c_production_factor,
            "c_demand_factor": c_demand_factor,
        },
        "expected_metrics": {
            "expected_oee": round(expected_oee, 2),
            "expected_daily_production": total_demand,
            "expected_service_level": target_service_level,
            "expected_network_dos": round(actual_network_dos, 1),
            "expected_turns": round(actual_turns, 1),
        },
        "industry_comparison": {
            "target_turns": target_turns,
            "colgate": {"turns": 4.1, "dos": 89},
            "pg": {"turns": 5.5, "dos": 66},
            "unilever": {"turns": 6.2, "dos": 59},
        },
        # Seasonal capacity analysis added in v0.30.0
        "seasonal_capacity": {},
    }


def validate_config_consistency(
    sim_config: dict[str, Any],
    derived: dict[str, Any],
) -> list[str]:
    """
    Check that config values are internally consistent.
    Returns a list of violations/warnings.
    """
    violations = []
    sim_params = sim_config.get("simulation_parameters", {})
    inv_init = sim_params.get("inventory", {}).get("initialization", {})
    mfg_config = sim_params.get("manufacturing", {})
    campaign_config = mfg_config.get("mrp_thresholds", {}).get("campaign_batching", {})
    validation_config = sim_params.get("validation", {})

    # 1. Priming vs Triggers (Multi-echelon)
    # Network-wide priming DOS must exceed trigger to avoid Day 1 production spike
    # Priming is across: stores + customer DCs + RDCs
    store_dos = inv_init.get("store_days_supply", 4.5)
    rdc_dos = inv_init.get("rdc_days_supply", 7.5)
    dc_dos = inv_init.get("customer_dc_days_supply", 7.5)
    abc_factors = inv_init.get("abc_velocity_factors", {"A": 1.2, "B": 1.0, "C": 0.8})

    trigger_a = campaign_config.get("trigger_dos_a", 14)
    trigger_b = campaign_config.get("trigger_dos_b", 10)
    trigger_c = campaign_config.get("trigger_dos_c", 7)

    # Calculate NETWORK priming DOS per ABC class (sum of echelons)
    # Note: These are cumulative - product flows through all echelons
    network_priming_dos = {
        "A": (store_dos + dc_dos + rdc_dos) * abc_factors.get("A", 1.2),
        "B": (store_dos + dc_dos + rdc_dos) * abc_factors.get("B", 1.0),
        "C": (store_dos + dc_dos + rdc_dos) * abc_factors.get("C", 0.8),
    }
    triggers = {"A": trigger_a, "B": trigger_b, "C": trigger_c}

    for abc_class in ["A", "B", "C"]:
        priming = network_priming_dos[abc_class]
        trigger = triggers[abc_class]
        if priming < trigger:
            violations.append(
                f"{abc_class}-item network priming ({priming:.1f}d) < trigger "
                f"({trigger}d) -> Day 1 production spike likely"
            )

    # 2. SLOB threshold vs Network DOS
    # SLOB threshold should be > expected DOS (else everything is SLOB)
    base_network_dos = derived.get("inventory_analysis", {}).get("base_network_dos", 18)
    slob_thresholds = validation_config.get("slob_abc_thresholds", {})
    slob_abc_factors = validation_config.get(
        "slob_abc_velocity_factors", {"A": 0.7, "B": 1.0, "C": 1.5}
    )

    for abc_class in ["A", "B", "C"]:
        velocity = slob_abc_factors.get(abc_class, 1.0)
        expected_dos = base_network_dos * velocity
        slob_threshold = slob_thresholds.get(abc_class, 60)
        if slob_threshold < expected_dos * 1.2:
            violations.append(
                f"{abc_class}-item SLOB threshold ({slob_threshold:.0f}d) too close to "
                f"expected DOS ({expected_dos:.1f}d) -> high SLOB %"
            )

    # 3. Capacity vs Demand balance
    utilization = derived.get("analysis", {}).get("capacity_utilization", 0)
    if utilization > 0.95:
        violations.append(
            f"Capacity utilization {utilization:.0%} > 95% -> stockouts likely"
        )
    if utilization < 0.50:
        violations.append(
            f"Capacity utilization {utilization:.0%} < 50% -> low OEE expected"
        )

    # 4. Expected inventory turns cross-validation
    # Note: turns_range in config is for observed metrics, not expected
    # Expected turns = 365 / network_dos is purely physics-based
    if base_network_dos > 0:
        expected_turns = 365.0 / base_network_dos
        # Warn if expected turns are extreme (too high = understock, too low = overstock)
        if expected_turns > 30:
            violations.append(
                f"Expected turns {expected_turns:.1f}x is very high -> "
                f"insufficient inventory buffer"
            )
        if expected_turns < 5:
            violations.append(
                f"Expected turns {expected_turns:.1f}x is very low -> "
                f"excess inventory holding cost"
            )

    return violations


def validate_seasonal_balance(
    sim_config: dict[str, Any],
    derived: dict[str, Any],
) -> list[str]:
    """
    Validate capacity meets demand across all seasons.

    v0.30.0: Ensures seasonal capacity flex doesn't create structural
    shortfalls during peak or insufficient buffer during trough.
    """
    warnings: list[str] = []

    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {}).get("seasonality", {})
    validation_config = sim_params.get("validation", {})

    demand_amp = demand_config.get("amplitude", 0.12)
    capacity_amp = demand_config.get("capacity_amplitude", 0.0)

    if capacity_amp == 0:
        return warnings  # No seasonal flex, skip validation

    base_demand = derived.get("analysis", {}).get("total_daily_demand", 0)
    base_capacity = derived.get("analysis", {}).get("current_capacity", 0)

    if base_capacity == 0:
        return warnings  # Can't validate without capacity data

    # Get margin thresholds from config (with sensible defaults)
    min_peak_margin = validation_config.get("seasonal_min_peak_margin", 0.05)
    min_trough_margin = validation_config.get("seasonal_min_trough_margin", 0.10)
    max_peak_oee = validation_config.get("seasonal_max_peak_oee", 0.95)
    min_trough_oee = validation_config.get("seasonal_min_trough_oee", 0.40)

    # Peak validation: capacity must exceed demand with margin
    peak_demand = base_demand * (1 + demand_amp)
    peak_capacity = base_capacity * (1 + capacity_amp)
    peak_margin = (peak_capacity - peak_demand) / peak_demand if peak_demand > 0 else 0

    if peak_margin < min_peak_margin:
        warnings.append(
            f"Peak season: capacity margin only {peak_margin:.1%} "
            f"(demand {peak_demand:,.0f}, capacity {peak_capacity:,.0f}) "
            f"-> stockouts likely during peak"
        )

    # Trough validation: need buffer for variability
    trough_demand = base_demand * (1 - demand_amp)
    trough_capacity = base_capacity * (1 - capacity_amp)
    trough_margin = (
        (trough_capacity - trough_demand) / trough_demand if trough_demand > 0 else 0
    )

    if trough_margin < min_trough_margin:
        warnings.append(
            f"Trough season: capacity margin only {trough_margin:.1%} "
            f"(demand {trough_demand:,.0f}, capacity {trough_capacity:,.0f}) "
            f"-> insufficient buffer for demand variability"
        )

    # OEE range validation
    if peak_capacity > 0:
        peak_oee = peak_demand / peak_capacity
        if peak_oee > max_peak_oee:
            warnings.append(
                f"Peak OEE would be {peak_oee:.0%} (>{max_peak_oee:.0%}) "
                f"-> capacity-constrained"
            )

    if trough_capacity > 0:
        trough_oee = trough_demand / trough_capacity
        if trough_oee < min_trough_oee:
            warnings.append(
                f"Trough OEE would be {trough_oee:.0%} (<{min_trough_oee:.0%}) "
                f"-> very low utilization"
            )

    # Amplitude relationship check
    if capacity_amp > demand_amp:
        warnings.append(
            f"capacity_amplitude ({capacity_amp}) > demand amplitude ({demand_amp}) "
            f"-> unused capacity during peak, risk during trough"
        )

    return warnings


def derive_seasonal_capacity_params(
    sim_config: dict[str, Any],
    derived: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive optimal capacity_amplitude based on physics.

    Key insight: capacity_amplitude should be LESS than demand_amplitude
    to maintain safety buffer during troughs.

    Real FMCG practice:
    - Peak: Easy to add overtime, temp workers
    - Trough: Labor contracts limit reduction

    Returns recommendations for seasonal capacity parameters.
    """
    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {}).get("seasonality", {})
    validation_config = sim_params.get("validation", {})

    demand_amp = demand_config.get("amplitude", 0.12)
    current_capacity_amp = demand_config.get("capacity_amplitude", 0.0)

    # Get target buffer from config (default 10% margin at trough)
    target_trough_buffer = validation_config.get(
        "seasonal_target_trough_buffer", 0.10
    )

    # Derive optimal capacity amplitude
    # If demand drops by demand_amp, capacity should drop less
    # to maintain buffer: capacity_amp = demand_amp * (1 - buffer_fraction)
    # This ensures trough_capacity / trough_demand >= 1 + buffer
    optimal_symmetric = demand_amp * (1 - target_trough_buffer)

    # Get base metrics for seasonal analysis
    base_demand = derived.get("analysis", {}).get("total_daily_demand", 0)
    base_capacity = derived.get("analysis", {}).get("current_capacity", 0)

    # Calculate seasonal metrics for reporting
    seasonal_metrics = {
        "peak_demand": base_demand * (1 + demand_amp),
        "peak_capacity_current": base_capacity * (1 + current_capacity_amp),
        "peak_capacity_recommended": base_capacity * (1 + optimal_symmetric),
        "trough_demand": base_demand * (1 - demand_amp),
        "trough_capacity_current": base_capacity * (1 - current_capacity_amp),
        "trough_capacity_recommended": base_capacity * (1 - optimal_symmetric),
    }

    # Calculate margins
    if seasonal_metrics["trough_demand"] > 0:
        current_trough_margin = (
            seasonal_metrics["trough_capacity_current"]
            - seasonal_metrics["trough_demand"]
        ) / seasonal_metrics["trough_demand"]
        recommended_trough_margin = (
            seasonal_metrics["trough_capacity_recommended"]
            - seasonal_metrics["trough_demand"]
        ) / seasonal_metrics["trough_demand"]
    else:
        current_trough_margin = 0
        recommended_trough_margin = 0

    return {
        "demand_amplitude": demand_amp,
        "current_capacity_amplitude": current_capacity_amp,
        "recommended_capacity_amplitude": round(optimal_symmetric, 3),
        "target_trough_buffer": target_trough_buffer,
        "current_trough_margin": round(current_trough_margin, 3),
        "recommended_trough_margin": round(recommended_trough_margin, 3),
        "seasonal_metrics": seasonal_metrics,
    }


def print_report(
    demand_analysis: dict[str, Any],
    capacity_analysis: dict[str, Any],
    recommendations: dict[str, Any],
    violations: list[str] | None = None,
) -> None:
    """Print a nice calibration report."""
    print("\n" + "=" * 70)
    print("     PRISM SIM CONFIGURATION CALIBRATION REPORT (v0.31.0)")
    print("     Industry-Benchmark Calibration: Turns-Driven Approach")
    print("=" * 70)

    # Industry comparison header
    analysis = recommendations["analysis"]
    industry = recommendations.get("industry_comparison", {})
    print("\n--- INDUSTRY BENCHMARK COMPARISON ---")
    print(f"Target Turns: {analysis.get('target_turns', 6.0)}x → {analysis.get('target_network_dos', 60)}d DOS")
    print("Reference Companies:")
    for company in ["colgate", "pg", "unilever"]:
        if company in industry:
            info = industry[company]
            print(f"  {company.upper()}: {info['turns']}x turns ({info['dos']}d DOS)")

    print("\n--- DEMAND ANALYSIS ---")
    print(f"Total Stores: {demand_analysis['n_stores']:,}")
    print(f"Total SKUs: {demand_analysis['total_skus']:,}")
    print("SKUs by Category:")
    for cat, count in demand_analysis["sku_counts"].items():
        print(f"  {cat}: {count}")
    print("\nDaily Demand by Category:")
    for cat, demand in demand_analysis["category_demand"].items():
        print(f"  {cat}: {demand:,.0f} cases/day")
    print(f"\nTOTAL DAILY DEMAND: {demand_analysis['total_daily_demand']:,.0f} cases/day")

    print("\n--- CAPACITY ANALYSIS ---")
    print(f"Base Capacity (1 line/plant): {capacity_analysis['total_theoretical_capacity']:,.0f} cases/day")
    print(f"Current Multiplier: {capacity_analysis['current_multiplier']}x")
    print(f"Current Effective Capacity: {capacity_analysis['total_with_multiplier']:,.0f} cases/day")
    util = analysis["capacity_utilization"]
    print(f"Capacity Utilization: {util:.1%}")

    print("\n--- ECHELON DOS BREAKDOWN (v0.31.0 Physics-Based) ---")
    inv = recommendations["inventory_analysis"]
    breakdown = inv.get("echelon_breakdown", {})
    print(f"Target Network DOS: {analysis.get('target_network_dos', 60)} days")
    print(f"Actual Network DOS: {inv['base_network_dos']} days (after physics minimums)")
    print(f"Resulting Turns: {analysis.get('actual_turns', 6.0)}x")
    print("\nEchelon Allocation:")
    print(f"  Store:    {inv['store_dos']:5.1f}d ({breakdown.get('store_pct', 0):5.1f}%)")
    print(f"  Cust DC:  {inv['dc_dos']:5.1f}d ({breakdown.get('dc_pct', 0):5.1f}%)")
    print(f"  RDC:      {inv['rdc_dos']:5.1f}d ({breakdown.get('rdc_pct', 0):5.1f}%)")
    print(f"  Plant FG: {inv.get('plant_dos', 0):5.1f}d ({breakdown.get('plant_pct', 0):5.1f}%)")
    print(f"  Pipeline: {inv.get('pipeline_dos', 0):5.1f}d ({breakdown.get('pipeline_pct', 0):5.1f}%)")
    print(f"\nStore DOS by ABC: A={inv['store_dos_a']}d, B={inv['store_dos_b']}d, C={inv['store_dos_c']}d")

    print("\n--- TRIGGER THRESHOLD ANALYSIS ---")
    trig = recommendations["trigger_analysis"]
    print(f"Replenishment Time: {trig.get('replenishment_time', 6)} days (prod + transit)")
    print(f"\nDerived Triggers:  A={trig['trigger_a']}d, B={trig['trigger_b']}d, C={trig['trigger_c']}d")
    print(f"Current Triggers:  A={trig['current_trigger_a']}d, B={trig['current_trigger_b']}d, C={trig['current_trigger_c']}d")

    # Highlight significant changes
    changes = []
    for abc, key in [("A", "trigger_a"), ("B", "trigger_b"), ("C", "trigger_c")]:
        curr = trig[f"current_{key}"]
        new = trig[key]
        if new != curr:
            changes.append(f"  {abc}: {curr}d → {new}d ({'+' if new > curr else ''}{new - curr}d)")
    if changes:
        print("\n  CHANGES NEEDED:")
        for c in changes:
            print(c)

    print("\n--- SLOB THRESHOLD CALIBRATION ---")
    slob = recommendations["slob_analysis"]
    print(f"SLOB Margin: {slob['slob_margin']}x expected DOS")
    print(f"ABC Velocity Factors: {slob['abc_velocity_factors']}")
    print(f"\nCurrent Thresholds: A={slob['current_thresholds'].get('A', 0):.0f}d, "
          f"B={slob['current_thresholds'].get('B', 0):.0f}d, C={slob['current_thresholds'].get('C', 0):.0f}d")
    print(f"Derived Thresholds: A={slob['derived_thresholds'].get('A', 0):.0f}d, "
          f"B={slob['derived_thresholds'].get('B', 0):.0f}d, C={slob['derived_thresholds'].get('C', 0):.0f}d")

    # Seasonal capacity analysis
    seasonal = recommendations.get("seasonal_capacity", {})
    if seasonal:
        print("\n--- SEASONAL CAPACITY ANALYSIS ---")
        d_amp = seasonal.get("demand_amplitude", 0)
        c_amp_cur = seasonal.get("current_capacity_amplitude", 0)
        c_amp_rec = seasonal.get("recommended_capacity_amplitude", 0)
        print(f"Demand Amplitude: ±{d_amp:.0%}")
        print(f"Current Capacity Amplitude: ±{c_amp_cur:.0%}")
        print(f"Recommended Capacity Amplitude: ±{c_amp_rec:.1%}")

    print("\n--- RECOMMENDATIONS (Apply with --apply) ---")
    rec = recommendations["recommendations"]
    print("Inventory Priming:")
    print(f"  store_days_supply: {rec['store_days_supply']}")
    print(f"  customer_dc_days_supply: {rec['customer_dc_days_supply']}")
    print(f"  rdc_days_supply: {rec['rdc_days_supply']}")
    print(f"  abc_velocity_factors: {rec['abc_velocity_factors']}")
    print("\nMRP Triggers:")
    print(f"  trigger_dos_a: {rec['trigger_dos_a']}")
    print(f"  trigger_dos_b: {rec['trigger_dos_b']}")
    print(f"  trigger_dos_c: {rec['trigger_dos_c']}")
    print(f"  production_horizon_days: {rec['production_horizon_days']}")
    print("\nABC Production Factors:")
    print(f"  a_production_buffer: {rec.get('a_production_buffer', 1.1)}")
    print(f"  c_production_factor: {rec.get('c_production_factor', 0.4)}")
    print(f"  c_demand_factor: {rec.get('c_demand_factor', 0.7)}")
    print("\nSLOB Thresholds:")
    print(f"  slob_abc_thresholds: {rec['slob_abc_thresholds']}")

    print("\n--- EXPECTED METRICS (after calibration) ---")
    exp = recommendations["expected_metrics"]
    print(f"Expected Service Level: {exp['expected_service_level']:.0%}")
    print(f"Expected Inventory Turns: {exp['expected_turns']}x")
    print(f"Expected Network DOS: {exp['expected_network_dos']} days")
    print(f"Expected OEE: {exp['expected_oee']:.0%}")
    print(f"Expected Daily Production: {exp['expected_daily_production']:,.0f} cases")

    # Print config consistency violations/warnings
    if violations:
        print("\n--- CONFIG CONSISTENCY WARNINGS ---")
        for v in violations:
            print(f"  WARNING: {v}")
    else:
        print("\n--- CONFIG CONSISTENCY ---")
        print("  All config parameters are internally consistent.")

    print("\n" + "=" * 70)


def apply_recommendations(
    sim_config_path: Path,
    recommendations: dict[str, Any],
) -> None:
    """Apply recommended parameters to simulation_config.json."""
    config = load_json(sim_config_path)
    rec = recommendations["recommendations"]

    sim_params = config.setdefault("simulation_parameters", {})

    # Update manufacturing
    mfg = sim_params.setdefault("manufacturing", {})
    mfg["production_rate_multiplier"] = rec["production_rate_multiplier"]

    # Update MRP thresholds (v0.31.0 - comprehensive update)
    mrp = mfg.setdefault("mrp_thresholds", {})

    # Campaign batching with triggers and horizon
    campaign = mrp.setdefault("campaign_batching", {})
    campaign["trigger_dos_a"] = rec["trigger_dos_a"]
    campaign["trigger_dos_b"] = rec["trigger_dos_b"]
    campaign["trigger_dos_c"] = rec["trigger_dos_c"]
    campaign["production_horizon_days"] = rec["production_horizon_days"]

    # ABC production factors (v0.31.0 - SLOB control)
    mrp["a_production_buffer"] = rec.get("a_production_buffer", 1.1)
    mrp["c_production_factor"] = rec.get("c_production_factor", 0.4)
    mrp["c_demand_factor"] = rec.get("c_demand_factor", 0.7)

    # Update inventory initialization
    inv = sim_params.setdefault("inventory", {})
    init = inv.setdefault("initialization", {})
    init["store_days_supply"] = rec["store_days_supply"]
    init["rdc_days_supply"] = rec["rdc_days_supply"]
    init["customer_dc_days_supply"] = rec["customer_dc_days_supply"]
    init["abc_velocity_factors"] = rec["abc_velocity_factors"]
    init["rdc_store_multiplier"] = rec["rdc_store_multiplier"]

    # Update SLOB thresholds
    validation = sim_params.setdefault("validation", {})
    validation["slob_abc_thresholds"] = rec["slob_abc_thresholds"]

    # Update seasonal capacity amplitude
    seasonal = recommendations.get("seasonal_capacity", {})
    if seasonal and "recommended_capacity_amplitude" in seasonal:
        demand = sim_params.setdefault("demand", {})
        seasonality = demand.setdefault("seasonality", {})
        seasonality["capacity_amplitude"] = seasonal["recommended_capacity_amplitude"]

    save_json(sim_config_path, config)
    print(f"\n✓ Applied recommendations to {sim_config_path}")
    print("\nKey changes applied:")
    print(f"  - Priming DOS: store={rec['store_days_supply']}d, dc={rec['customer_dc_days_supply']}d, rdc={rec['rdc_days_supply']}d")
    print(f"  - Triggers: A={rec['trigger_dos_a']}d, B={rec['trigger_dos_b']}d, C={rec['trigger_dos_c']}d")
    print(f"  - Production horizon: {rec['production_horizon_days']}d")
    print(f"  - C-item factors: prod={rec.get('c_production_factor', 0.4)}, demand={rec.get('c_demand_factor', 0.7)}")
    print(f"  - SLOB thresholds: {rec['slob_abc_thresholds']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Prism Sim configuration parameters using industry benchmarks"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply recommendations to simulation_config.json",
    )
    parser.add_argument(
        "--target-turns",
        type=float,
        default=None,
        help="Target inventory turns (default: 6.0, matches P&G/Unilever). "
             "Use 5.0 for Colgate-like conservative, 7.0 for leaner operation.",
    )
    parser.add_argument(
        "--target-service",
        type=float,
        default=0.97,
        help="Target service level (default: 0.97 = 97%%)",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    world_config_path = project_root / "src/prism_sim/config/world_definition.json"
    sim_config_path = project_root / "src/prism_sim/config/simulation_config.json"
    static_world_dir = project_root / "data/output/static_world"

    # Check if static world exists
    if not static_world_dir.exists():
        print("ERROR: Static world not found. Run generate_static_world.py first.")
        return

    # Load configurations
    world_config = load_json(world_config_path)
    sim_config = load_json(sim_config_path)

    # Load static world data
    node_counts = count_nodes_by_type(static_world_dir / "locations.csv")
    recipe_rates = get_recipe_run_rates(static_world_dir / "recipes.csv")
    products_by_category = get_products_by_category(static_world_dir / "products.csv")

    # Analyze
    demand_analysis = calculate_daily_demand(
        world_config, sim_config, node_counts, products_by_category
    )
    capacity_analysis = calculate_plant_capacity(
        sim_config, recipe_rates, products_by_category
    )

    # v0.31.0: Use target turns as primary driver
    recommendations = derive_optimal_parameters(
        demand_analysis,
        capacity_analysis,
        sim_config,
        target_turns=args.target_turns,  # None means use config default
        target_service_level=args.target_service,
    )

    # Add seasonal capacity analysis
    recommendations["seasonal_capacity"] = derive_seasonal_capacity_params(
        sim_config, recommendations
    )

    # Validate config consistency
    violations = validate_config_consistency(sim_config, recommendations)

    # Add seasonal balance warnings
    seasonal_warnings = validate_seasonal_balance(sim_config, recommendations)
    violations.extend(seasonal_warnings)

    # Report
    print_report(demand_analysis, capacity_analysis, recommendations, violations)

    # Apply if requested
    if args.apply:
        apply_recommendations(sim_config_path, recommendations)
    else:
        print("\nRun with --apply to update simulation_config.json")


if __name__ == "__main__":
    main()
