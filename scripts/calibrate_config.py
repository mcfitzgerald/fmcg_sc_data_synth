#!/usr/bin/env python3
"""
Calibration script for Prism Sim configuration parameters.

Derives optimal simulation parameters from world definition and static world data
using supply chain physics rather than guesswork.

Usage:
    poetry run python scripts/calibrate_config.py              # Analyze and recommend
    poetry run python scripts/calibrate_config.py --apply      # Apply recommendations
    poetry run python scripts/calibrate_config.py --target-oee 0.85  # Custom OEE target
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
    target_oee: float = 0.85,
    target_service_level: float = 0.95,
) -> dict[str, Any]:
    """
    Derive optimal configuration parameters based on physics.

    Key relationships:
    - OEE = Actual Production / Theoretical Capacity
    - For target OEE of 85%, we want Capacity ≈ Demand / 0.85
    - production_rate_multiplier scales theoretical capacity
    - SLOB thresholds = expected_network_DOS × margin
    """
    total_demand = demand_analysis["total_daily_demand"]
    theoretical_capacity = capacity_analysis["total_theoretical_capacity"]
    current_multiplier = capacity_analysis["current_multiplier"]

    # Required capacity to meet demand with target OEE
    # If OEE = Demand / Capacity, then Capacity = Demand / OEE
    required_capacity = total_demand / target_oee

    # Calculate optimal multiplier
    # new_capacity = theoretical_capacity × new_multiplier
    # required_capacity = theoretical_capacity × optimal_multiplier
    optimal_multiplier = required_capacity / theoretical_capacity if theoretical_capacity > 0 else 1.0

    # =================================================================
    # PHYSICS-BASED PRIMING DERIVATION (v0.27.0)
    # =================================================================
    # Priming must provide enough inventory to:
    # 1. Cover demand during replenishment lead time
    # 2. Buffer demand variability (safety stock)
    # 3. Account for order cycle (time between orders)
    # 4. NOT trigger immediate production (priming > trigger)
    #
    # Key formula: DOS = cycle_stock + safety_stock
    #   cycle_stock = order_cycle_days / 2 (average during cycle)
    #   safety_stock = z × σ × √(lead_time + review_period)
    # =================================================================
    sim_params = sim_config.get("simulation_parameters", {})
    mfg_config = sim_params.get("manufacturing", {})
    campaign_config = mfg_config.get("mrp_thresholds", {}).get("campaign_batching", {})
    replen_config = sim_params.get("agents", {}).get("replenishment", {})
    log_config = sim_params.get("logistics", {})

    # Core parameters from config
    lead_time_days = log_config.get("default_lead_time_days", 3.0)
    order_cycle_days = replen_config.get("order_cycle_days", 5)
    production_horizon = campaign_config.get("production_horizon_days", 6)

    # Service level parameters
    # z-score for target service level (1.65 = 95%, 2.33 = 99%)
    safety_factor_a = 2.33  # A-items: 99% service
    safety_factor_b = 1.65  # B-items: 95% service
    safety_factor_c = 1.28  # C-items: 90% service

    # Demand variability (CV = std/mean)
    # Store-level: high variability (~0.4)
    # DC-level: aggregation reduces variability (~0.25)
    # RDC-level: more aggregation (~0.15)
    cv_store = 0.4
    cv_dc = 0.25
    cv_rdc = 0.15

    # =================================================================
    # Store-Level Priming (highest variability, needs most buffer)
    # =================================================================
    # Cycle stock: half the order cycle (average inventory between orders)
    store_cycle_stock = order_cycle_days / 2

    # Safety stock: z × CV × demand × √(lead_time + review_period)
    # Simplified: safety_days = z × CV × √(lead_time)
    store_safety_a = safety_factor_a * cv_store * (lead_time_days ** 0.5)
    store_safety_b = safety_factor_b * cv_store * (lead_time_days ** 0.5)
    store_safety_c = safety_factor_c * cv_store * (lead_time_days ** 0.5)

    # Total store DOS by ABC class
    store_dos_a = round(store_cycle_stock + store_safety_a + lead_time_days, 1)
    store_dos_b = round(store_cycle_stock + store_safety_b + lead_time_days, 1)
    store_dos_c = round(store_cycle_stock + store_safety_c + lead_time_days, 1)

    # Weighted average for config (will apply ABC factors on top)
    # Use B-item as base since it's the majority
    store_dos = store_dos_b

    # =================================================================
    # DC/RDC-Level Priming (aggregation reduces variability)
    # =================================================================
    # DCs buffer for downstream stores + own replenishment cycle
    dc_cycle_stock = order_cycle_days / 2
    dc_safety = safety_factor_b * cv_dc * (lead_time_days ** 0.5)
    dc_dos = round(dc_cycle_stock + dc_safety + lead_time_days, 1)

    # RDCs buffer for DCs + production lead time
    rdc_cycle_stock = production_horizon / 2  # Half of production batch
    rdc_safety = safety_factor_b * cv_rdc * (lead_time_days ** 0.5)
    rdc_dos = round(rdc_cycle_stock + rdc_safety + lead_time_days, 1)

    # =================================================================
    # Trigger Threshold Derivation (Network-Level)
    # =================================================================
    # Triggers are NETWORK DOS thresholds that start production
    # When network DOS drops below trigger, production begins
    #
    # trigger = production_time + transit_time + safety_buffer
    # - production_time: production_horizon (time to produce batch)
    # - transit_time: lead_time_days (plant to RDC)
    # - safety_buffer: varies by ABC (higher for A-items)
    #
    # A-items: High service target, need more safety buffer
    # C-items: Lower service target, can run leaner
    production_time = production_horizon
    transit_time = lead_time_days
    safety_buffer_a = 5  # Extra buffer for 99% service
    safety_buffer_b = 3  # Moderate buffer for 95% service
    safety_buffer_c = 1  # Lean buffer for 90% service

    trigger_a = int(production_time + transit_time + safety_buffer_a)
    trigger_b = int(production_time + transit_time + safety_buffer_b)
    trigger_c = int(production_time + transit_time + safety_buffer_c)

    # =================================================================
    # ABC Velocity Factors for Priming
    # =================================================================
    # A-items get more buffer (higher service), C-items less
    abc_priming_factors = {
        "A": round(store_dos_a / store_dos, 2),
        "B": 1.0,
        "C": round(store_dos_c / store_dos, 2),
    }

    # Initial inventory multiplier for RDCs serving many stores
    # Should be lower since demand aggregates (law of large numbers)
    rdc_store_multiplier = demand_analysis["n_stores"] / 40  # ~150 for 6000 stores

    # =================================================================
    # SLOB THRESHOLD CALIBRATION (v0.26.0)
    # =================================================================
    # Expected network DOS = sum of echelon targets + batch buffer
    # SLOB threshold should be expected_DOS × margin to flag truly slow items
    #
    # Network DOS components:
    #   - Store inventory: store_dos
    #   - DC inventory: dc_dos (customer DCs)
    #   - RDC inventory: rdc_dos
    #   - In-transit: lead_time_days
    #   - Batch buffer: production_horizon / 2 (avg across batch cycle)
    # =================================================================
    validation_config = sim_params.get("validation", {})

    # Expected network-wide DOS by ABC class
    # A-items turn faster, C-items turn slower
    # Base network DOS = sum of all echelon targets
    base_network_dos = store_dos + dc_dos + rdc_dos + lead_time_days

    # ABC velocity factors from config (A turns faster, C turns slower)
    # These reflect that A-items have higher velocity and should turn faster
    abc_velocity_factors = validation_config.get(
        "slob_abc_velocity_factors", {"A": 0.7, "B": 1.0, "C": 1.5}
    )

    # SLOB margin from config: flag inventory with DOS > expected × margin
    # 1.5x margin means "50% more than expected = slow moving"
    slob_margin = validation_config.get("slob_margin", 1.5)

    slob_thresholds = {
        abc_class: round(base_network_dos * velocity_factor * slob_margin, 0)
        for abc_class, velocity_factor in abc_velocity_factors.items()
    }

    # Read current SLOB thresholds from config for comparison
    current_slob_thresholds = validation_config.get("slob_abc_thresholds", {})

    return {
        "analysis": {
            "total_daily_demand": total_demand,
            "theoretical_capacity_base": theoretical_capacity,
            "current_multiplier": current_multiplier,
            "current_capacity": theoretical_capacity * current_multiplier,
            "capacity_utilization": total_demand / (theoretical_capacity * current_multiplier) if theoretical_capacity * current_multiplier > 0 else 0,
            "target_oee": target_oee,
            "required_capacity": required_capacity,
        },
        "inventory_analysis": {
            "store_dos_a": store_dos_a,
            "store_dos_b": store_dos_b,
            "store_dos_c": store_dos_c,
            "store_dos": round(store_dos, 1),
            "dc_dos": round(dc_dos, 1),
            "rdc_dos": round(rdc_dos, 1),
            "lead_time_days": lead_time_days,
            "order_cycle_days": order_cycle_days,
            "production_horizon": production_horizon,
            "base_network_dos": round(base_network_dos, 1),
        },
        "trigger_analysis": {
            "trigger_a": int(trigger_a),
            "trigger_b": int(trigger_b),
            "trigger_c": int(trigger_c),
            "current_trigger_a": campaign_config.get("trigger_dos_a", 14),
            "current_trigger_b": campaign_config.get("trigger_dos_b", 10),
            "current_trigger_c": campaign_config.get("trigger_dos_c", 7),
        },
        "slob_analysis": {
            "current_thresholds": current_slob_thresholds,
            "derived_thresholds": slob_thresholds,
            "slob_margin": slob_margin,
            "abc_velocity_factors": abc_velocity_factors,
        },
        "recommendations": {
            "production_rate_multiplier": round(optimal_multiplier, 1),
            "store_days_supply": round(store_dos, 1),
            "rdc_days_supply": round(rdc_dos, 1),
            "customer_dc_days_supply": round(dc_dos, 1),
            "abc_velocity_factors": abc_priming_factors,
            "trigger_dos_a": int(trigger_a),
            "trigger_dos_b": int(trigger_b),
            "trigger_dos_c": int(trigger_c),
            "rdc_store_multiplier": round(rdc_store_multiplier, 1),
            "slob_abc_thresholds": slob_thresholds,
        },
        "expected_metrics": {
            "expected_oee": target_oee,
            "expected_daily_production": total_demand,
            "expected_service_level": target_service_level,
            "expected_network_dos": round(base_network_dos, 1),
            "expected_turns": round(365.0 / base_network_dos, 1),
        },
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


def print_report(
    demand_analysis: dict[str, Any],
    capacity_analysis: dict[str, Any],
    recommendations: dict[str, Any],
    violations: list[str] | None = None,
) -> None:
    """Print a nice calibration report."""
    print("\n" + "=" * 60)
    print("       PRISM SIM CONFIGURATION CALIBRATION REPORT")
    print("=" * 60)

    print("\n--- DEMAND ANALYSIS ---")
    print(f"Total Stores: {demand_analysis['n_stores']:,}")
    print(f"Total SKUs: {demand_analysis['total_skus']:,}")
    print(f"SKUs by Category:")
    for cat, count in demand_analysis["sku_counts"].items():
        print(f"  {cat}: {count}")
    print(f"\nDaily Demand by Category:")
    for cat, demand in demand_analysis["category_demand"].items():
        print(f"  {cat}: {demand:,.0f} cases/day")
    print(f"\nTOTAL DAILY DEMAND: {demand_analysis['total_daily_demand']:,.0f} cases/day")

    print("\n--- CAPACITY ANALYSIS (per line, before multiplier) ---")
    print(f"Base Capacity (1 line/plant): {capacity_analysis['total_theoretical_capacity']:,.0f} cases/day")
    print(f"Current Multiplier: {capacity_analysis['current_multiplier']}x")
    print(f"Current Effective Capacity: {capacity_analysis['total_with_multiplier']:,.0f} cases/day")
    print(f"\nPlant Breakdown (1 line each):")
    for plant_id, cap in capacity_analysis["plant_capacities"].items():
        print(f"  {plant_id}:")
        print(f"    Categories: {', '.join(cap['supported_categories'])}")
        print(f"    Avg Run Rate: {cap['avg_run_rate_per_hour']:,.0f} cases/hr")
        print(f"    SKUs Supported: {cap['n_supported_skus']}")
        print(f"    Daily Capacity/Line: {cap['theoretical_capacity_per_line']:,.0f} cases")

    print("\n--- CURRENT STATE ---")
    analysis = recommendations["analysis"]
    util = analysis["capacity_utilization"]
    print(f"Capacity Utilization: {util:.1%}")
    if util > 1:
        print("  WARNING: Demand exceeds capacity! Service will suffer.")
    elif util < 0.5:
        print("  WARNING: Capacity vastly exceeds demand. OEE will be low.")

    print("\n--- INVENTORY DOS ANALYSIS (Physics-Based) ---")
    inv_analysis = recommendations["inventory_analysis"]
    print(f"Store DOS by ABC: A={inv_analysis['store_dos_a']}d, "
          f"B={inv_analysis['store_dos_b']}d, C={inv_analysis['store_dos_c']}d")
    print(f"Store DOS (Base): {inv_analysis['store_dos']} days")
    print(f"DC DOS: {inv_analysis['dc_dos']} days")
    print(f"RDC DOS: {inv_analysis['rdc_dos']} days")
    print(f"Lead Time: {inv_analysis['lead_time_days']} days")
    print(f"Order Cycle: {inv_analysis['order_cycle_days']} days")
    print(f"Production Horizon: {inv_analysis['production_horizon']} days")
    print(f"Expected Network DOS: {inv_analysis['base_network_dos']} days")

    print("\n--- TRIGGER THRESHOLD ANALYSIS ---")
    trig = recommendations["trigger_analysis"]
    print(f"Derived Triggers: A={trig['trigger_a']}d, B={trig['trigger_b']}d, "
          f"C={trig['trigger_c']}d")
    print(f"Current Triggers: A={trig['current_trigger_a']}d, "
          f"B={trig['current_trigger_b']}d, C={trig['current_trigger_c']}d")

    print("\n--- SLOB THRESHOLD CALIBRATION ---")
    slob = recommendations["slob_analysis"]
    print(f"SLOB Margin: {slob['slob_margin']}x expected DOS")
    print(f"ABC Velocity Factors: {slob['abc_velocity_factors']}")
    print(f"\nCurrent SLOB Thresholds: {slob['current_thresholds']}")
    print(f"Derived SLOB Thresholds: {slob['derived_thresholds']}")

    # Flag mismatches
    current = slob["current_thresholds"]
    derived = slob["derived_thresholds"]
    if current:
        mismatches = []
        for abc_class in ["A", "B", "C"]:
            curr_val = current.get(abc_class, 0)
            deriv_val = derived.get(abc_class, 0)
            if abs(curr_val - deriv_val) > 5:  # >5 day difference
                mismatches.append(f"  {abc_class}: {curr_val} → {deriv_val}")
        if mismatches:
            print("\n  WARNING: SLOB thresholds mismatch physics!")
            for m in mismatches:
                print(m)

    print("\n--- RECOMMENDATIONS ---")
    rec = recommendations["recommendations"]
    print(f"production_rate_multiplier: {rec['production_rate_multiplier']}")
    print(f"store_days_supply: {rec['store_days_supply']}")
    print(f"rdc_days_supply: {rec['rdc_days_supply']}")
    print(f"customer_dc_days_supply: {rec['customer_dc_days_supply']}")
    print(f"abc_velocity_factors: {rec['abc_velocity_factors']}")
    print(f"trigger_dos: A={rec['trigger_dos_a']}, B={rec['trigger_dos_b']}, "
          f"C={rec['trigger_dos_c']}")
    print(f"rdc_store_multiplier: {rec['rdc_store_multiplier']}")
    print(f"slob_abc_thresholds: {rec['slob_abc_thresholds']}")

    print("\n--- EXPECTED METRICS (after calibration) ---")
    exp = recommendations["expected_metrics"]
    print(f"Expected OEE: {exp['expected_oee']:.0%}")
    print(f"Expected Daily Production: {exp['expected_daily_production']:,.0f} cases")
    print(f"Expected Service Level: {exp['expected_service_level']:.0%}")
    print(f"Expected Network DOS: {exp['expected_network_dos']} days")
    print(f"Expected Inventory Turns: {365.0 / exp['expected_network_dos']:.1f}x")

    # Print config consistency violations/warnings
    if violations:
        print("\n--- CONFIG CONSISTENCY WARNINGS ---")
        for v in violations:
            print(f"  WARNING: {v}")
    else:
        print("\n--- CONFIG CONSISTENCY ---")
        print("  All config parameters are internally consistent.")

    print("\n" + "=" * 60)


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

    # Update trigger thresholds (v0.27.0)
    mrp = mfg.setdefault("mrp_thresholds", {})
    campaign = mrp.setdefault("campaign_batching", {})
    campaign["trigger_dos_a"] = rec["trigger_dos_a"]
    campaign["trigger_dos_b"] = rec["trigger_dos_b"]
    campaign["trigger_dos_c"] = rec["trigger_dos_c"]

    # Update inventory initialization
    inv = sim_params.setdefault("inventory", {})
    init = inv.setdefault("initialization", {})
    init["store_days_supply"] = rec["store_days_supply"]
    init["rdc_days_supply"] = rec["rdc_days_supply"]
    init["customer_dc_days_supply"] = rec["customer_dc_days_supply"]
    init["abc_velocity_factors"] = rec["abc_velocity_factors"]
    init["rdc_store_multiplier"] = rec["rdc_store_multiplier"]

    # Update SLOB thresholds (v0.26.0)
    validation = sim_params.setdefault("validation", {})
    validation["slob_abc_thresholds"] = rec["slob_abc_thresholds"]

    save_json(sim_config_path, config)
    print(f"\nApplied recommendations to {sim_config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Prism Sim configuration parameters"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply recommendations to simulation_config.json",
    )
    parser.add_argument(
        "--target-oee",
        type=float,
        default=0.85,
        help="Target OEE (default: 0.85)",
    )
    parser.add_argument(
        "--target-service",
        type=float,
        default=0.95,
        help="Target service level (default: 0.95)",
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
    recommendations = derive_optimal_parameters(
        demand_analysis,
        capacity_analysis,
        sim_config,
        target_oee=args.target_oee,
        target_service_level=args.target_service,
    )

    # Validate config consistency
    violations = validate_config_consistency(sim_config, recommendations)

    # Report
    print_report(demand_analysis, capacity_analysis, recommendations, violations)

    # Apply if requested
    if args.apply:
        apply_recommendations(sim_config_path, recommendations)
    else:
        print("\nRun with --apply to update simulation_config.json")


if __name__ == "__main__":
    main()
