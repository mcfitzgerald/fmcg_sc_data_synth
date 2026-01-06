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
    target_oee: float = 0.85,
    target_service_level: float = 0.95,
) -> dict[str, Any]:
    """
    Derive optimal configuration parameters based on physics.

    Key relationships:
    - OEE = Actual Production / Theoretical Capacity
    - For target OEE of 85%, we want Capacity ≈ Demand / 0.85
    - production_rate_multiplier scales theoretical capacity
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

    # Calculate inventory parameters
    # For 95% service level with ~3 day lead time, need safety stock
    lead_time_days = 3.0
    safety_factor = 1.65  # z-score for 95% service level

    # Store DOS = lead_time + safety_stock_days
    # Simplified: store_dos ≈ lead_time × (1 + safety_factor × CV)
    # Assuming CV ≈ 0.3 for demand variability
    cv = 0.3
    store_dos = lead_time_days * (1 + safety_factor * cv)

    # RDC needs to cover downstream lead time + store DOS
    rdc_dos = store_dos + lead_time_days

    # Initial inventory multiplier for RDCs serving many stores
    # Should be lower since demand aggregates (law of large numbers)
    rdc_store_multiplier = demand_analysis["n_stores"] / 40  # ~100 for 4000 stores

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
        "recommendations": {
            "production_rate_multiplier": round(optimal_multiplier, 1),
            "store_days_supply": round(store_dos, 1),
            "rdc_days_supply": round(rdc_dos, 1),
            "customer_dc_days_supply": round(rdc_dos, 1),
            "rdc_store_multiplier": round(rdc_store_multiplier, 1),
        },
        "expected_metrics": {
            "expected_oee": target_oee,
            "expected_daily_production": total_demand,
            "expected_service_level": target_service_level,
        },
    }


def print_report(
    demand_analysis: dict[str, Any],
    capacity_analysis: dict[str, Any],
    recommendations: dict[str, Any],
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

    print("\n--- RECOMMENDATIONS ---")
    rec = recommendations["recommendations"]
    print(f"production_rate_multiplier: {rec['production_rate_multiplier']}")
    print(f"store_days_supply: {rec['store_days_supply']}")
    print(f"rdc_days_supply: {rec['rdc_days_supply']}")
    print(f"customer_dc_days_supply: {rec['customer_dc_days_supply']}")
    print(f"rdc_store_multiplier: {rec['rdc_store_multiplier']}")

    print("\n--- EXPECTED METRICS (after calibration) ---")
    exp = recommendations["expected_metrics"]
    print(f"Expected OEE: {exp['expected_oee']:.0%}")
    print(f"Expected Daily Production: {exp['expected_daily_production']:,.0f} cases")
    print(f"Expected Service Level: {exp['expected_service_level']:.0%}")

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

    # Update inventory initialization
    inv = sim_params.setdefault("inventory", {})
    init = inv.setdefault("initialization", {})
    init["store_days_supply"] = rec["store_days_supply"]
    init["rdc_days_supply"] = rec["rdc_days_supply"]
    init["customer_dc_days_supply"] = rec["customer_dc_days_supply"]
    init["rdc_store_multiplier"] = rec["rdc_store_multiplier"]

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
        target_oee=args.target_oee,
        target_service_level=args.target_service,
    )

    # Report
    print_report(demand_analysis, capacity_analysis, recommendations)

    # Apply if requested
    if args.apply:
        apply_recommendations(sim_config_path, recommendations)
    else:
        print("\nRun with --apply to update simulation_config.json")


if __name__ == "__main__":
    main()
