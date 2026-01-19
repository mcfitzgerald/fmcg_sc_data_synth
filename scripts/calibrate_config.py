#!/usr/bin/env python3
"""
Calibration script for Prism Sim configuration parameters.

Derives optimal simulation parameters from world definition and static world data
using supply chain physics and industry benchmarks (FMCG).

v0.35.2: Capacity planning - derive num_lines from target OEE.
- New --derive-lines flag derives num_lines per plant from target OEE
- Inverts OEE formula: OEE = Availability × Performance × Quality
- Warns for target OEE > 85% (VUT curve lead time explosion)
- Errors for target OEE > 95% (physically impossible with downtime)

v0.33.0: Multi-echelon lead time cascade and MRP signal lag awareness.
- Calculates cumulative lead times through 4-tier network
- Accounts for FTL consolidation delays
- Adds MRP rolling window signal lag to trigger thresholds
- Derives ABC priming factors from z-scores (no hardcodes)
- Validates against v0.32.1 baseline for regression prevention

v0.31.0: Major rewrite to use target inventory turns as primary driver,
matching real FMCG company performance (Colgate 4.1x, P&G 5.5x, Unilever 6.2x).

Usage:
    poetry run python scripts/calibrate_config.py              # Analyze and recommend
    poetry run python scripts/calibrate_config.py --apply      # Apply recommendations
    poetry run python scripts/calibrate_config.py --target-turns 6.0  # Industry benchmark
    poetry run python scripts/calibrate_config.py --derive-lines --target-oee 0.60  # Capacity planning
"""

import argparse
import csv
import json
import math
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


def calculate_multi_echelon_lead_times(
    links_path: Path,
    sim_config: dict[str, Any],
) -> dict[str, Any]:
    """
    v0.33.0: Calculate actual multi-echelon lead times from network topology.

    Analyzes links.csv to compute average lead times for each echelon transition:
    - Store ← Customer DC
    - Customer DC ← RDC
    - RDC ← Plant
    - Plant ← Supplier

    Also estimates FTL consolidation delays based on logistics config.

    Returns:
        Dict with lead times by echelon and cumulative totals.
    """
    # Load lead times from links
    echelon_lead_times: dict[str, list[float]] = {
        "dc_to_store": [],
        "rdc_to_dc": [],
        "plant_to_rdc": [],
        "supplier_to_plant": [],
    }

    with open(links_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row["source_id"]
            target_id = row["target_id"]
            lt = float(row["lead_time_days"])

            # Classify by echelon transition
            if source_id.startswith("RET-DC-") or source_id.startswith("DIST-DC-"):
                # Customer DC -> Store
                echelon_lead_times["dc_to_store"].append(lt)
            elif source_id.startswith("RDC-"):
                # RDC -> Customer DC or RDC -> Club Store
                if target_id.startswith("STORE-"):
                    echelon_lead_times["dc_to_store"].append(lt)
                else:
                    echelon_lead_times["rdc_to_dc"].append(lt)
            elif source_id.startswith("PLANT-"):
                # Plant -> RDC
                echelon_lead_times["plant_to_rdc"].append(lt)
            elif source_id.startswith("SUP-"):
                # Supplier -> Plant
                echelon_lead_times["supplier_to_plant"].append(lt)

    # Calculate statistics
    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0}
        sorted_vals = sorted(values)
        p90_idx = int(len(sorted_vals) * 0.9)
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p90": sorted_vals[min(p90_idx, len(sorted_vals) - 1)],
        }

    echelon_stats = {k: stats(v) for k, v in echelon_lead_times.items()}

    # Get FTL consolidation estimate from config
    log_config = sim_config.get("simulation_parameters", {}).get("logistics", {})
    calibration = sim_config.get("simulation_parameters", {}).get("calibration", {})
    multi_echelon_cfg = calibration.get("multi_echelon_lead_times", {})

    # FTL consolidation adds delay waiting for minimum pallets
    ftl_consolidation = multi_echelon_cfg.get("ftl_consolidation_buffer", 2.0)

    # Production lead time
    mfg_config = sim_config.get("simulation_parameters", {}).get("manufacturing", {})
    prod_lt = mfg_config.get("production_lead_time_days", 3)

    # Calculate cumulative lead times for replenishment paths
    # Using P90 values to ensure adequate coverage (not just average)
    store_lt = echelon_stats["dc_to_store"]["p90"]
    dc_lt = echelon_stats["rdc_to_dc"]["p90"] + ftl_consolidation
    rdc_lt = echelon_stats["plant_to_rdc"]["p90"] + ftl_consolidation

    # Full cascade: Store replenishment from Plant production
    # Store <- DC <- RDC <- Plant
    cumulative = {
        "store": store_lt,
        "customer_dc": store_lt + dc_lt,
        "rdc": store_lt + dc_lt + rdc_lt,
        "plant": store_lt + dc_lt + rdc_lt + prod_lt,
    }

    return {
        "echelon_stats": echelon_stats,
        "ftl_consolidation": ftl_consolidation,
        "production_lead_time": prod_lt,
        "cumulative": cumulative,
        "network_replenishment_time": cumulative["rdc"],  # RDC sees Plant production after this
    }


def derive_abc_priming_factors(
    z_scores: dict[str, float],
) -> dict[str, float]:
    """
    v0.33.0: Derive ABC priming factors from service level z-scores.

    The priming factor determines how much inventory buffer each ABC class gets.
    Higher service level target (higher z-score) = more buffer needed.

    Formula: factor_X = z_X / z_B (B-items are baseline)

    This ensures:
    - A-items (z=2.33, 99% SL) get ~1.17x buffer
    - B-items (z=2.00, 97% SL) get 1.0x buffer (baseline)
    - C-items (z=1.65, 95% SL) get ~0.83x buffer

    CRITICAL: A > B > C must hold for proper service level hierarchy.
    """
    z_a = z_scores.get("A", 2.33)
    z_b = z_scores.get("B", 2.0)
    z_c = z_scores.get("C", 1.65)

    if z_b == 0:
        z_b = 2.0  # Prevent division by zero

    factors = {
        "A": round(z_a / z_b, 3),
        "B": 1.0,
        "C": round(z_c / z_b, 3),
    }

    # Validate hierarchy
    if not (factors["A"] >= factors["B"] >= factors["C"]):
        print(f"WARNING: ABC hierarchy violated! A={factors['A']}, B={factors['B']}, C={factors['C']}")
        # Force correct hierarchy
        factors["A"] = max(factors["A"], 1.1)
        factors["C"] = min(factors["C"], 0.9)

    return factors


def validate_against_baseline(
    recommendations: dict[str, Any],
    sim_config: dict[str, Any],
) -> list[str]:
    """
    v0.33.0: Validate derived values against v0.32.1 baseline.

    Prevents regressions like v0.32.0 where calibration overwrote
    empirically-tuned values with physics-derived values that were wrong.

    Returns list of warnings if values deviate significantly from baseline.
    """
    warnings: list[str] = []

    calibration = sim_config.get("simulation_parameters", {}).get("calibration", {})
    baseline = calibration.get("baseline_reference", {})

    if not baseline:
        warnings.append("No baseline_reference in config - cannot validate")
        return warnings

    rec = recommendations.get("recommendations", {})

    # Check critical parameters against baseline
    checks = [
        ("store_days_supply", 0.3),  # Allow 30% deviation
        ("rdc_days_supply", 0.3),
        ("customer_dc_days_supply", 0.3),
        ("trigger_dos_a", 0.5),  # Triggers can vary more
        ("trigger_dos_b", 0.5),
        ("trigger_dos_c", 0.5),
    ]

    for param, max_deviation in checks:
        baseline_val = baseline.get(param)
        derived_val = rec.get(param)

        if baseline_val is None or derived_val is None:
            continue

        if baseline_val == 0:
            continue

        deviation = abs(derived_val - baseline_val) / baseline_val

        if deviation > max_deviation:
            warnings.append(
                f"{param}: derived={derived_val:.1f} vs baseline={baseline_val:.1f} "
                f"(deviation={deviation:.0%} > {max_deviation:.0%})"
            )

    # Check ABC factor hierarchy
    abc_factors = rec.get("abc_velocity_factors", {})
    if abc_factors:
        a_factor = abc_factors.get("A", 1.0)
        b_factor = abc_factors.get("B", 1.0)
        c_factor = abc_factors.get("C", 1.0)

        if not (a_factor >= b_factor >= c_factor):
            warnings.append(
                f"ABC hierarchy VIOLATED: A={a_factor}, B={b_factor}, C={c_factor} "
                f"(must be A >= B >= C)"
            )

    return warnings


def calculate_daily_demand(
    world_config: dict[str, Any],
    sim_config: dict[str, Any],
    node_counts: dict[str, int],
    products_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Calculate expected daily demand from configuration.

    Demand = sum over stores of (base_demand × format_scale × sku_count) × realism_factor

    The realism_factor accounts for:
    - Segment weights (products without value_segment get 0.5 default)
    - Zipf distribution effects (heavy tail reduces average contribution)
    - Format scale distribution across store types

    Empirically measured: actual POSEngine demand / theoretical = ~0.41
    """
    sim_params = sim_config.get("simulation_parameters", {})
    demand_config = sim_params.get("demand", {})
    category_profiles = demand_config.get("category_profiles", {})
    format_scales = demand_config.get("format_scale_factors", {})

    # Get demand realism factor from capacity_planning config
    calibration_config = sim_params.get("calibration", {})
    capacity_planning = calibration_config.get("capacity_planning", {})
    demand_realism_factor = capacity_planning.get("demand_realism_factor", 0.41)

    # Count stores (all STORE type nodes)
    n_stores = node_counts.get("STORE", 0)

    # Count SKUs per category
    sku_counts = {cat: len(skus) for cat, skus in products_by_category.items()}
    total_skus = sum(sku_counts.values())

    # Calculate demand per category
    category_demand = {}
    theoretical_daily_demand = 0.0

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
        theoretical_daily_demand += cat_daily_demand

    # Apply realism factor to get effective demand
    effective_daily_demand = theoretical_daily_demand * demand_realism_factor

    return {
        "n_stores": n_stores,
        "total_skus": total_skus,
        "sku_counts": sku_counts,
        "category_demand": category_demand,
        "theoretical_daily_demand": theoretical_daily_demand,
        "demand_realism_factor": demand_realism_factor,
        "total_daily_demand": effective_daily_demand,  # Use effective demand for planning
    }


def calculate_plant_capacity(
    sim_config: dict[str, Any],
    recipe_rates: dict[str, float],
    products_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Calculate theoretical plant capacity from configuration.

    v0.32.0: Now uses explicit line count per plant instead of rate multiplier.
    Capacity per plant = effective_hours × avg_run_rate × num_lines
    """
    sim_params = sim_config.get("simulation_parameters", {})
    mfg_config = sim_params.get("manufacturing", {})

    hours_per_day = mfg_config.get("production_hours_per_day", 24.0)
    efficiency = mfg_config.get("efficiency_factor", 0.95)
    downtime = mfg_config.get("unplanned_downtime_pct", 0.05)
    default_num_lines = mfg_config.get("default_num_lines", 4)
    # multiplier should now be 1.0, but we keep it for reference or if used as a tuning knob
    current_multiplier = mfg_config.get("production_rate_multiplier", 1.0)

    plant_params = mfg_config.get("plant_parameters", {})

    # Calculate capacity per plant
    plant_capacities = {}
    total_theoretical_capacity = 0.0

    for plant_id, params in plant_params.items():
        supported_cats = params.get("supported_categories", [])
        plant_efficiency = params.get("efficiency_factor", efficiency)
        plant_downtime = params.get("unplanned_downtime_pct", downtime)
        num_lines = params.get("num_lines", default_num_lines)

        # Get AVERAGE run rate for supported SKUs
        run_rates = []
        for cat in supported_cats:
            for product_id in products_by_category.get(cat, []):
                rate = recipe_rates.get(product_id, 0.0)
                if rate > 0:
                    run_rates.append(rate)

        avg_run_rate = sum(run_rates) / len(run_rates) if run_rates else 0.0

        # Theoretical capacity = hours × avg_run_rate × efficiency × (1 - downtime) × num_lines
        effective_hours = hours_per_day * plant_efficiency * (1 - plant_downtime)
        plant_capacity = effective_hours * avg_run_rate * num_lines

        plant_capacities[plant_id] = {
            "supported_categories": supported_cats,
            "avg_run_rate_per_hour": avg_run_rate,
            "n_supported_skus": len(run_rates),
            "effective_hours": effective_hours,
            "num_lines": num_lines,
            "theoretical_capacity_total": plant_capacity,
        }
        total_theoretical_capacity += plant_capacity

    return {
        "hours_per_day": hours_per_day,
        "current_multiplier": current_multiplier,
        "plant_capacities": plant_capacities,
        "total_theoretical_capacity": total_theoretical_capacity,
        # With line logic, total_theoretical_capacity IS the total capacity (multiplier should be 1.0)
        "total_with_multiplier": total_theoretical_capacity * current_multiplier,
    }


def calculate_campaign_efficiency(
    production_horizon_days: int,
    trigger_dos_a: int,
    trigger_dos_b: int,
    trigger_dos_c: int,
    abc_mix: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    v0.36.3: Calculate DOS cycling efficiency for campaign batching.

    Campaign batching creates idle time because products only produce when
    DOS < trigger. After producing production_horizon_days worth, DOS jumps
    above trigger and the product waits until DOS drops again.

    The production cycle for each product is:
    - Produce when DOS < trigger (e.g., 31 days)
    - After producing 14-day horizon, DOS jumps to ~45
    - Wait ~31 days for DOS to drop below trigger again

    Returns:
        (dos_coverage_factor, effective_efficiency)
    """
    if abc_mix is None:
        abc_mix = {"A": 0.80, "B": 0.15, "C": 0.05}

    # Weighted average trigger based on ABC mix
    avg_trigger = (
        trigger_dos_a * abc_mix["A"]
        + trigger_dos_b * abc_mix["B"]
        + trigger_dos_c * abc_mix["C"]
    )

    # Theoretical: fraction of cycle spent producing
    # cycle_length = production_horizon + avg_trigger (time until DOS drops again)
    theoretical = production_horizon_days / (production_horizon_days + avg_trigger)

    # Empirical correction: staggered product cycles + scheduling friction
    # Calibrated from empirical data: theoretical ~0.32, actual ~0.45-0.50
    # The network-wide effect is better than per-product due to staggering
    stagger_benefit = 1.5

    dos_coverage = min(theoretical * stagger_benefit, 0.65)

    return dos_coverage, dos_coverage


def calculate_variability_buffer(
    seasonality_amplitude: float,
    noise_cv: float,
    safety_z: float = 1.28,
) -> float:
    """
    v0.36.3: Calculate capacity buffer for demand variability.

    Point estimate of demand doesn't account for ±2σ swings. Need reserve
    capacity to handle peaks without stockouts.

    Args:
        seasonality_amplitude: e.g., 0.12 for ±12%
        noise_cv: coefficient of variation for random demand
        safety_z: z-score for capacity planning (1.28 = 90%)

    Returns:
        Multiplier for required capacity (e.g., 1.25 = need 25% more)
    """
    combined_cv = (seasonality_amplitude**2 + noise_cv**2) ** 0.5

    # Ensure no divide-by-zero and cap at 2x
    if safety_z * combined_cv >= 0.95:
        return 2.0

    return 1.0 / (1.0 - safety_z * combined_cv)


def derive_num_lines_from_oee(
    target_oee: float,
    total_daily_demand: float,
    capacity_analysis: dict[str, Any],
    sim_config: dict[str, Any],
    products_by_category: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Derive num_lines per plant to achieve target OEE.

    Inverts: OEE = Availability × Performance × Quality

    Where:
    - Availability = (run + changeover) / total_scheduled
    - Performance = efficiency_factor (0.78-0.88 per plant)
    - Quality = yield_percent / 100 (0.985)

    v0.35.3: Now simulates MRP's round-robin product distribution to allocate
    lines proportionally to actual workload per plant.

    v0.35.4: Fixed changeover calculation for campaign batching. Instead of a
    simple overhead factor, explicitly calculate changeover hours based on:
    - num_batches_per_day = num_products / production_horizon_days
    - changeover_hours = num_batches × avg_changeover_time

    Args:
        target_oee: Target OEE (e.g., 0.60 for 60%)
        total_daily_demand: Total daily demand in cases
        capacity_analysis: Output from calculate_plant_capacity()
        sim_config: Simulation configuration
        products_by_category: Product lists per category (for accurate allocation)

    Returns:
        Dict with derived num_lines and supporting analysis.
    """
    mfg = sim_config["simulation_parameters"]["manufacturing"]
    hours_per_day = mfg.get("production_hours_per_day", 24.0)
    yield_pct = mfg.get("default_yield_percent", 98.5)
    quality = yield_pct / 100.0

    # Get capacity planning parameters from config
    calibration = sim_config["simulation_parameters"].get("calibration", {})
    cap_planning = calibration.get("capacity_planning", {})
    min_lines_per_plant = cap_planning.get("min_lines_per_plant", 2)
    max_oee_target = cap_planning.get("max_oee_target", 0.85)

    # Get campaign batching parameters for changeover calculation
    mrp_thresholds = mfg.get("mrp_thresholds", {})
    campaign_config = mrp_thresholds.get("campaign_batching", {})
    production_horizon_days = campaign_config.get("production_horizon_days", 7)

    # Validate target OEE
    if target_oee > 0.95:
        raise ValueError(
            f"Target OEE {target_oee:.1%} > 95% is physically impossible with downtime"
        )
    if target_oee > max_oee_target:
        print(
            f"WARNING: Target OEE {target_oee:.1%} > {max_oee_target:.0%} - "
            f"VUT curve causes lead time explosion at high utilization"
        )

    # Calculate weighted average efficiency across plants
    plant_params = mfg.get("plant_parameters", {})
    efficiencies = [p.get("efficiency_factor", 0.85) for p in plant_params.values()]
    avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.85

    # Inverse OEE calculation
    # OEE = Availability × Performance × Quality
    # target_availability = target_oee / (Performance × Quality)
    target_availability = target_oee / (avg_efficiency * quality)
    max_availability = 0.95  # Physical limit with downtime

    if target_availability > max_availability:
        raise ValueError(
            f"Target OEE {target_oee:.1%} requires availability "
            f"{target_availability:.1%} > max {max_availability:.1%}. "
            f"Lower the target OEE or improve efficiency/quality."
        )

    # Get average RAW run rate from plant capacities (not adjusted for efficiency/downtime)
    # This is critical - we need the raw rate since we account for availability separately
    plant_caps = capacity_analysis["plant_capacities"]
    current_lines = sum(p["num_lines"] for p in plant_caps.values())

    # Weighted average of raw run rates by number of SKUs per plant
    total_skus = sum(p["n_supported_skus"] for p in plant_caps.values())
    if total_skus > 0:
        avg_run_rate = sum(
            p["avg_run_rate_per_hour"] * p["n_supported_skus"]
            for p in plant_caps.values()
        ) / total_skus
    else:
        avg_run_rate = 20000  # Fallback

    # v0.35.4: Calculate operating hours explicitly for campaign batching
    #
    # With campaign batching, each product is produced once every production_horizon_days.
    # This means:
    # - num_batches_per_day = num_finished_products / production_horizon_days
    # - changeover_hours = num_batches × avg_changeover_time
    # - run_hours = total_daily_demand / avg_run_rate
    # - operating_hours = run_hours + changeover_hours
    #
    # IMPORTANT: If demand exceeds capacity, use capacity-constrained production volume.
    # This reflects what MRP actually produces (it scales down to capacity).
    #
    # Count finished products (non-ingredients)
    num_finished_products = sum(
        len(prods) for cat, prods in (products_by_category or {}).items()
        if cat != "INGREDIENT"
    )
    if num_finished_products == 0:
        num_finished_products = total_skus  # Fallback to plant capacity analysis

    # Get current capacity from analysis
    current_capacity = capacity_analysis["total_theoretical_capacity"]

    # Use capacity-constrained demand if capacity is binding
    # MRP scales sustainable demand to capacity when constrained
    capacity_constrained_production = min(total_daily_demand, current_capacity)
    capacity_constrained = total_daily_demand > current_capacity

    # v0.36.3: Physics-based campaign efficiency decomposition
    # Campaign batching creates idle time because DOS > trigger between production cycles.
    campaign_enabled = campaign_config.get("enabled", True)
    if campaign_enabled:
        dos_coverage, campaign_efficiency = calculate_campaign_efficiency(
            production_horizon_days=production_horizon_days,
            trigger_dos_a=campaign_config.get("trigger_dos_a", 31),
            trigger_dos_b=campaign_config.get("trigger_dos_b", 27),
            trigger_dos_c=campaign_config.get("trigger_dos_c", 22),
        )
    else:
        dos_coverage = 0.85
        campaign_efficiency = 0.85

    # v0.36.3: Demand variability buffer
    # Point estimate of demand doesn't account for ±2σ swings. Need reserve capacity.
    demand_config = sim_config["simulation_parameters"].get("demand", {})
    season_config = demand_config.get("seasonality", {})
    variability_buffer = calculate_variability_buffer(
        seasonality_amplitude=season_config.get("amplitude", 0.12),
        noise_cv=0.10,  # Typical demand noise CV
        safety_z=cap_planning.get("variability_safety_z", 1.28),
    )

    # v0.36.3: Apply campaign efficiency
    # Efficiency represents fraction of capacity actually used due to DOS cycling.
    # Lower efficiency with multiplication gives fewer effective production hours,
    # but the variability buffer compensates by requiring more total lines.
    effective_production = capacity_constrained_production * campaign_efficiency

    # Calculate batches per day (products cycle every production_horizon_days)
    batches_per_day = num_finished_products / production_horizon_days

    # Get average changeover time from category profiles
    category_profiles = mfg.get("category_profiles", {})
    changeover_times = [
        profile.get("changeover_time_hours", 1.0)
        for profile in category_profiles.values()
    ]
    avg_changeover_time = (
        sum(changeover_times) / len(changeover_times) if changeover_times else 1.0
    )

    # Calculate run hours from effective production volume (capacity-constrained if needed)
    run_hours_needed = effective_production / avg_run_rate

    # Calculate changeover hours from batch count
    changeover_hours = batches_per_day * avg_changeover_time

    # Total operating hours = run + changeover
    operating_hours = run_hours_needed + changeover_hours

    # Calculate implied changeover overhead factor (for reporting)
    changeover_overhead_implied = operating_hours / run_hours_needed if run_hours_needed > 0 else 1.0

    # Derive required scheduled hours and lines
    # scheduled_hours = operating_hours / target_availability
    # num_lines = scheduled_hours / hours_per_day
    scheduled_hours = operating_hours / target_availability
    total_lines_needed = math.ceil(scheduled_hours / hours_per_day)

    # v0.36.3: Apply efficiency compensation and variability buffer
    # Lower campaign efficiency means we need MORE capacity to meet demand.
    # The multiplication formula gives base_lines = f(demand * efficiency).
    # To get capacity needed for full demand: base_lines / efficiency * buffer
    total_lines_needed = math.ceil(total_lines_needed / campaign_efficiency * variability_buffer)

    # v0.35.3: Get plant capabilities for accurate allocation
    plant_capabilities = {
        pid: params.get("supported_categories", [])
        for pid, params in plant_params.items()
    }

    # Allocate lines to plants based on actual product distribution
    plant_allocations = allocate_lines_to_plants(
        total_lines_needed,
        capacity_analysis,
        min_lines_per_plant,
        products_by_category,
        plant_capabilities,
    )

    # Calculate estimated OEE with derived lines
    # OEE = A × P × Q, where A = operating_hours / scheduled_hours
    scheduled_hours_derived = total_lines_needed * hours_per_day
    estimated_availability = operating_hours / scheduled_hours_derived if scheduled_hours_derived > 0 else 0
    estimated_oee = estimated_availability * avg_efficiency * quality

    # v0.35.4: Calculate realistic OEE bound for campaign batching
    # Campaign batching creates natural idle time - not all products produce every day
    # because DOS stays above trigger between production cycles.
    # The estimated_oee already incorporates campaign_batch_efficiency, so it should
    # be realistic. However, warn if target exceeds typical campaign batching limits.
    # Empirically observed: OEE ≈ 46-50% with typical campaign batching parameters.
    max_campaign_batch_oee = 0.50  # Practical upper bound for campaign batching
    realistic_oee = min(estimated_oee, max_campaign_batch_oee)
    oee_limited_by_batching = target_oee > max_campaign_batch_oee

    return {
        "target_oee": target_oee,
        "target_availability": target_availability,
        "avg_efficiency": avg_efficiency,
        "quality": quality,
        "avg_run_rate_per_hour": avg_run_rate,
        "run_hours_needed": run_hours_needed,
        "changeover_hours": changeover_hours,
        "operating_hours": operating_hours,
        "scheduled_hours": scheduled_hours,
        "total_lines_needed": total_lines_needed,
        "current_total_lines": current_lines,
        "plant_allocations": plant_allocations,
        "estimated_availability": estimated_availability,
        "estimated_oee": estimated_oee,
        # Campaign batching parameters
        "num_finished_products": num_finished_products,
        "production_horizon_days": production_horizon_days,
        "batches_per_day": batches_per_day,
        "avg_changeover_time": avg_changeover_time,
        "changeover_overhead_implied": changeover_overhead_implied,
        # Capacity constraint info
        "total_daily_demand": total_daily_demand,
        "capacity_constrained_production": capacity_constrained_production,
        "campaign_batch_efficiency": campaign_efficiency,  # v0.36.3: Now calculated dynamically
        "effective_production": effective_production,
        "capacity_constrained": capacity_constrained,
        "current_capacity": current_capacity,
        # Realistic OEE for campaign batching
        "realistic_oee": realistic_oee,
        "oee_limited_by_batching": oee_limited_by_batching,
        # v0.36.3: Decomposed efficiency factors
        "dos_coverage_factor": dos_coverage,
        "variability_buffer": variability_buffer,
    }


def simulate_product_plant_assignments(
    products_by_category: dict[str, list[str]],
    plant_capabilities: dict[str, list[str]],
) -> dict[str, int]:
    """
    Simulate MRP's round-robin plant selection to determine product counts per plant.

    This mirrors the logic in MRPEngine._select_plant() which uses per-category
    counters to distribute products evenly across eligible plants.

    Args:
        products_by_category: Dict mapping category name to list of product IDs
        plant_capabilities: Dict mapping plant_id to list of supported categories

    Returns:
        Dict mapping plant_id to number of products assigned
    """
    plant_product_counts: dict[str, int] = {pid: 0 for pid in plant_capabilities}
    category_counters: dict[str, int] = {}

    for category, products in products_by_category.items():
        if category == "INGREDIENT":
            continue

        # Find eligible plants for this category
        eligible_plants = [
            pid for pid, caps in plant_capabilities.items()
            if category in caps
        ]

        if not eligible_plants:
            continue

        # Initialize counter for this category
        if category not in category_counters:
            category_counters[category] = 0

        # Round-robin assign products (mirrors MRP logic)
        for _ in products:
            plant_idx = category_counters[category] % len(eligible_plants)
            plant_id = eligible_plants[plant_idx]
            plant_product_counts[plant_id] += 1
            category_counters[category] += 1

    return plant_product_counts


def allocate_lines_to_plants(
    total_lines: int,
    capacity_analysis: dict[str, Any],
    min_lines: int = 2,
    products_by_category: dict[str, list[str]] | None = None,
    plant_capabilities: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """
    Allocate lines proportionally to plants based on actual product assignments.

    v0.35.3: Now simulates MRP's round-robin product distribution to allocate
    lines proportionally to actual workload, not theoretical capacity.

    Args:
        total_lines: Total number of lines to allocate
        capacity_analysis: Output from calculate_plant_capacity()
        min_lines: Minimum lines per plant
        products_by_category: Product lists per category (for accurate allocation)
        plant_capabilities: Plant-to-category mapping (for accurate allocation)

    Returns:
        Dict mapping plant_id to allocated num_lines
    """
    plant_caps = capacity_analysis["plant_capacities"]
    num_plants = len(plant_caps)

    # Ensure we have enough lines for minimums
    if total_lines < num_plants * min_lines:
        total_lines = num_plants * min_lines

    # v0.35.3: Use actual product distribution if available
    if products_by_category and plant_capabilities:
        # Simulate MRP's round-robin to get actual product counts per plant
        product_counts = simulate_product_plant_assignments(
            products_by_category, plant_capabilities
        )
        weights = {pid: float(count) for pid, count in product_counts.items()}
    else:
        # Fallback to theoretical capacity
        weights = {pid: p["theoretical_capacity_total"] for pid, p in plant_caps.items()}

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0  # Avoid division by zero

    allocations: dict[str, int] = {}
    remaining = total_lines

    # Sort by weight descending to allocate to larger plants first
    sorted_plants = sorted(weights.items(), key=lambda x: -x[1])

    for i, (pid, weight) in enumerate(sorted_plants):
        if i == len(sorted_plants) - 1:
            # Last plant gets remaining (ensures sum equals total)
            allocations[pid] = max(min_lines, remaining)
        else:
            # Proportional allocation
            proportion = weight / total_weight if total_weight > 0 else 1.0 / num_plants
            lines = max(min_lines, round(total_lines * proportion))
            # Don't exceed remaining minus minimums for other plants
            plants_left = num_plants - i - 1
            max_for_this = remaining - plants_left * min_lines
            lines = min(lines, max_for_this)
            allocations[pid] = lines
            remaining -= lines

    return allocations


def derive_optimal_parameters(
    demand_analysis: dict[str, Any],
    capacity_analysis: dict[str, Any],
    sim_config: dict[str, Any],
    echelon_lead_times: dict[str, Any] | None = None,
    target_turns: float | None = None,
    target_oee: float = 0.65,
    target_service_level: float = 0.97,
) -> dict[str, Any]:
    """
    Derive optimal configuration parameters based on physics and industry benchmarks.

    v0.33.0: Multi-echelon lead time awareness and MRP signal lag.
    - Uses cumulative lead times through 4-tier network
    - Accounts for FTL consolidation delays
    - Adds MRP rolling window signal lag to trigger thresholds
    - Derives ABC priming factors from z-scores (no hardcodes)

    v0.31.0: Complete rewrite to use target inventory turns as primary driver.

    Key insight: Real FMCG companies (Colgate 4.1x, P&G 5.5x, Unilever 6.2x) run
    with 52-89 days of network inventory to achieve 97%+ service. The previous
    OEE-driven approach resulted in too-lean inventory (17x turns) causing
    service level degradation.

    Physics relationships:
    - Network DOS = 365 / Target Turns
    - Network DOS = Store DOS + DC DOS + RDC DOS + Pipeline
    - Each echelon DOS = Lead Time + Safety Stock + Cycle Stock + Strategic Buffer
    - Trigger DOS = Replenishment Time + Safety Buffer + Review Period + MRP Signal Lag
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

    # v0.33.0: Load multi-echelon and MRP signal lag config
    multi_echelon_cfg = calibration.get("multi_echelon_lead_times", {})
    mrp_signal_cfg = calibration.get("mrp_signal_lag", {})
    cold_start_cfg = calibration.get("cold_start", {})

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

    # =================================================================
    # v0.33.0: USE MULTI-ECHELON LEAD TIMES (Major Fix)
    # =================================================================
    # Previously: Single lead_time_days = 3.0 used everywhere
    # Now: Cumulative lead times by echelon from actual network analysis
    # =================================================================
    if echelon_lead_times and multi_echelon_cfg.get("use_dynamic_calculation", True):
        # Use dynamically calculated lead times from links.csv
        cumulative = echelon_lead_times.get("cumulative", {})
        store_lt = cumulative.get("store", 1.0)
        dc_lt = cumulative.get("customer_dc", 3.0)
        rdc_lt = cumulative.get("rdc", 5.0)
        network_replenishment_time = echelon_lead_times.get("network_replenishment_time", 7.0)
        ftl_consolidation = echelon_lead_times.get("ftl_consolidation", 2.0)
    else:
        # Use static config values (fallback)
        store_lt = multi_echelon_cfg.get("store_from_dc", 1.0)
        dc_lt = multi_echelon_cfg.get("customer_dc_from_rdc", 3.0)
        rdc_lt = multi_echelon_cfg.get("rdc_from_plant", 5.0)
        ftl_consolidation = multi_echelon_cfg.get("ftl_consolidation_buffer", 2.0)
        prod_lt_cfg = multi_echelon_cfg.get("plant_production", 3.0)
        network_replenishment_time = rdc_lt + prod_lt_cfg

    # Legacy single lead time for backwards compatibility in some formulas
    lead_time_days = log_config.get("default_lead_time_days", 3.0)
    order_cycle_days = replen_config.get("order_cycle_days", 5)
    current_production_horizon = campaign_config.get("production_horizon_days", 4)

    # v0.33.0: MRP signal lag (MRP uses 14-day rolling window)
    mrp_signal_lag = mrp_signal_cfg.get("effective_lag_days", 7)
    include_mrp_lag_in_triggers = mrp_signal_cfg.get("include_in_triggers", True)

    # v0.33.0: Cold-start buffer
    cold_start_buffer_pct = cold_start_cfg.get("buffer_pct", 0.15)
    apply_cold_start = cold_start_cfg.get("apply_to_priming", True)

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
    # v0.33.0: Now uses ECHELON-SPECIFIC lead times, not single value
    # =================================================================
    # Ensure each echelon has minimum DOS to cover:
    # 1. Lead time (replenishment runway) - NOW ECHELON-SPECIFIC
    # 2. Safety stock (variability buffer)
    # 3. Cycle stock (order cycle average)
    # 4. Strategic buffer (presentation stock, forward positioning)
    # =================================================================

    # Strategic buffers from config
    presentation_stock = buffers.get("presentation_stock_days", 3.0)
    forward_positioning = buffers.get("forward_positioning_days", 5.0)
    production_smoothing = buffers.get("production_smoothing_days", 5.0)

    # Store minimum: Lead time + Safety + Cycle + Presentation
    # v0.33.0: Uses store-specific lead time (from DC)
    # Safety = z × √(demand_var² × LT + supply_var² × demand²)
    # Simplified: z × combined_cv × √LT
    combined_cv_store = (demand_cv.get("store", 0.5)**2 + supply_cv.get("store", 0.3)**2)**0.5
    store_safety_b = z_scores.get("B", 2.0) * combined_cv_store * (store_lt ** 0.5)
    store_cycle = order_cycle_days / 2
    store_min = store_lt + store_safety_b + store_cycle + presentation_stock

    # DC minimum: Lead time + Safety + Cycle + Forward positioning
    # v0.33.0: Uses DC-specific lead time (from RDC, includes FTL delay)
    combined_cv_dc = (demand_cv.get("customer_dc", 0.3)**2 + supply_cv.get("customer_dc", 0.2)**2)**0.5
    dc_safety_b = z_scores.get("B", 2.0) * combined_cv_dc * (dc_lt ** 0.5)
    dc_cycle = order_cycle_days / 2
    dc_min = dc_lt + dc_safety_b + dc_cycle + forward_positioning

    # RDC minimum: Lead time + Safety + Production cycle + Smoothing buffer
    # v0.33.0: Uses RDC-specific lead time (from Plant, includes FTL delay)
    combined_cv_rdc = (demand_cv.get("rdc", 0.2)**2 + supply_cv.get("rdc", 0.15)**2)**0.5
    rdc_safety_b = z_scores.get("B", 2.0) * combined_cv_rdc * (rdc_lt ** 0.5)
    # Production horizon determines RDC cycle stock
    production_horizon = max(current_production_horizon, 7)  # Recommend at least 7 days
    rdc_cycle = production_horizon / 2
    rdc_min = rdc_lt + rdc_safety_b + rdc_cycle + production_smoothing

    # Apply minimums (ensure physics requirements are met)
    store_dos = max(store_dos_base, store_min)
    dc_dos = max(dc_dos_base, dc_min)
    rdc_dos = max(rdc_dos_base, rdc_min)

    # Recalculate actual network DOS after applying minimums
    actual_network_dos = store_dos + dc_dos + rdc_dos + plant_dos + pipeline_dos
    actual_turns = 365.0 / actual_network_dos

    # =================================================================
    # STEP 4: ABC-DIFFERENTIATED DOS (Service level stratification)
    # v0.33.0: ABC priming factors now DERIVED from z-scores, not hardcoded
    # =================================================================
    # A-items: Higher service target (99%) → MORE buffer (premium service)
    # B-items: Standard service (97%) → base buffer
    # C-items: Lower service (95%) → SLIGHTLY less buffer (SLOB control)
    #
    # CRITICAL: A-items must have MORE inventory than B-items, not less!
    # The z-score affects safety stock calculation, but for priming factors
    # we want A > B > C to ensure service level hierarchy.
    # =================================================================
    # v0.33.0: Use echelon-specific lead times for ABC safety stock
    store_safety_a = z_scores.get("A", 2.33) * combined_cv_store * (store_lt ** 0.5)
    store_safety_c = z_scores.get("C", 1.65) * combined_cv_store * (store_lt ** 0.5)

    store_dos_a = round(store_lt + store_safety_a + store_cycle + presentation_stock, 1)
    store_dos_b = round(store_dos, 1)
    store_dos_c = round(store_lt + store_safety_c + store_cycle + presentation_stock * 0.5, 1)

    # =================================================================
    # v0.33.0: DERIVE ABC priming factors from z-scores (no hardcodes!)
    # =================================================================
    # Formula: factor_X = z_X / z_B (B-items are baseline)
    # This ensures the hierarchy A > B > C is mathematically guaranteed
    # as long as z_A > z_B > z_C in config.
    # =================================================================
    abc_priming_factors = derive_abc_priming_factors(z_scores)

    # v0.33.0: Apply cold-start buffer if enabled
    if apply_cold_start:
        # Boost all priming factors to compensate for Day 1-30 stabilization
        abc_priming_factors = {
            k: round(v * (1.0 + cold_start_buffer_pct), 3)
            for k, v in abc_priming_factors.items()
        }

    # =================================================================
    # STEP 5: TRIGGER THRESHOLD DERIVATION (Network DOS triggers)
    # v0.33.0: Now includes MRP signal lag and multi-echelon replenishment time
    # =================================================================
    # Trigger = when to start production for a SKU
    # Must cover: Replenishment Time + Safety + Review Period + MRP Signal Lag
    #
    # v0.33.0 Changes:
    # - Replenishment Time = NETWORK replenishment time (multi-echelon)
    # - Added MRP Signal Lag (MRP uses 14-day rolling window → 7-day effective lag)
    # =================================================================
    prod_lead_time = trigger_cfg.get("production_lead_time_days", 3)
    transit_time = trigger_cfg.get("transit_time_days", 3)

    # v0.33.0: Use network replenishment time from multi-echelon analysis
    # This is the time from Plant production to RDC availability
    replenishment_time = max(
        prod_lead_time + transit_time,  # Config minimum
        network_replenishment_time,      # Actual network cascade
    )

    safety_a = trigger_cfg.get("safety_buffer_a", 10)
    safety_b = trigger_cfg.get("safety_buffer_b", 6)
    safety_c = trigger_cfg.get("safety_buffer_c", 3)

    review_a = trigger_cfg.get("review_period_a", 5)
    review_b = trigger_cfg.get("review_period_b", 5)
    review_c = trigger_cfg.get("review_period_c", 3)

    # v0.33.0: Add MRP signal lag to trigger thresholds
    # MRP uses 14-day rolling window → production responds to demand with ~7-day lag
    # Without this buffer, triggers fire too late and inventory runs out
    mrp_lag_component = mrp_signal_lag if include_mrp_lag_in_triggers else 0

    trigger_a = int(replenishment_time + safety_a + review_a + mrp_lag_component)
    trigger_b = int(replenishment_time + safety_b + review_b + mrp_lag_component)
    trigger_c = int(replenishment_time + safety_c + review_c + mrp_lag_component)

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
        # v0.33.0: Multi-echelon lead time analysis
        "lead_time_analysis": {
            "store_lt": round(store_lt, 2),
            "dc_lt": round(dc_lt, 2),
            "rdc_lt": round(rdc_lt, 2),
            "network_replenishment_time": round(network_replenishment_time, 2),
            "ftl_consolidation": round(ftl_consolidation, 2),
            "mrp_signal_lag": mrp_signal_lag,
            "cold_start_buffer_pct": cold_start_buffer_pct,
            "legacy_single_lt": lead_time_days,
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
            "replenishment_time": round(replenishment_time, 1),
            "mrp_signal_lag_included": include_mrp_lag_in_triggers,
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
    print("     PRISM SIM CONFIGURATION CALIBRATION REPORT (v0.33.0)")
    print("     Multi-Echelon Lead Time & MRP Signal Lag Awareness")
    print("=" * 70)

    # v0.33.0: Multi-echelon lead time analysis (NEW)
    lt_analysis = recommendations.get("lead_time_analysis", {})
    if lt_analysis:
        print("\n--- MULTI-ECHELON LEAD TIME ANALYSIS (v0.33.0 NEW) ---")
        print("Echelon-Specific Lead Times (P90 from links.csv):")
        print(f"  Store ← DC:     {lt_analysis.get('store_lt', 1.0):5.2f} days")
        print(f"  DC ← RDC:       {lt_analysis.get('dc_lt', 3.0):5.2f} days (incl FTL)")
        print(f"  RDC ← Plant:    {lt_analysis.get('rdc_lt', 5.0):5.2f} days (incl FTL)")
        print(f"  FTL Buffer:     {lt_analysis.get('ftl_consolidation', 2.0):5.2f} days")
        print(f"\nNetwork Replenishment Time: {lt_analysis.get('network_replenishment_time', 7.0):.1f} days")
        print(f"MRP Signal Lag:             {lt_analysis.get('mrp_signal_lag', 7)} days")
        print(f"Cold-Start Buffer:          {lt_analysis.get('cold_start_buffer_pct', 0.15):.0%}")
        print(f"\nLegacy Single Lead Time:    {lt_analysis.get('legacy_single_lt', 3.0)} days")
        gap = lt_analysis.get('network_replenishment_time', 7.0) - lt_analysis.get('legacy_single_lt', 3.0)
        if gap > 2:
            print(f"  WARNING: Network cascade is {gap:.1f}d longer than legacy assumption!")

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
    print("\nDaily Demand by Category (theoretical):")
    for cat, demand in demand_analysis["category_demand"].items():
        print(f"  {cat}: {demand:,.0f} cases/day")

    theoretical = demand_analysis.get("theoretical_daily_demand", demand_analysis["total_daily_demand"])
    realism_factor = demand_analysis.get("demand_realism_factor", 1.0)
    effective = demand_analysis["total_daily_demand"]
    print(f"\nTheoretical Demand: {theoretical:,.0f} cases/day")
    print(f"Realism Factor:     {realism_factor:.0%} (segment weights, Zipf, format scales)")
    print(f"EFFECTIVE DEMAND:   {effective:,.0f} cases/day")

    print("\n--- CAPACITY ANALYSIS ---")
    print(f"Total Theoretical Capacity: {capacity_analysis['total_theoretical_capacity']:,.0f} cases/day")
    print(f"Production Rate Multiplier: {capacity_analysis['current_multiplier']}x")
    print(f"Effective Capacity: {capacity_analysis['total_with_multiplier']:,.0f} cases/day")
    util = analysis["capacity_utilization"]
    print(f"Capacity Utilization: {util:.1%}")

    print("Plant Breakdown:")
    for pid, caps in capacity_analysis["plant_capacities"].items():
        print(f"  {pid}: {caps['num_lines']} lines, {caps['theoretical_capacity_total']:,.0f} cap")

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

    print("\n--- TRIGGER THRESHOLD ANALYSIS (v0.33.0 Enhanced) ---")
    trig = recommendations["trigger_analysis"]
    print(f"Network Replenishment Time: {trig.get('replenishment_time', 6)} days")
    mrp_lag_included = trig.get('mrp_signal_lag_included', False)
    print(f"MRP Signal Lag Included:    {mrp_lag_included}")
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

    # Capacity planning section (when --derive-lines is used)
    cap_plan = recommendations.get("capacity_planning")
    if cap_plan:
        print("\n" + "=" * 70)
        print("     CAPACITY PLANNING (--derive-lines)")
        print("=" * 70)
        print(f"\nTarget OEE:             {cap_plan['target_oee']:.1%}")
        print(f"Required Availability:  {cap_plan['target_availability']:.1%}")
        print(f"Avg Efficiency (P):     {cap_plan['avg_efficiency']:.1%}")
        print(f"Quality (Q):            {cap_plan['quality']:.1%}")
        print(f"\nCampaign Batching Parameters:")
        print(f"  Finished products:    {cap_plan['num_finished_products']}")
        print(f"  Production horizon:   {cap_plan['production_horizon_days']} days")
        print(f"  Batches per day:      {cap_plan['batches_per_day']:.1f}")
        print(f"  Avg changeover time:  {cap_plan['avg_changeover_time']:.2f} hrs")
        if cap_plan.get('capacity_constrained'):
            print(f"\n  *** CAPACITY CONSTRAINED ***")
            print(f"  Demand:               {cap_plan['total_daily_demand']:,.0f} cases/day")
            print(f"  Capacity:             {cap_plan['current_capacity']:,.0f} cases/day")
            print(f"  Cap-constrained prod: {cap_plan['capacity_constrained_production']:,.0f} cases/day")
        print(f"\n  Campaign batch eff:   {cap_plan['campaign_batch_efficiency']:.0%}")
        print(f"  Effective production: {cap_plan['effective_production']:,.0f} cases/day")
        print(f"\nDerived Capacity Calculation:")
        print(f"  Avg run rate:         {cap_plan['avg_run_rate_per_hour']:,.0f} cases/hr/line")
        print(f"  Run hours needed:     {cap_plan['run_hours_needed']:,.1f} hrs")
        print(f"  Changeover hours:     {cap_plan['changeover_hours']:,.1f} hrs")
        print(f"  Operating hours:      {cap_plan['operating_hours']:,.1f} hrs (run + changeover)")
        print(f"  Changeover overhead:  {cap_plan['changeover_overhead_implied']:.0%} (implied)")
        print(f"  Scheduled hours:      {cap_plan['scheduled_hours']:,.1f} hrs")
        print(f"\nLine Count:")
        print(f"  Current total lines:  {cap_plan['current_total_lines']}")
        print(f"  Derived total lines:  {cap_plan['total_lines_needed']}")
        print(f"  Estimated OEE:        {cap_plan['estimated_oee']:.1%}")
        print(f"  Realistic OEE:        {cap_plan['realistic_oee']:.1%} (campaign batching limit)")
        if cap_plan.get('oee_limited_by_batching'):
            print(f"\n  *** WARNING: Target OEE {cap_plan['target_oee']:.0%} exceeds campaign batching limit ***")
            print(f"  Campaign batching creates natural idle time (DOS cycling).")
            print(f"  Max achievable OEE with current settings: ~46-50%")
            print(f"  To increase OEE: reduce production_horizon_days, increase triggers,")
        print(f"\nPer-Plant Allocation:")
        for pid, lines in sorted(cap_plan["plant_allocations"].items()):
            # Get current lines for comparison
            current = capacity_analysis["plant_capacities"].get(pid, {}).get("num_lines", 0)
            diff = lines - current
            diff_str = f"({'+' if diff >= 0 else ''}{diff})" if diff != 0 else "(no change)"
            print(f"  {pid}: {lines} lines {diff_str}")

        # v0.36.3: Decomposed efficiency factors
        print(f"\nEfficiency Decomposition (v0.36.3):")
        print(f"  DOS coverage factor:    {cap_plan.get('dos_coverage_factor', 'N/A'):.1%}")
        print(f"  Variability buffer:     {cap_plan.get('variability_buffer', 1.0):.2f}x")

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

    # Update num_lines from capacity planning (when --derive-lines is used)
    cap_plan = recommendations.get("capacity_planning")
    if cap_plan:
        plant_params = mfg.setdefault("plant_parameters", {})
        for pid, lines in cap_plan["plant_allocations"].items():
            if pid not in plant_params:
                plant_params[pid] = {}
            plant_params[pid]["num_lines"] = lines

    save_json(sim_config_path, config)
    print(f"\n✓ Applied recommendations to {sim_config_path}")
    print("\nKey changes applied:")
    print(f"  - Priming DOS: store={rec['store_days_supply']}d, dc={rec['customer_dc_days_supply']}d, rdc={rec['rdc_days_supply']}d")
    print(f"  - Triggers: A={rec['trigger_dos_a']}d, B={rec['trigger_dos_b']}d, C={rec['trigger_dos_c']}d")
    print(f"  - Production horizon: {rec['production_horizon_days']}d")
    print(f"  - C-item factors: prod={rec.get('c_production_factor', 0.4)}, demand={rec.get('c_demand_factor', 0.7)}")
    print(f"  - SLOB thresholds: {rec['slob_abc_thresholds']}")

    # Print capacity planning changes if applied
    if cap_plan:
        print("\nCapacity planning (--derive-lines) applied:")
        print(f"  - Target OEE: {cap_plan['target_oee']:.0%}")
        print(f"  - Total lines: {cap_plan['current_total_lines']} → {cap_plan['total_lines_needed']}")
        print(f"  - Per-plant: {cap_plan['plant_allocations']}")


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
    parser.add_argument(
        "--target-oee",
        type=float,
        default=None,
        help="Target OEE (e.g., 0.60 for 60%%). Derives num_lines when --derive-lines is set.",
    )
    parser.add_argument(
        "--derive-lines",
        action="store_true",
        help="Derive num_lines from --target-oee instead of reading from config",
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

    # v0.33.0: Calculate multi-echelon lead times from actual network topology
    links_path = static_world_dir / "links.csv"
    echelon_lead_times = None
    if links_path.exists():
        echelon_lead_times = calculate_multi_echelon_lead_times(links_path, sim_config)
    else:
        print("WARNING: links.csv not found - using static lead time config")

    # Analyze
    demand_analysis = calculate_daily_demand(
        world_config, sim_config, node_counts, products_by_category
    )
    capacity_analysis = calculate_plant_capacity(
        sim_config, recipe_rates, products_by_category
    )

    # v0.33.0: Pass multi-echelon lead times to derive function
    recommendations = derive_optimal_parameters(
        demand_analysis,
        capacity_analysis,
        sim_config,
        echelon_lead_times=echelon_lead_times,
        target_turns=args.target_turns,  # None means use config default
        target_service_level=args.target_service,
    )

    # Add seasonal capacity analysis
    recommendations["seasonal_capacity"] = derive_seasonal_capacity_params(
        sim_config, recommendations
    )

    # Capacity planning: derive num_lines if requested
    if args.derive_lines:
        target_oee = args.target_oee if args.target_oee is not None else 0.60
        try:
            capacity_planning_result = derive_num_lines_from_oee(
                target_oee,
                demand_analysis["total_daily_demand"],
                capacity_analysis,
                sim_config,
                products_by_category,  # v0.35.3: For accurate plant-workload allocation
            )
            recommendations["capacity_planning"] = capacity_planning_result
        except ValueError as e:
            print(f"\nERROR: {e}")
            return

    # Validate config consistency
    violations = validate_config_consistency(sim_config, recommendations)

    # Add seasonal balance warnings
    seasonal_warnings = validate_seasonal_balance(sim_config, recommendations)
    violations.extend(seasonal_warnings)

    # v0.33.0: Validate against baseline reference values
    baseline_warnings = validate_against_baseline(recommendations, sim_config)
    violations.extend(baseline_warnings)

    # Report
    print_report(demand_analysis, capacity_analysis, recommendations, violations)

    # Apply if requested
    if args.apply:
        apply_recommendations(sim_config_path, recommendations)
    else:
        print("\nRun with --apply to update simulation_config.json")


if __name__ == "__main__":
    main()
