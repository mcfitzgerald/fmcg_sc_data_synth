"""
Layer 3: End-to-End Flow & Stability Analysis.

3.1 End-to-End Throughput Map — route-type flows + ASCII diagram
3.2 Deployment Effectiveness — plant-to-RDC/DC split, retention
3.3 Lead Time Analysis — actual vs configured by route type
3.4 Bullwhip Measurement — order variance amplification per echelon
3.5 Control System Stability — convergence assessment
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .loader import (
    ECHELON_ORDER,
    DataBundle,
    classify_node,
    is_demand_endpoint,
    is_finished_good,
)

# Thresholds
_BULLWHIP_THRESH = 2.0
_STABILITY_SLOPE_ZERO = 0.0001
_STABILITY_SLOPE_SLOW = 0.001
_INV_SLOPE_STABLE = 0.01
_INV_SLOPE_SLOW = 0.05
_TREND_MIN_SAMPLES = 4
_TREND_STEP_DIVISOR = 6
_LARGE_VALUE_THRESH = 1000
_MIN_TREND_POINTS = 2
_STABILITY_WINDOW_DAYS = 180
_MIN_REGRESSION_POINTS = 10
_MIN_ECHELON_POINTS = 5
_MILLION = 1_000_000
_THOUSAND = 1_000

# Route type classification
ROUTE_TYPES = [
    ("Supplier -> Plant", "Supplier", "Plant"),
    ("Plant -> RDC", "Plant", "RDC"),
    ("Plant -> Customer DC", "Plant", "Customer DC"),
    ("RDC -> Customer DC", "RDC", "Customer DC"),
    ("Customer DC -> Store", "Customer DC", "Store"),
    ("Customer DC -> Club", "Customer DC", "Club"),
    ("RDC -> Store", "RDC", "Store"),
]


def _classify_route(source_ech: str, target_ech: str) -> str:
    """Map source/target echelons to a named route type."""
    for name, src, tgt in ROUTE_TYPES:
        if source_ech == src and target_ech == tgt:
            return name
    return f"{source_ech} -> {target_ech}"


# ---------------------------------------------------------------------------
# 3.1 End-to-End Throughput Map
# ---------------------------------------------------------------------------

def analyze_throughput_map(data: DataBundle) -> dict[str, Any]:
    """Classify shipments by route type, compute daily volumes.

    Returns:
        dict with 'routes' (per-route stats), 'flow_data' (for ASCII diagram).
    """
    ships = data.shipments.copy()
    ships["source_echelon"] = ships["source_id"].map(classify_node)
    ships["target_echelon"] = ships["target_id"].map(classify_node)
    ships["route_type"] = ships.apply(
        lambda r: _classify_route(r["source_echelon"], r["target_echelon"]), axis=1
    )

    sim_days = data.sim_days
    fg_ships = ships[ships["product_id"].apply(is_finished_good)]

    routes: dict[str, dict[str, Any]] = {}
    for route_type in fg_ships["route_type"].unique():
        route_ships = fg_ships[fg_ships["route_type"] == route_type]
        total_qty = route_ships["quantity"].sum()
        daily_qty = total_qty / sim_days if sim_days > 0 else 0
        n_shipments = len(route_ships)
        routes[route_type] = {
            "total_quantity": total_qty,
            "daily_quantity": daily_qty,
            "n_shipments": n_shipments,
            "daily_shipments": n_shipments / sim_days if sim_days > 0 else 0,
        }

    # Flow data for ASCII diagram
    fg_batches = data.batches[data.batches["product_id"].apply(is_finished_good)]
    production_daily = fg_batches["quantity"].sum() / sim_days if sim_days > 0 else 0

    demand_ships = fg_ships[fg_ships["target_id"].apply(is_demand_endpoint)]
    pos_daily = demand_ships["quantity"].sum() / sim_days if sim_days > 0 else 0

    # Inventory at end of sim
    inv = data.inv_by_echelon
    max_day = int(inv["day"].max()) if len(inv) > 0 else 0
    latest_inv = inv[inv["day"] == max_day] if max_day > 0 else inv.head(0)

    inv_by_ech: dict[str, float] = {}
    for ech in ECHELON_ORDER:
        ech_data = latest_inv[latest_inv["echelon"] == ech]
        inv_by_ech[ech] = float(ech_data["total"].sum()) if len(ech_data) > 0 else 0

    flow_data = {
        "production_daily": production_daily,
        "pos_daily": pos_daily,
        "prod_demand_ratio": (
            production_daily / pos_daily if pos_daily > 0 else np.nan
        ),
        "inv_by_echelon": inv_by_ech,
        "routes": routes,
    }

    return {"routes": routes, "flow_data": flow_data}


# ---------------------------------------------------------------------------
# 3.2 Deployment Effectiveness
# ---------------------------------------------------------------------------

def analyze_deployment_effectiveness(data: DataBundle) -> dict[str, Any]:
    """Analyze plant deployment split (RDC vs direct DC), retention.

    Returns:
        dict with 'split', 'plant_inv_trend', 'retention'.
    """
    fg_ships = data.shipments[
        data.shipments["product_id"].apply(is_finished_good)
    ].copy()
    fg_ships["source_echelon"] = fg_ships["source_id"].map(classify_node)
    fg_ships["target_echelon"] = fg_ships["target_id"].map(classify_node)

    plant_ships = fg_ships[fg_ships["source_echelon"] == "Plant"]
    to_rdc = plant_ships[plant_ships["target_echelon"] == "RDC"]["quantity"].sum()
    to_dc = plant_ships[plant_ships["target_echelon"] == "Customer DC"][
        "quantity"
    ].sum()
    total_deployed = to_rdc + to_dc

    split = {
        "to_rdc": to_rdc,
        "to_dc": to_dc,
        "total_deployed": total_deployed,
        "rdc_pct": to_rdc / total_deployed * 100 if total_deployed > 0 else 0,
        "dc_pct": to_dc / total_deployed * 100 if total_deployed > 0 else 0,
    }

    # Plant FG inventory trend
    inv = data.inv_by_echelon
    plant_inv = inv[inv["echelon"] == "Plant"].sort_values("day")
    plant_trend: list[dict[str, Any]] = []
    if len(plant_inv) > 0:
        days = sorted(plant_inv["day"].unique())
        for d in days:
            day_data = plant_inv[plant_inv["day"] == d]
            plant_trend.append({
                "day": int(d),
                "inventory": float(day_data["total"].sum()),
            })

    # Plant FG stability
    if len(plant_trend) >= _MIN_TREND_POINTS:
        first_inv = plant_trend[0]["inventory"]
        last_inv = plant_trend[-1]["inventory"]
        growth_pct = (
            (last_inv - first_inv) / first_inv * 100 if first_inv > 0 else np.nan
        )
    else:
        growth_pct = np.nan

    # Retention rate: production vs deployed
    fg_batches = data.batches[data.batches["product_id"].apply(is_finished_good)]
    total_production = fg_batches["quantity"].sum()
    retention_pct = (
        (1 - total_deployed / total_production) * 100
        if total_production > 0
        else 0
    )

    return {
        "split": split,
        "plant_inv_trend": plant_trend,
        "plant_inv_growth_pct": growth_pct,
        "retention_pct": retention_pct,
        "total_production": total_production,
    }


# ---------------------------------------------------------------------------
# 3.3 Lead Time Analysis
# ---------------------------------------------------------------------------

def analyze_lead_times(data: DataBundle) -> dict[str, Any]:
    """Actual vs configured lead times by route type.

    Returns:
        dict with 'by_route' (p50/p90/p99, configured LT).
    """
    ships = data.shipments[
        data.shipments["product_id"].apply(is_finished_good)
    ].copy()
    ships["source_echelon"] = ships["source_id"].map(classify_node)
    ships["target_echelon"] = ships["target_id"].map(classify_node)
    ships["route_type"] = ships.apply(
        lambda r: _classify_route(r["source_echelon"], r["target_echelon"]), axis=1
    )
    ships["actual_lt"] = ships["arrival_day"] - ships["creation_day"]

    # Configured LTs from links.csv
    links = data.links.copy()
    links["source_echelon"] = links["source_id"].map(classify_node)
    links["target_echelon"] = links["target_id"].map(classify_node)
    links["route_type"] = links.apply(
        lambda r: _classify_route(r["source_echelon"], r["target_echelon"]), axis=1
    )
    config_lt = links.groupby("route_type")["lead_time_days"].mean()

    by_route: dict[str, dict[str, Any]] = {}
    for route_type in sorted(ships["route_type"].unique()):
        route_ships = ships[ships["route_type"] == route_type]
        lts = route_ships["actual_lt"].dropna()
        if len(lts) == 0:
            continue
        by_route[route_type] = {
            "n_shipments": len(lts),
            "p50": float(np.percentile(lts, 50)),
            "p90": float(np.percentile(lts, 90)),
            "p99": float(np.percentile(lts, 99)),
            "mean": float(lts.mean()),
            "configured_lt": float(config_lt.get(route_type, np.nan)),
        }

    return {"by_route": by_route}


# ---------------------------------------------------------------------------
# 3.4 Bullwhip Measurement
# ---------------------------------------------------------------------------

def analyze_bullwhip(data: DataBundle) -> dict[str, Any]:
    """Order variance amplification ratio at each echelon.

    Bullwhip = order_variance[echelon] / pos_variance.
    """
    fg_ships = data.shipments[
        data.shipments["product_id"].apply(is_finished_good)
    ].copy()
    fg_ships["source_echelon"] = fg_ships["source_id"].map(classify_node)
    fg_ships["target_echelon"] = fg_ships["target_id"].map(classify_node)

    # POS demand variance (baseline)
    demand_ships = fg_ships[fg_ships["target_id"].apply(is_demand_endpoint)]
    pos_daily = demand_ships.groupby("creation_day")["quantity"].sum()
    pos_var = float(pos_daily.var()) if len(pos_daily) > 1 else 1.0
    pos_mean = float(pos_daily.mean())
    pos_cv = float(pos_daily.std() / pos_mean) if pos_mean > 0 else 0

    # Order variance per echelon (using orders.parquet)
    orders = data.orders[data.orders["product_id"].apply(is_finished_good)].copy()
    orders["source_echelon"] = orders["source_id"].map(classify_node)

    echelon_results: dict[str, dict[str, Any]] = {}
    for ech in ECHELON_ORDER:
        # Orders placed BY this echelon
        ech_orders = orders[orders["source_echelon"] == ech]
        if len(ech_orders) == 0:
            continue
        ech_daily = ech_orders.groupby("day")["quantity"].sum()
        ech_var = float(ech_daily.var()) if len(ech_daily) > 1 else 0
        ech_mean = float(ech_daily.mean())
        ech_cv = float(ech_daily.std() / ech_mean) if ech_mean > 0 else 0
        bullwhip = ech_var / pos_var if pos_var > 0 else np.nan

        echelon_results[ech] = {
            "daily_mean": ech_mean,
            "daily_var": ech_var,
            "cv": ech_cv,
            "bullwhip_ratio": bullwhip,
        }

    # Overall bullwhip (max ratio across echelons)
    ratios = [
        r["bullwhip_ratio"]
        for r in echelon_results.values()
        if not np.isnan(r.get("bullwhip_ratio", np.nan))
    ]
    max_bullwhip = max(ratios) if ratios else np.nan

    return {
        "pos_variance": pos_var,
        "pos_mean": pos_mean,
        "pos_cv": pos_cv,
        "echelons": echelon_results,
        "max_bullwhip": max_bullwhip,
    }


# ---------------------------------------------------------------------------
# 3.5 Control System Stability Assessment
# ---------------------------------------------------------------------------

def analyze_control_stability(
    data: DataBundle, window: int = 30
) -> dict[str, Any]:
    """Assess whether key indicators are converging, stable, or diverging.

    Uses linear regression slope over last 180 days on:
    1. prod/demand ratio
    2. inventory growth rate per echelon
    3. DOS per echelon
    """
    fg_batches = data.batches[data.batches["product_id"].apply(is_finished_good)]
    demand_ships = data.shipments[
        data.shipments["target_id"].apply(is_demand_endpoint)
        & data.shipments["product_id"].apply(is_finished_good)
    ]
    inv = data.inv_by_echelon

    demand_daily = demand_ships.groupby("creation_day")["quantity"].sum()
    prod_daily = fg_batches.groupby("day_produced")["quantity"].sum()
    all_days = sorted(set(demand_daily.index) | set(prod_daily.index))
    demand_s = demand_daily.reindex(all_days, fill_value=0)
    prod_s = prod_daily.reindex(all_days, fill_value=0)

    demand_roll = demand_s.rolling(window, min_periods=1).mean()
    prod_roll = prod_s.rolling(window, min_periods=1).mean()
    ratio_roll = prod_roll / demand_roll.replace(0, np.nan)

    # Use last 180 days for stability assessment
    last_180_start = (
        max(all_days) - _STABILITY_WINDOW_DAYS
        if max(all_days) > _STABILITY_WINDOW_DAYS
        else 0
    )
    indicators: dict[str, dict[str, Any]] = {}

    # 1. Prod/demand ratio trend
    ratio_last180 = ratio_roll[ratio_roll.index >= last_180_start].dropna()
    if len(ratio_last180) > _MIN_REGRESSION_POINTS:
        x = np.arange(len(ratio_last180), dtype=float)
        slope, _intercept = np.polyfit(x, ratio_last180.values, 1)
        last_val = float(ratio_last180.iloc[-1])
        indicators["prod_demand_ratio"] = {
            "last_value": last_val,
            "target": 1.0,
            "slope_per_day": float(slope),
            "verdict": _classify_stability(slope, last_val, 1.0),
        }

    # 2. Inventory growth rate per echelon
    if len(inv) > 0:
        for ech in ECHELON_ORDER:
            ech_inv = inv[inv["echelon"] == ech]
            if len(ech_inv) == 0:
                continue
            daily_inv = ech_inv.groupby("day")["total"].sum()
            # Growth rate = day-over-day change (rolling)
            last_days = daily_inv[daily_inv.index >= last_180_start]
            if len(last_days) > _MIN_ECHELON_POINTS:
                x = np.arange(len(last_days), dtype=float)
                slope, _intercept = np.polyfit(x, last_days.values, 1)
                mean_inv = float(last_days.mean())
                pct_slope = slope / mean_inv * 100 if mean_inv > 0 else 0
                indicators[f"{ech}_inventory"] = {
                    "last_value": float(last_days.iloc[-1]),
                    "target": "stable",
                    "slope_per_day": float(slope),
                    "slope_pct_per_day": float(pct_slope),
                    "verdict": _classify_inv_stability(pct_slope),
                }

    # Overall system verdict
    verdicts = [ind["verdict"] for ind in indicators.values()]
    if all(v in ("STABLE", "CONVERGING") for v in verdicts):
        overall = "STABLE"
    elif any(v == "DIVERGING" for v in verdicts):
        overall = "DIVERGING"
    elif any(v == "OSCILLATING" for v in verdicts):
        overall = "OSCILLATING"
    else:
        overall = "MIXED"

    return {"indicators": indicators, "overall_verdict": overall}


def _classify_stability(slope: float, current: float, target: float) -> str:
    """Classify trend toward/away from target."""
    if abs(slope) < _STABILITY_SLOPE_ZERO:
        return "STABLE"
    distance = current - target
    # If slope moves current toward target
    if distance > 0 and slope < 0:
        return "CONVERGING"
    if distance < 0 and slope > 0:
        return "CONVERGING"
    # If slope moves current away from target
    if abs(slope) > _STABILITY_SLOPE_SLOW:
        return "DIVERGING"
    return "STABLE"


def _classify_inv_stability(pct_slope: float) -> str:
    """Classify inventory trend."""
    if abs(pct_slope) < _INV_SLOPE_STABLE:
        return "STABLE"
    if abs(pct_slope) < _INV_SLOPE_SLOW:
        return "CONVERGING"  # Slow change
    return "DIVERGING" if pct_slope > 0 else "DRAINING"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_throughput_map(results: dict[str, Any], width: int = 78) -> str:
    """Format throughput map with route table and ASCII flow diagram."""
    lines = [
        f"\n{'[3.1] End-to-End Throughput Map':=^{width}}",
        "",
        f"  {'Route':>28}  {'Daily Qty':>14}  {'#Ship/day':>10}  {'Total Qty':>16}",
        f"  {'-'*28}  {'-'*14}  {'-'*10}  {'-'*16}",
    ]
    routes = results["routes"]
    for route_name, _, _ in ROUTE_TYPES:
        if route_name in routes:
            r = routes[route_name]
            lines.append(
                f"  {route_name:>28}  {r['daily_quantity']:>14,.0f}"
                f"  {r['daily_shipments']:>10,.0f}  {r['total_quantity']:>16,.0f}"
            )
    # Show any extra routes
    known = {name for name, _, _ in ROUTE_TYPES}
    for route_name, r in sorted(routes.items()):
        if route_name not in known:
            lines.append(
                f"  {route_name:>28}  {r['daily_quantity']:>14,.0f}"
                f"  {r['daily_shipments']:>10,.0f}  {r['total_quantity']:>16,.0f}"
            )

    # ASCII flow diagram
    fd = results["flow_data"]
    inv = fd["inv_by_echelon"]
    r = fd["routes"]

    def _route_daily(name: str) -> float:
        return r.get(name, {}).get("daily_quantity", 0)

    def _fmt_qty(q: float) -> str:
        if q >= _MILLION:
            return f"{q / _MILLION:.1f}M"
        if q >= _THOUSAND:
            return f"{q / _THOUSAND:.0f}K"
        return f"{q:.0f}"

    def _fmt_inv(q: float) -> str:
        if q >= _MILLION:
            return f"{q / _MILLION:.0f}M"
        if q >= _THOUSAND:
            return f"{q / _THOUSAND:.0f}K"
        return f"{q:.0f}"

    prod_d = fd["production_daily"]
    pos_d = fd["pos_daily"]
    ratio = fd["prod_demand_ratio"]
    ratio_s = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"

    to_rdc_d = _route_daily("Plant -> RDC")
    to_dc_d = _route_daily("Plant -> Customer DC")
    rdc_to_dc_d = _route_daily("RDC -> Customer DC")
    dc_to_store_d = _route_daily("Customer DC -> Store")
    dc_to_club_d = _route_daily("Customer DC -> Club")

    lines.append("\n  SUPPLY CHAIN FLOW MAP (daily averages)")
    lines.append(f"  {'=' * 50}")
    lines.append(
        f"  Prod: {_fmt_qty(prod_d)}/d  ---->"
        f"  [ PLANT ]  FG: {_fmt_inv(inv.get('Plant', 0))}"
    )
    lines.append(f"  (ratio: {ratio_s})")
    lines.append(
        "                     |"
        "                |"
    )
    lines.append(
        f"           {_fmt_qty(to_rdc_d)}/d v"
        f"      {_fmt_qty(to_dc_d)}/d v"
    )
    lines.append(
        "              [ RDC ]"
        "       [ Direct DC ]"
    )
    lines.append(
        f"              {_fmt_inv(inv.get('RDC', 0))} inv"
        f"       {_fmt_inv(inv.get('Customer DC', 0))} inv (total DC)"
    )
    lines.append(
        "                |"
    )
    lines.append(
        f"        {_fmt_qty(rdc_to_dc_d)}/d v"
    )
    lines.append(
        "          [ Customer DC ] <---- (combined)"
    )
    lines.append(
        "                |"
    )
    lines.append(
        f"  {_fmt_qty(dc_to_store_d + dc_to_club_d)}/d v"
    )
    lines.append(
        f"             [ STORE ]"
        f"  {_fmt_inv(inv.get('Store', 0) + inv.get('Club', 0))} inv"
    )
    lines.append(
        "                |"
    )
    lines.append(
        f"    {_fmt_qty(pos_d)}/d v"
    )
    lines.append(
        "              [ POS ]"
    )
    lines.append(f"  {'=' * 50}")

    return "\n".join(lines)


def format_deployment_effectiveness(
    results: dict[str, Any], width: int = 78
) -> str:
    """Format deployment effectiveness analysis."""
    lines = [
        f"\n{'[3.2] Deployment Effectiveness':=^{width}}",
        "",
    ]
    s = results["split"]
    lines.append("  Plant deployment split:")
    lines.append(f"    To RDC:         {s['to_rdc']:>14,.0f}  ({s['rdc_pct']:.1f}%)")
    lines.append(f"    To Customer DC: {s['to_dc']:>14,.0f}  ({s['dc_pct']:.1f}%)")
    lines.append(f"    Total deployed: {s['total_deployed']:>14,.0f}")

    gp = results["plant_inv_growth_pct"]
    gp_s = f"{gp:+.1f}%" if not np.isnan(gp) else "N/A"
    lines.append(f"\n  Plant FG inventory growth: {gp_s}")
    lines.append(f"  Plant retention rate: {results['retention_pct']:.1f}%")
    lines.append(f"  Total production: {results['total_production']:>14,.0f}")

    # Mini trend (sampled)
    trend = results["plant_inv_trend"]
    if len(trend) > _TREND_MIN_SAMPLES:
        lines.append("\n  Plant FG inventory trend (sampled):")
        step = max(1, len(trend) // _TREND_STEP_DIVISOR)
        for i in range(0, len(trend), step):
            t = trend[i]
            lines.append(f"    Day {t['day']:>4}: {t['inventory']:>14,.0f}")
        # Always show last
        t = trend[-1]
        lines.append(f"    Day {t['day']:>4}: {t['inventory']:>14,.0f}")

    return "\n".join(lines)


def format_lead_times(results: dict[str, Any], width: int = 78) -> str:
    """Format lead time analysis."""
    lines = [
        f"\n{'[3.3] Lead Time Analysis':=^{width}}",
        "",
        f"  {'Route':>28}  {'p50':>6}  {'p90':>6}  {'p99':>6}"
        f"  {'Mean':>6}  {'Config':>8}  {'#Ships':>10}",
        f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}",
    ]
    for route, r in sorted(results["by_route"].items()):
        cfg = r["configured_lt"]
        cfg_s = f"{cfg:.1f}d" if not np.isnan(cfg) else "N/A"
        lines.append(
            f"  {route:>28}  {r['p50']:>5.1f}d  {r['p90']:>5.1f}d"
            f"  {r['p99']:>5.1f}d  {r['mean']:>5.1f}d  {cfg_s:>8}"
            f"  {r['n_shipments']:>10,}"
        )
    return "\n".join(lines)


def format_bullwhip(results: dict[str, Any], width: int = 78) -> str:
    """Format bullwhip measurement."""
    lines = [
        f"\n{'[3.4] Bullwhip Measurement':=^{width}}",
        f"  POS demand: mean={results['pos_mean']:,.0f}/day"
        f"  CV={results['pos_cv']:.3f}",
        "",
        f"  {'Echelon':<14}  {'Order Mean':>14}  {'CV':>8}  {'Bullwhip':>10}",
        f"  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*10}",
    ]
    for ech in ECHELON_ORDER:
        if ech not in results["echelons"]:
            continue
        r = results["echelons"][ech]
        bw = r["bullwhip_ratio"]
        bw_s = f"{bw:.2f}x" if not np.isnan(bw) else "N/A"
        lines.append(
            f"  {ech:<14}  {r['daily_mean']:>14,.0f}  {r['cv']:>8.3f}  {bw_s:>10}"
        )

    mb = results["max_bullwhip"]
    verdict = (
        "OK" if (not np.isnan(mb) and mb < _BULLWHIP_THRESH)
        else "AMPLIFIED"
    )
    lines.append(
        f"\n  Max bullwhip ratio: "
        f"{(f'{mb:.2f}x' if not np.isnan(mb) else 'N/A')}"
        f"  (target: <2.0x)  Verdict: {verdict}"
    )
    return "\n".join(lines)


def format_control_stability(results: dict[str, Any], width: int = 78) -> str:
    """Format stability assessment."""
    lines = [
        f"\n{'[3.5] Control System Stability':=^{width}}",
        f"  Overall verdict: {results['overall_verdict']}",
        "  (based on last 180 days, linear regression slopes)",
        "",
        f"  {'Indicator':<24}  {'Last Value':>12}  {'Slope/day':>12}  {'Verdict':>12}",
        f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*12}",
    ]
    for name, ind in results["indicators"].items():
        last = ind["last_value"]
        slope = ind["slope_per_day"]
        if isinstance(last, float) and last > _LARGE_VALUE_THRESH:
            last_s = f"{last:,.0f}"
        else:
            last_s = f"{last:.4f}"
        lines.append(
            f"  {name:<24}  {last_s:>12}  {slope:>+12.6f}  {ind['verdict']:>12}"
        )
    return "\n".join(lines)
